"""frisbee_analyzer — 极限飞盘比赛事件统计引擎。

设计依据 docs/2026-09-08-match-analysis-survey-and-plan.md §2.4/§4：
- 持盘判定：几何规则（盘-球员手部区域距离 + 盘速阈值 + 连续 N 帧惯性）
- 交换手 = 持盘人变化；攻守转换(turnover) = 跨队变化或盘落地
- 得分 = 持盘人位置落入对方得分区多边形（几何谓词，主判定）
- pull（开盘）重置攻防状态
所有坐标一律使用世界坐标（米），由上层经单应性映射提供。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

import numpy as np


class EventType(Enum):
    POSSESSION_START = "possession_start"
    POSSESSION_END = "possession_end"
    TRANSFER = "transfer"          # 交换手：持盘人 A -> B（同队或跨队）
    TURNOVER = "turnover"          # 攻守转换（跨队 transfer 或落地后被对方拾起）
    DISC_ON_GROUND = "disc_on_ground"
    SCORE = "score"                # 得分
    PULL = "pull"                  # 开盘重置


@dataclass
class Player:
    track_id: int
    team_id: Optional[int]      # 0/1，None=未知(裁判/未分队)
    x: float                    # 世界坐标（米）
    y: float


@dataclass
class DiscObservation:
    x: float
    y: float
    conf: float = 1.0
    frame: int = -1
    source: Optional[str] = None   # None=真实检出；"possession_imputed"=持盘补全合成（§13.14）


@dataclass
class Event:
    type: EventType
    frame: int
    from_track: Optional[int] = None
    to_track: Optional[int] = None
    team: Optional[int] = None
    x: float = 0.0
    y: float = 0.0
    detail: str = ""
    candidate: bool = False        # True=复核队列候选（不计入 score，如端区规则/源感知守卫）


@dataclass
class EndZone:
    """得分区多边形（世界坐标顶点，顺时针或逆时针均可）。"""
    team_attacking: int          # 哪个队攻这个区
    polygon: Sequence[tuple]

    def contains(self, x: float, y: float) -> bool:
        poly = np.asarray(self.polygon, dtype=float)
        inside = False
        j = len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
                inside = not inside
            j = i
        return inside


@dataclass
class PossessionConfig:
    hold_radius_m: float = 1.2        # 盘中心到球员"手部点"判定距离
    hand_height_ratio: float = 0.72   # 手部点在 bbox 顶部的比例（脚点 y 向上为 bbox 上端）
    max_hold_speed_ms: float = 6.0    # 盘速低于此值才可能被判"持盘"
    min_hold_frames: int = 5          # 候选持盘需连续 N 帧成立
    change_persist_frames: int = 4    # 换持盘人需新候选连续 M 帧（防抖惯性）
    ground_speed_ms: float = 1.5      # 低于此速且无人持有 → 视为落地
    ground_min_frames: int = 8        # 落地判定需要的持续帧数
    lose_hold_frames: int = 6         # 持盘判定连续丢失 N 帧后结束持有
    player_height_m: float = 1.8      # 用于从脚点推算手部点的身高假设
    imputed_score_guard_frames: int = 90  # 近期出现合成观测时，状态机得分降级 candidate 的保护窗（§13.14）
    score_cooloff_frames: int = 300   # 得分后冷却窗（§13.16 C0：引擎自动 pull 解锁；
                                      # 10s@30fps，防得分后庆祝走动被误选持盘）


@dataclass
class _Candidate:
    track_id: int
    streak: int = 0


class MatchEventEngine:
    """逐帧喂入球员与盘观测，输出事件流并维护比分/攻防状态。

    用法:
        eng = MatchEventEngine(end_zones=[...], team_of={track_id: team})
        for frame_idx, players, disc in video_stream:
            events = eng.update(frame_idx, players, disc)
    """

    def __init__(
        self,
        end_zones: Sequence[EndZone],
        team_of: Optional[dict] = None,
        config: PossessionConfig = PossessionConfig(),
        fps: float = 30.0,
    ) -> None:
        self.ezs = list(end_zones)
        self.team_of = dict(team_of or {})
        self.cfg = config
        self.fps = fps
        self.frame = -1
        self.holder: Optional[int] = None
        self._candidate: Optional[_Candidate] = None
        self._hold_miss = 0
        self._ground_streak = 0
        self._prev_disc: Optional[DiscObservation] = None
        self._last_holder_team: Optional[int] = None   # 最近一次持有的队伍（落地后被拾起时判 turnover）
        self._on_ground = False
        self.score = {0: 0, 1: 0}
        self.last_events: list[Event] = []
        self.events: list[Event] = []
        # 攻防方向: end_zones[i].team_attacking 即该区被哪个队攻击
        self._score_lock = False
        self._imputed_seen_frame: Optional[int] = None  # 最近一次合成盘观测的帧号（源感知记分门）

    # ---------- 对外主入口 ----------
    def update(self, frame: int, players: Sequence[Player], disc: Optional[DiscObservation]) -> list[Event]:
        self.frame = frame
        self.last_events = []

        # 得分后到 pull 之间为死球：不进行持有选举/落地判定
        if self._score_lock:
            if disc is not None:
                self._prev_disc = disc
            return self.last_events

        disc_speed = self._disc_speed(disc)

        # 源感知记分门：记录最近一次合成观测帧号（§13.14 实测补全可经级联伪造记分）
        if disc is not None and disc.source == "possession_imputed":
            self._imputed_seen_frame = frame

        # 1) 候选持盘人（最近手部点 + 盘速约束）
        cand_id, cand_dist = self._nearest_hand(players, disc) if disc else (None, None)
        cand_ok = (
            cand_id is not None
            and cand_dist is not None
            and cand_dist <= self.cfg.hold_radius_m
            and disc_speed <= self.cfg.max_hold_speed_ms
        )

        # 2) 持盘人状态机
        if self.holder is None:
            self._elect_holder(cand_id if cand_ok else None, disc)
        else:
            self._maintain_holder(cand_id, cand_ok, players, disc)

        # 3) 落地检测（无持有 + 低速持续）
        self._check_ground(disc, disc_speed, players)

        # 4) 得分检测（持盘人位于其进攻端区内）
        self._check_score(players)

        if disc is not None:
            self._prev_disc = disc
        self.events.extend(self.last_events)
        return self.last_events

    # ---------- 内部 ----------
    def _disc_speed(self, disc: Optional[DiscObservation]) -> float:
        if disc is None or self._prev_disc is None or self.frame - self._prev_disc.frame > 2:
            return 0.0
        dt = max((disc.frame - self._prev_disc.frame) / self.fps, 1e-6)
        return float(np.hypot(disc.x - self._prev_disc.x, disc.y - self._prev_disc.y) / dt)

    def _nearest_hand(self, players: Sequence[Player], disc: DiscObservation):
        best_id, best_d = None, None
        for p in players:
            if p.team_id is not None and p.team_id < 0:
                continue  # 负 team_id 表示非球员(裁判/观众)
            hx, hy = self._hand_point(p)
            d = float(np.hypot(disc.x - hx, disc.y - hy))
            if best_d is None or d < best_d:
                best_id, best_d = p.track_id, d
        return best_id, best_d

    def _hand_point(self, p: Player) -> tuple:
        """Player.x/y 为脚点；手部点按身高比例上移。"""
        return p.x, p.y + self.cfg.player_height_m * self.cfg.hand_height_ratio

    def _elect_holder(self, cand_id: Optional[int], disc: Optional[DiscObservation]) -> None:
        if cand_id is None:
            self._candidate = None
            return
        if self._candidate and self._candidate.track_id == cand_id:
            self._candidate.streak += 1
        else:
            self._candidate = _Candidate(track_id=cand_id, streak=1)
        if self._candidate.streak >= self.cfg.min_hold_frames:
            self.holder = cand_id
            self._candidate = None
            self._hold_miss = 0
            self._emit(Event(EventType.POSSESSION_START, self.frame, to_track=cand_id,
                             x=disc.x if disc else 0, y=disc.y if disc else 0))
            new_team = self.team_of.get(cand_id)
            if self._on_ground and self._last_holder_team is not None \
                    and new_team is not None and new_team != self._last_holder_team:
                # 落地盘被对方拾起 = 攻守转换（同队捡回不算）
                self._emit(Event(EventType.TURNOVER, self.frame, to_track=cand_id,
                                 team=new_team, x=disc.x if disc else 0, y=disc.y if disc else 0,
                                 detail="ground_recovery"))
            self._on_ground = False

    def _maintain_holder(self, cand_id: Optional[int], cand_ok: bool,
                         players: Sequence[Player], disc: Optional[DiscObservation]) -> None:
        if cand_ok and cand_id == self.holder:
            self._hold_miss = 0
            self._candidate = None
            return
        if cand_ok and cand_id != self.holder:
            # 换人候选：需连续 change_persist_frames 帧才切换
            if self._candidate and self._candidate.track_id == cand_id:
                self._candidate.streak += 1
            else:
                self._candidate = _Candidate(track_id=cand_id, streak=1)
            if self._candidate.streak >= self.cfg.change_persist_frames:
                self._emit(Event(EventType.POSSESSION_END, self.frame, from_track=self.holder,
                                 x=disc.x if disc else 0, y=disc.y if disc else 0))
                self._emit_transfer(self.holder, cand_id, disc)
                self.holder = cand_id
                self._candidate = None
                self._hold_miss = 0
            return
        # 无有效候选：丢失计数
        self._hold_miss += 1
        if self._hold_miss >= self.cfg.lose_hold_frames:
            self._emit(Event(EventType.POSSESSION_END, self.frame, from_track=self.holder,
                             x=disc.x if disc else 0, y=disc.y if disc else 0,
                             detail="lost"))
            self._last_holder_team = self.team_of.get(self.holder)
            self.holder = None
            self._candidate = None
            self._hold_miss = 0

    def _check_ground(self, disc: Optional[DiscObservation], speed: float,
                      players: Sequence[Player]) -> None:
        if disc is None or self.holder is not None:
            self._ground_streak = 0
            return
        if speed <= self.cfg.ground_speed_ms:
            self._ground_streak += 1
            if self._ground_streak == self.cfg.ground_min_frames:
                self._on_ground = True
                self._emit(Event(EventType.DISC_ON_GROUND, self.frame, x=disc.x, y=disc.y))
        else:
            self._ground_streak = 0

    def _check_score(self, players: Sequence[Player]) -> None:
        if self.holder is None or self._score_lock:
            return
        p = next((q for q in players if q.track_id == self.holder), None)
        if p is None:
            return
        team = self.team_of.get(p.track_id)
        if team is None:
            return
        for ez in self.ezs:
            if ez.team_attacking == team and ez.contains(p.x, p.y):
                # 源感知记分门：保护窗内有合成观测 → 降级为候选，不计入 score。
                # 实测（§13.14 chunk2）：少量补全观测经融合级联即可把轨迹带到错误
                # 位置伪造记分；候选仍进复核队列（candidate=True）。
                guarded = (self._imputed_seen_frame is not None
                           and self.frame - self._imputed_seen_frame
                           <= self.cfg.imputed_score_guard_frames)
                self._score_lock = True
                self.holder = None
                self._ground_streak = 0
                if guarded:
                    self._emit(Event(EventType.SCORE, self.frame, to_track=p.track_id, team=team,
                                     x=p.x, y=p.y, candidate=True,
                                     detail=f"imputed-guard ({self.frame - self._imputed_seen_frame}f) "
                                            f"— review, not counted"))
                else:
                    self.score[team] += 1
                    self._emit(Event(EventType.SCORE, self.frame, to_track=p.track_id, team=team,
                                     x=p.x, y=p.y, detail=f"score={dict(self.score)}"))
                    # USAU 9.B（§13.16 B）：计入比分的得分后两队进攻端区立即互换
                    if len(self.ezs) == 2:
                        self.ezs[0].team_attacking, self.ezs[1].team_attacking = \
                            self.ezs[1].team_attacking, self.ezs[0].team_attacking
                return

    def notify_pull(self, frame: int, x: float = 0.0, y: float = 0.0) -> Event:
        """外部（长传开盘检测或人工）通知一次 pull，解锁得分锁并重置持有。"""
        self.holder = None
        self._candidate = None
        self._hold_miss = 0
        self._ground_streak = 0
        self._score_lock = False
        self._imputed_seen_frame = None  # pull 是可信外部信号，清除源守卫
        ev = Event(EventType.PULL, frame, x=x, y=y)
        self._emit(ev)
        return ev

    def set_team(self, track_id: int, team_id: Optional[int]) -> None:
        self.team_of[track_id] = team_id

    def _emit_transfer(self, old: Optional[int], new: int, disc: Optional[DiscObservation]) -> None:
        old_team = self.team_of.get(old) if old is not None else None
        new_team = self.team_of.get(new)
        cross = old_team is not None and new_team is not None and old_team != new_team
        ev = Event(EventType.TRANSFER, self.frame, from_track=old, to_track=new, team=new_team,
                   x=disc.x if disc else 0, y=disc.y if disc else 0)
        self._emit(ev)
        if cross:
            self._emit(Event(EventType.TURNOVER, self.frame, from_track=old, to_track=new,
                             team=new_team, x=disc.x if disc else 0, y=disc.y if disc else 0))

    def _emit(self, ev: Event) -> None:
        self.last_events.append(ev)
