import json
import pathlib

src = json.loads(
    pathlib.Path.home().joinpath("social-auto-upload/cookies/bilibili_diyi.json").read_text()
)
parts = ["{}={}".format(c["name"], c["value"]) for c in src["cookie_info"]["cookies"]]
print("; ".join(parts))
