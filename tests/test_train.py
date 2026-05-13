import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.train import train_frisbee_detector


def test_freeze_passed_to_model_train():
    with patch("models.train.YOLO") as MockYOLO:
        mock_model = MagicMock()
        mock_results = MagicMock()
        mock_results.save_dir = "/tmp/test_run"
        mock_model.train.return_value = mock_results
        MockYOLO.return_value = mock_model

        train_frisbee_detector(
            data_yaml="fake.yaml",
            freeze=10,
            resume_from=None,
        )

        call_kwargs = mock_model.train.call_args[1]
        assert call_kwargs["freeze"] == 10


def test_freeze_none_not_passed():
    with patch("models.train.YOLO") as MockYOLO:
        mock_model = MagicMock()
        mock_results = MagicMock()
        mock_results.save_dir = "/tmp/test_run"
        mock_model.train.return_value = mock_results
        MockYOLO.return_value = mock_model

        train_frisbee_detector(
            data_yaml="fake.yaml",
            freeze=None,
            resume_from=None,
        )

        call_kwargs = mock_model.train.call_args[1]
        assert "freeze" not in call_kwargs or call_kwargs.get("freeze") is None


def test_cls_passed_to_model_train():
    with patch("models.train.YOLO") as MockYOLO:
        mock_model = MagicMock()
        mock_results = MagicMock()
        mock_results.save_dir = "/tmp/test_run"
        mock_model.train.return_value = mock_results
        MockYOLO.return_value = mock_model

        train_frisbee_detector(
            data_yaml="fake.yaml",
            cls=0.8,
            resume_from=None,
        )

        call_kwargs = mock_model.train.call_args[1]
        assert call_kwargs["cls"] == 0.8
