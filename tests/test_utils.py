"""Tests for shared dataset utilities."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset import generate_yaml_config, split_items


def test_generate_yaml_config():
    content = generate_yaml_config("/tmp/test", "images/train", "images/val", "images/test")
    assert "path: /tmp/test" in content
    assert "train: images/train" in content
    assert "val: images/val" in content
    assert "test: images/test" in content
    assert "nc: 1" in content
    assert "frisbee" in content


def test_generate_yaml_config_custom():
    content = generate_yaml_config(
        "/a/b", "img/a", "img/b", "img/c", nc=3, names=["cat", "dog", "bird"]
    )
    assert "path: /a/b" in content
    assert "nc: 3" in content
    assert "cat" in content
    assert "dog" in content
    assert "bird" in content


def test_split_items_basic():
    items = list(range(100))
    splits = split_items(items, seed=42)
    assert len(splits["train"]) == 80
    assert len(splits["val"]) == 10
    assert len(splits["test"]) == 10
    # Verify no overlap
    all_items = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
    assert len(all_items) == 100


def test_split_items_reproducible():
    items = list(range(50))
    s1 = split_items(items, seed=123)
    s2 = split_items(items, seed=123)
    assert s1["train"] == s2["train"]


def test_split_items_different_seeds():
    items = list(range(100))
    s1 = split_items(items, seed=1)
    s2 = split_items(items, seed=999)
    assert s1["train"] != s2["train"]
