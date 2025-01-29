import os
from dataclasses import dataclass
from typing import Optional

from ai_prepline.base.config import ModelConfig


def test_base_model_config_data():
    cfg = ModelConfig(name="test", author="test")
    assert cfg.name == "test"
    assert cfg.author == "test"
    assert cfg.comments == ""
    assert cfg.as_dict() == {"name": "test", "author": "test", "comments": ""}


def test_base_model_config_subclass():
    @dataclass
    class MyModelConfig(ModelConfig):
        test_field: Optional[str] = ""

    cfg = MyModelConfig(
        name="test", author="test", test_field="test", comments="test comment"
    )
    assert cfg.name == "test"
    assert cfg.author == "test"
    assert cfg.test_field == "test"
    assert cfg.comments == "test comment"
    assert cfg.as_dict() == {
        "name": "test",
        "author": "test",
        "test_field": "test",
        "comments": "test comment",
    }


def test_model_config_save_load_json(tmp_path):
    cfg_path = os.path.join(tmp_path, "cfg.json")
    cfg = ModelConfig(name="test", author="test")
    cfg.save_json(cfg_path)

    read_cfg = ModelConfig.load_json(cfg_path)
    assert cfg.as_dict() == read_cfg.as_dict()
