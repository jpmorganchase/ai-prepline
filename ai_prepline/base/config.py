import json
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """
    Base model config - extend and add model specific configuration params

    Usage:

        >>> cfg = ModelConfig(name="my-model")
        >>> cfg.load_json('/read/from/config.json')
        >>> cfg.save_json('/save/to/config.json')
    """

    name: Optional[str] = ""
    author: Optional[str] = ""
    comments: Optional[str] = ""

    def as_dict(self) -> dict:
        """
        Return configuration data as dictionary

        """
        return asdict(self)

    @classmethod
    def load_json(cls, path: str) -> "ModelConfig":
        """
        Load configuration from a json file

        :param path: path to json file.
        """
        with open(path, "r", encoding="utf-8") as fp:
            config_dict = json.load(fp)
        return cls(**config_dict)

    def save_json(self, path: str):
        """
        Save configuration as json file

        :param path: path to json file.
        """
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(self.as_dict(), fp)
