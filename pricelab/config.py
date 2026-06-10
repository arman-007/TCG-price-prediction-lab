"""Configuration loading.

YAML -> nested mapping with attribute access. Sections are intentionally
permissive (unknown keys pass through) so experiment configs can carry
model-specific knobs without code changes. Research code churns too fast
for rigid schemas; correctness is enforced where it matters (tests on the
data/eval layers), not in config plumbing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Section(dict):
    """dict with recursive attribute access: cfg.backtest.n_cutoffs."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    @classmethod
    def wrap(cls, obj: Any) -> Any:
        if isinstance(obj, dict):
            return cls({k: cls.wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [cls.wrap(v) for v in obj]
        return obj


def load_config(path: str | Path, overrides: list[str] | None = None) -> Section:
    """Load a YAML config, then apply dotted-path overrides like
    ``backtest.n_cutoffs=6`` (values parsed as YAML scalars)."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = Section.wrap(raw)
    for item in overrides or []:
        key, _, value = item.partition("=")
        set_dotted(cfg, key.strip(), yaml.safe_load(value))
    return cfg


def set_dotted(cfg: Section, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = Section()
        node = node[part]
    node[parts[-1]] = Section.wrap(value)


def merge_into(cfg: Section, extra: dict) -> None:
    """Recursively merge ``extra`` into ``cfg`` (extra wins on conflicts)."""
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            merge_into(cfg[key], value)
        else:
            cfg[key] = Section.wrap(value)


def dump_config(cfg: Section, path: str | Path) -> None:
    def plain(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: plain(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [plain(v) for v in obj]
        return obj

    Path(path).write_text(yaml.safe_dump(plain(cfg), sort_keys=False))
