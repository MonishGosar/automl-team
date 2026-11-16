"""Rebuild saved objects from their JSON, so a run can be picked up where it stopped.

Every artefact in a run folder is a dataclass written with ``asdict``. This turns
the JSON back into the same dataclasses, following the type hints: nested
dataclasses, lists, tuples, dicts and optional fields.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from typing import Any, TypeVar

T = TypeVar("T")


def from_dict(cls: type[T], data: Any) -> T:
    if not isinstance(data, dict):
        raise TypeError(f"{cls.__name__} needs a JSON object, got {type(data).__name__}.")
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for f in dataclasses.fields(cls):  # type: ignore[arg-type]
        if f.init and f.name in data:
            kwargs[f.name] = _convert(hints.get(f.name, Any), data[f.name])
    return cls(**kwargs)


def _convert(tp: Any, value: Any) -> Any:
    if value is None or tp is Any:
        return value
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin in (typing.Union, types.UnionType):
        options = [a for a in args if a is not type(None)]
        for option in options:
            try:
                return _convert(option, value)
            except (TypeError, ValueError, KeyError):
                continue
        return value
    if dataclasses.is_dataclass(tp):
        return from_dict(tp, value)  # type: ignore[arg-type]
    if origin is list:
        if not isinstance(value, list):
            raise TypeError("expected a list")
        return [_convert(args[0], v) for v in value] if args else list(value)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise TypeError("expected a list")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_convert(args[0], v) for v in value)
        return tuple(_convert(a, v) for a, v in zip(args, value, strict=False)) if args else tuple(value)
    if origin is dict:
        if not isinstance(value, dict):
            raise TypeError("expected an object")
        return {k: _convert(args[1], v) for k, v in value.items()} if args else dict(value)
    if tp in (int, float, str, bool) and not isinstance(value, (int, float, str, bool)):
        raise TypeError(f"expected {tp.__name__}")
    return value
