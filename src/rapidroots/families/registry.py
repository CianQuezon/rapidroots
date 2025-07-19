from .base import FunctionFamily
from typing import Dict


_registry: Dict[str, FunctionFamily] = {}
_registry_version: int = 0

def register_family(family: FunctionFamily) -> None:
    global _registry_version
    _registry[family.name] = family
    _registry_version += 1

def get_family(name: str) -> FunctionFamily:
    return _registry[name]

def all_families() -> Dict[str, FunctionFamily]:
    return dict(_registry)

def registry_version() -> int:
    return _registry_version