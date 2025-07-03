from dataclasses import dataclass, field

from plugins.class_registry import BasePlugin

@dataclass(frozen=True)
class Component:
    component_typename: str
    component_name: str
    component_values: list[str]

