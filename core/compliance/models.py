from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ResourceRef:
    table: str
    column: str = ""
    output_alias: str = ""

    @property
    def key(self) -> str:
        table = self.table.upper()
        return f"{table}.{self.column.upper()}" if self.column else table


@dataclass
class PolicyContext:
    account_id: str
    user_id: str = ""
    groups: list[str] = field(default_factory=list)
    purpose_id: str = ""
    channel: str = "portal"
    action: str = "query_execution"
    policy_version: int = 0
    break_glass_grant_id: str | None = None
    role: str = "analyst"
    provider: str = ""
    user_attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyDecision:
    allowed: bool
    reason_code: str
    permitted_resources: list[ResourceRef] = field(default_factory=list)
    row_obligations: list[dict] = field(default_factory=list)
    masking: dict[str, str] = field(default_factory=dict)
    aggregate_only: list[ResourceRef] = field(default_factory=list)
    export_allowed: bool = False
    cache_ttl_seconds: int = 0
    policy_version: int = 0
    audit_id: str = ""
    shadow: bool = False
    explanation: str = ""

    @property
    def effective_allowed(self) -> bool:
        return self.allowed or self.shadow
