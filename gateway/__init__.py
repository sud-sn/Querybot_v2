"""
gateway/__init__.py

Adapter registry.
get_adapter(platform_type, credentials) returns the right adapter instance.
"""

from gateway.base import PlatformAdapter, PlatformEvent
from gateway.zoom_adapter import ZoomAdapter
from gateway.teams_adapter import TeamsAdapter
from gateway.slack_adapter import SlackAdapter

_ADAPTERS: dict[str, type[PlatformAdapter]] = {
    "zoom":  ZoomAdapter,
    "teams": TeamsAdapter,
    "slack": SlackAdapter,
}


def get_adapter(platform_type: str, credentials: dict) -> PlatformAdapter:
    """
    Instantiate the correct adapter for the given platform type.

    Usage:
        adapter = get_adapter("zoom", platform_config["credentials"])
        event   = adapter.parse_event(body, headers)
        await adapter.send_message(event, "Hello!")
    """
    cls = _ADAPTERS.get(platform_type)
    if cls is None:
        raise ValueError(
            f"Unknown platform_type: {platform_type!r}. "
            f"Supported: {list(_ADAPTERS)}"
        )
    return cls(credentials)


__all__ = [
    "PlatformAdapter", "PlatformEvent",
    "ZoomAdapter", "TeamsAdapter", "SlackAdapter",
    "get_adapter",
]
