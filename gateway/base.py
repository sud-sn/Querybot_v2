"""
gateway/base.py

Defines the normalised internal event that all platform adapters produce.
The bot core (main.py) only ever sees PlatformEvent — never raw Zoom/Teams/Slack payloads.

Every platform adapter must:
  1. verify_request()  — validate the incoming webhook signature
  2. parse_event()     — convert the raw payload to a PlatformEvent
  3. send_message()    — post a reply back to the platform
  4. upload_file()     — send a file (chart PNG) back to the platform
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from abc import ABC, abstractmethod


@dataclass
class PlatformEvent:
    """
    Normalised message event — same shape regardless of platform.

    account_id  : unique workspace/tenant identifier
                  Zoom  → accountId
                  Teams → tenantId
                  Slack → team_id (workspace ID)

    user_id     : the person who sent the message
                  Zoom  → userId
                  Teams → Activity.from.id
                  Slack → event.user

    channel_id  : where to send the reply
                  Zoom  → toJid
                  Teams → Activity.conversation.id
                  Slack → event.channel

    text        : the raw message text / command

    platform    : "zoom" | "teams" | "slack"

    raw         : the original parsed payload (for platform-specific edge cases)
    """
    account_id  : str
    user_id     : str
    channel_id  : str
    text        : str
    platform    : str
    raw         : dict = field(default_factory=dict)
    # Optional FQN hint from suggested-question clicks (DB.SCHEMA.TABLE).
    table_hint  : str = ""
    # Optional schema name selected in the portal UI (e.g. "HR").
    # When set, the query pipeline scopes RAG retrieval and SQL generation
    # to only the tables belonging to this schema.
    schema_hint : str = ""


class PlatformAdapter(ABC):
    """
    Abstract base class for all platform adapters.
    Each adapter wraps one specific chat platform.
    """

    platform_type: str  # "zoom" | "teams" | "slack"

    def __init__(self, credentials: dict):
        """
        credentials: the decrypted dict from platform_config.credentials
        """
        self.credentials = credentials

    @abstractmethod
    async def verify_request(
        self,
        body: bytes,
        headers: dict,
    ) -> bool:
        """
        Verify the webhook signature.
        Returns True if authentic, False if rejected.
        """
        ...

    @abstractmethod
    def parse_event(self, body: bytes, headers: dict) -> Optional[PlatformEvent]:
        """
        Parse a raw webhook payload into a PlatformEvent.
        Returns None if the payload is not a user message (e.g. presence events).
        """
        ...

    @abstractmethod
    async def send_message(
        self,
        event: PlatformEvent,
        text: str,
    ) -> None:
        """Send a text reply back to the originating channel."""
        ...

    @abstractmethod
    async def upload_file(
        self,
        event: PlatformEvent,
        file_bytes: bytes,
        filename: str,
        mime_type: str = "image/png",
    ) -> None:
        """Upload a file (chart PNG) to the originating channel."""
        ...
