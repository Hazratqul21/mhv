from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from app.utils.logger import get_logger

log = get_logger(__name__)


class MessageType(str, Enum):
    REQUEST = "request"
    RESPONSE = "response"
    DELEGATE = "delegate"
    BROADCAST = "broadcast"
    STATUS = "status"
    ERROR = "error"


@dataclass
class AgentMessage:
    """Protocol message for inter-agent communication."""
    id: str = field(default_factory=lambda: uuid4().hex[:12])
    sender: str = ""
    receiver: str = ""
    msg_type: MessageType = MessageType.REQUEST
    content: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    priority: int = 0
    ttl: int = 300  # seconds before message expires

    @property
    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl

    def reply(self, content: str, payload: dict | None = None) -> AgentMessage:
        return AgentMessage(
            sender=self.receiver,
            receiver=self.sender,
            msg_type=MessageType.RESPONSE,
            content=content,
            payload=payload or {},
            parent_id=self.id,
        )

    def delegate(self, target: str, content: str, payload: dict | None = None) -> AgentMessage:
        return AgentMessage(
            sender=self.receiver,
            receiver=target,
            msg_type=MessageType.DELEGATE,
            content=content,
            payload={**(payload or {}), "original_sender": self.sender},
            parent_id=self.id,
        )


class AgentProtocol:
    """Message bus enabling agent-to-agent communication.

    Agents register mailboxes and can send/receive typed messages.
    Supports request-response, delegation, and broadcast patterns.
    """

    def __init__(self) -> None:
        self._mailboxes: dict[str, asyncio.Queue[AgentMessage]] = {}
        self._history: list[AgentMessage] = []
        self._max_history = 5000
        self._handlers: dict[str, Any] = {}

    def register(self, agent_name: str) -> None:
        if agent_name not in self._mailboxes:
            self._mailboxes[agent_name] = asyncio.Queue(maxsize=100)
            log.info("agent_protocol_registered", agent=agent_name)

    def unregister(self, agent_name: str) -> None:
        self._mailboxes.pop(agent_name, None)

    async def send(self, message: AgentMessage) -> None:
        if message.is_expired:
            log.warning("message_expired", msg_id=message.id)
            return

        self._history.append(message)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        if message.msg_type == MessageType.BROADCAST:
            for name, mailbox in self._mailboxes.items():
                if name != message.sender:
                    try:
                        mailbox.put_nowait(message)
                    except asyncio.QueueFull:
                        log.warning("mailbox_full", agent=name)
        else:
            mailbox = self._mailboxes.get(message.receiver)
            if mailbox is None:
                log.warning("agent_not_found", receiver=message.receiver)
                return
            try:
                mailbox.put_nowait(message)
            except asyncio.QueueFull:
                log.warning("mailbox_full", agent=message.receiver)

        log.debug(
            "message_sent",
            msg_id=message.id,
            sender=message.sender,
            receiver=message.receiver,
            type=message.msg_type.value,
        )

    async def receive(self, agent_name: str, timeout: float = 30.0) -> AgentMessage | None:
        mailbox = self._mailboxes.get(agent_name)
        if mailbox is None:
            return None
        try:
            return await asyncio.wait_for(mailbox.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def peek(self, agent_name: str) -> int:
        mailbox = self._mailboxes.get(agent_name)
        return mailbox.qsize() if mailbox else 0

    async def request_response(
        self, sender: str, receiver: str, content: str, payload: dict | None = None, timeout: float = 60.0,
    ) -> AgentMessage | None:
        msg = AgentMessage(
            sender=sender, receiver=receiver,
            msg_type=MessageType.REQUEST,
            content=content, payload=payload or {},
        )
        await self.send(msg)

        deadline = time.time() + timeout
        while time.time() < deadline:
            reply = await self.receive(sender, timeout=min(5.0, deadline - time.time()))
            if reply and reply.parent_id == msg.id:
                return reply
        return None

    def get_conversation(self, msg_id: str) -> list[AgentMessage]:
        chain: list[AgentMessage] = []
        ids = {msg_id}
        for msg in self._history:
            if msg.id in ids or msg.parent_id in ids:
                chain.append(msg)
                ids.add(msg.id)
        return sorted(chain, key=lambda m: m.timestamp)

    @property
    def registered_agents(self) -> list[str]:
        return list(self._mailboxes.keys())
