"""Support code for the Remember blackbox system tests.

Nothing here is imported by the product. The package exists so the tests
themselves stay short: they declare what the LLM says, then talk to a real
server through the public client.
"""

from .reflect import reflect_loop
from .rulebook import ChatRequest, LLMStub, Stubs
from .server import (
    RememberServer,
    StubServer,
    run_remember_admin,
    start_remember_server,
    start_remember_worker,
    start_stub_server,
)
from .waiting import wait_until_settled

__all__ = [
    "ChatRequest",
    "RememberServer",
    "LLMStub",
    "StubServer",
    "Stubs",
    "start_remember_server",
    "start_remember_worker",
    "reflect_loop",
    "run_remember_admin",
    "start_stub_server",
    "wait_until_settled",
]
