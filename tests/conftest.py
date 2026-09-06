"""Shared pytest configuration.

Registers the ``local_llm`` marker used by the Increment 1 privacy tests that
exercise the REAL llama.cpp server on 127.0.0.1:3782. Those tests skip cleanly
when the server is not running; everything else in the suite stays offline
(SPEC: tests never hit real APIs).
"""

from __future__ import annotations


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "local_llm: integration test against the local llama.cpp server "
        "(skipped when it is not reachable)",
    )
