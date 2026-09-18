from __future__ import annotations

import pytest

# streamlit ships in the "dashboard" uv dependency group (pyproject.toml),
# included by [tool.uv] default-groups = "all" but still skippable via
# `uv sync --no-default-groups`. Guard the whole directory so that case skips
# cleanly instead of every test file failing collection.
pytest.importorskip("streamlit", reason="dashboard tests require the optional streamlit dependency group")
