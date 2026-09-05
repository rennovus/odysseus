"""An unconfigured Default Chat Model must render as unset, not as a model.

The Default Chat Model selects had no blank option and were filled with
keepBlank=false, so with nothing persisted the browser auto-selected option[0].
Because the list is sorted by sortModelIds(), Settings reported the
alphabetically-first model as the configured default while the composer
resolved something else entirely — the endpoint's first model as returned by
the provider. Settings must show the unset state instead.
"""

from pathlib import Path

import pytest
from bs4 import BeautifulSoup


_REPO = Path(__file__).resolve().parents[1]


def _default_chat_source() -> str:
    source = (_REPO / "static" / "js" / "settings.js").read_text(encoding="utf-8")
    start = source.index("async function initDefaultChat()")
    end = source.index("/* ── Utility Model ── */", start)
    return source[start:end]


def _soup() -> BeautifulSoup:
    return BeautifulSoup(
        (_REPO / "static" / "index.html").read_text(encoding="utf-8"),
        "html.parser",
    )


@pytest.mark.parametrize("select_id", ["set-defaultEpSelect", "set-defaultModelSelect"])
def test_default_chat_selects_offer_an_explicit_unset_option(select_id):
    select = _soup().find(id=select_id)
    assert select is not None, f"{select_id} missing from settings markup"

    options = select.find_all("option")
    assert options, f"{select_id} needs a placeholder option for the unset state"

    placeholder = options[0]
    assert placeholder.get("value", "") == "", (
        f"{select_id} option[0] must be the empty-valued unset placeholder, so a "
        "browser with nothing persisted cannot report a real model as configured"
    )
    assert placeholder.get_text(strip=True), (
        f"{select_id} placeholder needs visible text naming the unset state"
    )


@pytest.mark.parametrize("fill_call", ["_fillModelSelect", "_fillEndpointSelect"])
def test_default_chat_fills_preserve_the_unset_option(fill_call):
    source = _default_chat_source()
    calls = [
        line.strip() for line in source.splitlines()
        if fill_call + "(" in line
    ]
    assert calls, f"initDefaultChat no longer calls {fill_call}"
    for call in calls:
        assert not call.rstrip().endswith("false);"), (
            f"{fill_call} must be called with keepBlank=true in initDefaultChat; "
            "dropping the blank option lets option[0] of the sorted list be "
            f"auto-selected as the default. Offending call: {call}"
        )


def test_default_chat_does_not_persist_on_load():
    # The displayed value is only a default *proposal* until the user picks one.
    # Saving on load would turn the auto-selected option[0] into a real setting.
    source = _default_chat_source()
    load_block = source[:source.index("epSel.addEventListener")]
    assert "saveDefault()" not in load_block, (
        "initDefaultChat must not save while loading persisted settings"
    )
