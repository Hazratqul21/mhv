"""Quick voice commands that execute instantly without LLM.

Pattern-matched commands for common tasks where calling the LLM would
add unnecessary latency.
"""

from __future__ import annotations

import datetime
import logging
import platform
import re
import subprocess
from dataclasses import dataclass
from typing import Callable

log = logging.getLogger("miya.voice.commands")


@dataclass
class CommandRule:
    patterns: list[re.Pattern]
    handler: Callable[[], str]
    description: str


def _time_now() -> str:
    now = datetime.datetime.now()
    return f"Hozir soat {now.strftime('%H:%M')}"


def _date_today() -> str:
    today = datetime.date.today()
    return f"Bugun {today.strftime('%Y-yil %d-%B, %A')}"


def _open_browser() -> str:
    try:
        if platform.system() == "Linux":
            subprocess.Popen(["xdg-open", "https://www.google.com"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", "https://www.google.com"])
        else:
            subprocess.Popen(["start", "https://www.google.com"], shell=True)
        return "Brauzer ochildi"
    except Exception as e:
        log.error("failed to open browser: %s", e)
        return "Brauzerni ochib bo'lmadi"


def _open_file_manager() -> str:
    try:
        if platform.system() == "Linux":
            subprocess.Popen(["xdg-open", "."], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", "."])
        else:
            subprocess.Popen(["explorer", "."])
        return "Fayl menejeri ochildi"
    except Exception as e:
        log.error("failed to open file manager: %s", e)
        return "Fayl menejerni ochib bo'lmadi"


def _lock_screen() -> str:
    try:
        if platform.system() == "Linux":
            subprocess.Popen(["xdg-screensaver", "lock"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return "Ekran qulflandi"
    except Exception as e:
        log.error("failed to lock screen: %s", e)
        return "Ekranni qulflab bo'lmadi"


def _system_info() -> str:
    import psutil

    cpu = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory()
    return (
        f"CPU: {cpu}%, RAM: {mem.percent}% "
        f"({mem.used // (1024**3)}/{mem.total // (1024**3)} GB)"
    )


class QuickCommands:
    """Registry of quick voice commands."""

    def __init__(self) -> None:
        self._rules: list[CommandRule] = self._default_rules()

    def _default_rules(self) -> list[CommandRule]:
        def _patterns(*raw: str) -> list[re.Pattern]:
            return [re.compile(p, re.IGNORECASE) for p in raw]

        return [
            CommandRule(
                patterns=_patterns(
                    r"\bsoat\s+nech",
                    r"\bwhat\s+time",
                    r"\btime\b",
                    r"\bvaqt\b",
                ),
                handler=_time_now,
                description="Hozirgi vaqt",
            ),
            CommandRule(
                patterns=_patterns(
                    r"\bbugun\s+qanday\s+kun",
                    r"\bwhat\s+day",
                    r"\bsana\b",
                    r"\bdate\b",
                ),
                handler=_date_today,
                description="Bugungi sana",
            ),
            CommandRule(
                patterns=_patterns(
                    r"\bbrauzer\s+och",
                    r"\bchrome\s+och",
                    r"\bopen\s+browser",
                    r"\bopen\s+chrome",
                ),
                handler=_open_browser,
                description="Brauzerni ochish",
            ),
            CommandRule(
                patterns=_patterns(
                    r"\bfayl\b.*\boch",
                    r"\bfile\s+manager",
                    r"\bfayl\s+menej",
                ),
                handler=_open_file_manager,
                description="Fayl menejerni ochish",
            ),
            CommandRule(
                patterns=_patterns(
                    r"\bekran\b.*\bqulf",
                    r"\block\s+screen",
                    r"\bekran\b.*\bo'chir",
                ),
                handler=_lock_screen,
                description="Ekranni qulflash",
            ),
            CommandRule(
                patterns=_patterns(
                    r"\bsistema\s+holat",
                    r"\bsystem\s+status",
                    r"\bcpu\b",
                    r"\bram\b",
                ),
                handler=_system_info,
                description="Tizim holati",
            ),
        ]

    def match(self, text: str) -> str | None:
        """Try to match *text* against quick commands.

        Returns the response string if matched, None otherwise.
        """
        for rule in self._rules:
            for pat in rule.patterns:
                if pat.search(text):
                    log.info("quick command matched: %s", rule.description)
                    try:
                        return rule.handler()
                    except Exception as e:
                        log.error("quick command failed: %s", e)
                        return f"Buyruqni bajarib bo'lmadi: {e}"
        return None

    def add(self, patterns: list[str], handler: Callable[[], str], description: str) -> None:
        compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
        self._rules.append(CommandRule(compiled, handler, description))

    def list_commands(self) -> list[str]:
        return [r.description for r in self._rules]
