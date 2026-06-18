"""Stdlib interactive prompt helpers for the remote_handler CLI.

Every prompt takes an injectable ``ask`` callable (defaulting to ``input``)
so the CLI flows can be unit-tested by scripting answers instead of
monkeypatching builtins.
"""

from typing import Callable, Optional, Sequence

Reader = Callable[[str], str]


def prompt_text(
    label: str,
    default: Optional[str] = None,
    ask: Reader = input,
) -> str:
    """Prompt for free text.

    Empty input returns ``default`` (or "" when there is no default). The
    default, if any, is shown in brackets.
    """
    suffix = f" [{default}]" if default is not None else ""
    answer = ask(f"{label}{suffix}: ").strip()
    if not answer:
        return default if default is not None else ""
    return answer


def prompt_choice(
    label: str,
    choices: Sequence[str],
    default: Optional[str] = None,
    allow_custom: bool = True,
    ask: Reader = input,
) -> str:
    """Show a numbered pick-list and return the chosen value.

    Accepts a 1-based index, a literal value from ``choices``, or (when
    ``allow_custom``) any typed string. Empty input returns ``default`` if
    set. Re-prompts on invalid input.
    """
    for i, choice in enumerate(choices, start=1):
        print(f"  {i}) {choice}")
    if allow_custom:
        print("  (or type a custom value)")

    suffix = f" [{default}]" if default is not None else ""
    while True:
        answer = ask(f"{label}{suffix}: ").strip()
        if not answer:
            if default is not None:
                return default
            print("A value is required.")
            continue
        if answer.isdigit():
            idx = int(answer)
            if 1 <= idx <= len(choices):
                return choices[idx - 1]
            print(f"Pick a number between 1 and {len(choices)}.")
            continue
        if answer in choices or allow_custom:
            return answer
        print("Not a listed option; pick a number or listed value.")


def prompt_yes_no(
    label: str,
    default: bool = False,
    ask: Reader = input,
) -> bool:
    """Yes/no confirmation. Empty input returns ``default``."""
    hint = "[Y/n]" if default else "[y/N]"
    while True:
        answer = ask(f"{label} {hint}: ").strip().lower()
        if not answer:
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Please answer y or n.")
