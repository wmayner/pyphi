"""Registration of the PyPhi MCP server with a client, and of the PyPhi facts
an assistant needs before it acts.

``pyphi-mcp install`` writes two things: the server's entry in the client's
Model Context Protocol configuration, and a short block of PyPhi facts in the
project's agent instruction file. The second exists because the first is not
enough — a client may or may not surface a server's ``instructions``, and an
assistant that decides to drive PyPhi from a shell never reads them at all,
while a project's instruction file is in context before the first tool call
either way.

Nothing here imports the optional ``mcp`` dependency.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

#: The instruction file the block is written to. Codex, Cursor and others read
#: this name; Claude Code reads ``CLAUDE.md`` and is bridged by an import.
INSTRUCTIONS_FILE = "AGENTS.md"

#: The file Claude Code reads, and the line that points it at the block.
CLAUDE_FILE = "CLAUDE.md"
CLAUDE_IMPORT = f"@{INSTRUCTIONS_FILE}"

BLOCK_BEGIN = (
    "<!-- pyphi:begin — managed by `pyphi-mcp install`; edits inside are overwritten -->"
)
BLOCK_END = "<!-- pyphi:end -->"

#: The facts that go wrong unaided, and a pointer to the rest of the reference.
#: Deliberately short: a long block gets skimmed, and restating the reference
#: here would leave two copies to keep correct.
BLOCK_BODY = """\
## PyPhi

φₛ and Φ are different quantities under IIT 4.0. `analyze(...).phi` is φₛ,
system integrated information — whether the system exists as one whole.
`.big_phi` is Φ, structure integrated information — the sum of φ over the
Φ-structure's distinctions and relations.

States are little-endian: the first node is the least-significant bit.

Analyses are superexponential in substrate size. `pyphi.cost.estimate_analysis`
is free; call it before any run over more than a handful of units.

These three are the ones that go wrong unaided; the rest of the reference is
`get_iit_reference("theory")` and `("equations")` where the MCP server is
connected, or
`python -c "from pyphi.mcp import content; print(content.load('gotchas'))"`
otherwise.

Where the server is connected, use its tools for exploration and for
interpreting results: they report which formalism produced each number, refuse
runs too large to finish, and keep φₛ and Φ distinct. The server holds results
only in memory, so anything that has to be reproducible belongs in a script —
where these same facts still apply."""


def block() -> str:
    """The instruction block, delimiters included."""
    return f"{BLOCK_BEGIN}\n{BLOCK_BODY}\n{BLOCK_END}"


def registration(spec: str | None = None) -> dict[str, Any]:
    """The client configuration entry that launches the server.

    Parameters
    ----------
    spec : str, optional
        A package specification for ``uvx`` to resolve at each launch, such as
        ``"pyphi[mcp] @ git+https://github.com/wmayner/pyphi.git@main"``. If
        None, the entry runs this interpreter, whose environment already
        provides the server.

    Notes
    -----
    Resolving a specification names a version rather than an environment, so
    the server a client starts need not be the one that wrote the entry. The
    interpreter form has no such gap, and starts without a network.
    """
    if spec is None:
        return {"command": sys.executable, "args": ["-m", "pyphi.mcp"]}
    return {"command": "uvx", "args": ["--from", spec, "pyphi-mcp"]}


def config_path(directory: Path, scope: str, client: str) -> Path:
    """Where ``client`` keeps its Model Context Protocol server configuration.

    Parameters
    ----------
    directory : Path
        The project directory, used by the project-scoped Claude Code case.
    scope : {"project", "user"}
        Whether the registration applies to this directory or to every session.
    client : {"claude-code", "claude-desktop"}
        Claude Desktop is a single user-level application and ignores ``scope``.

    Raises
    ------
    ValueError
        If ``scope`` or ``client`` is not one of the values above.
    """
    if scope not in ("project", "user"):
        raise ValueError(f"scope must be 'project' or 'user', not {scope!r}")
    if client == "claude-code":
        return (
            directory / ".mcp.json"
            if scope == "project"
            else Path.home() / ".claude.json"
        )
    if client == "claude-desktop":
        system = platform.system()
        if system == "Darwin":
            base = Path.home() / "Library" / "Application Support" / "Claude"
        elif system == "Windows":
            base = Path.home() / "AppData" / "Roaming" / "Claude"
        else:
            base = Path.home() / ".config" / "Claude"
        return base / "claude_desktop_config.json"
    raise ValueError(f"client must be 'claude-code' or 'claude-desktop', not {client!r}")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    return json.loads(text)


def write_registration(path: Path, entry: dict[str, Any], force: bool = False) -> bool:
    """Add the ``pyphi`` server to the config at ``path``, keeping any others.

    Returns
    -------
    bool
        Whether the file was written. ``False`` means the entry was already
        present and identical.

    Raises
    ------
    FileExistsError
        If a different ``pyphi`` entry exists and ``force`` is not set.
    """
    config = _read_json(path)
    servers = config.setdefault("mcpServers", {})
    existing = servers.get("pyphi")
    if existing == entry:
        return False
    if existing is not None and not force:
        raise FileExistsError(
            f"{path} already registers a different pyphi server:\n"
            f"  existing: {json.dumps(existing)}\n"
            f"  new:      {json.dumps(entry)}\n"
            "Pass --force to replace it."
        )
    servers["pyphi"] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return True


def remove_registration(path: Path) -> bool:
    """Drop the ``pyphi`` server from the config at ``path``.

    Other servers, and every other key, are left as they were. Returns whether
    anything was removed.
    """
    if not path.exists():
        return False
    config = _read_json(path)
    servers = config.get("mcpServers", {})
    if "pyphi" not in servers:
        return False
    del servers["pyphi"]
    if not servers:
        del config["mcpServers"]
    if not config:
        path.unlink()
    else:
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return True


def _split_on_block(text: str) -> tuple[str, str] | None:
    """Return the text before and after the managed block, or None if absent."""
    start = text.find(BLOCK_BEGIN)
    if start == -1:
        return None
    end = text.find(BLOCK_END, start)
    if end == -1:
        return None
    return text[:start], text[end + len(BLOCK_END) :]


def write_block(path: Path) -> None:
    """Write the instruction block to ``path``, replacing an earlier one.

    Content outside the delimiters is preserved exactly, so a later install
    refreshes the block without touching anything the user wrote around it.
    """
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    split = _split_on_block(text)
    if split is None:
        body = f"{text.rstrip()}\n\n{block()}\n" if text.strip() else f"{block()}\n"
    else:
        before, after = split
        body = f"{before}{block()}{after}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def remove_block(path: Path) -> bool:
    """Remove the instruction block from ``path``.

    Deletes the file if nothing but the block was in it. Returns whether
    anything was removed.
    """
    if not path.exists():
        return False
    split = _split_on_block(path.read_text(encoding="utf-8"))
    if split is None:
        return False
    before, after = split
    remainder = f"{before}{after}"
    if remainder.strip():
        path.write_text(remainder.strip() + "\n", encoding="utf-8")
    else:
        path.unlink()
    return True


def ensure_claude_import(path: Path) -> bool:
    """Make sure Claude Code loads the instruction file.

    Claude Code reads ``CLAUDE.md`` and not ``AGENTS.md``, so the block is
    reached through the ``@AGENTS.md`` import. A symlink also works but needs
    Administrator privileges or Developer Mode on Windows. Returns whether the
    file was written.

    References
    ----------
    https://code.claude.com/docs/en/memory
    """
    if path.is_symlink():
        # Already bridged, by the other documented mechanism.
        return False
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if any(line.strip() == CLAUDE_IMPORT for line in text.splitlines()):
        return False
    body = (
        f"{CLAUDE_IMPORT}\n\n{text.lstrip()}" if text.strip() else f"{CLAUDE_IMPORT}\n"
    )
    path.write_text(body, encoding="utf-8")
    return True


def remove_claude_import(path: Path) -> bool:
    """Drop the ``@AGENTS.md`` import, and the file if that was all it held."""
    if not path.exists() or path.is_symlink():
        return False
    lines = path.read_text(encoding="utf-8").splitlines()
    kept = [line for line in lines if line.strip() != CLAUDE_IMPORT]
    if len(kept) == len(lines):
        return False
    remainder = "\n".join(kept).strip()
    if remainder:
        path.write_text(remainder + "\n", encoding="utf-8")
    else:
        path.unlink()
    return True


def install(
    directory: Path,
    *,
    scope: str = "project",
    client: str = "claude-code",
    spec: str | None = None,
    force: bool = False,
) -> list[str]:
    """Register the server and write the instruction block.

    Returns
    -------
    list of str
        One line per action taken, for reporting.
    """
    directory = Path(directory)
    actions = []
    path = config_path(directory, scope, client)
    if write_registration(path, registration(spec), force=force):
        actions.append(f"registered the pyphi server in {path}")
    else:
        actions.append(f"{path} already registers this server")

    instructions = directory / INSTRUCTIONS_FILE
    write_block(instructions)
    actions.append(f"wrote the PyPhi block to {instructions}")

    claude = directory / CLAUDE_FILE
    if ensure_claude_import(claude):
        actions.append(f"added `{CLAUDE_IMPORT}` to {claude} so Claude Code reads it")
    return actions


def uninstall(
    directory: Path, *, scope: str = "project", client: str = "claude-code"
) -> list[str]:
    """Undo :func:`install`, leaving everything it did not write untouched."""
    directory = Path(directory)
    actions = []
    path = config_path(directory, scope, client)
    if remove_registration(path):
        actions.append(f"removed the pyphi server from {path}")
    if remove_block(directory / INSTRUCTIONS_FILE):
        actions.append(f"removed the PyPhi block from {directory / INSTRUCTIONS_FILE}")
    if remove_claude_import(directory / CLAUDE_FILE):
        actions.append(f"removed `{CLAUDE_IMPORT}` from {directory / CLAUDE_FILE}")
    return actions or ["nothing to remove"]


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--scope",
        choices=("project", "user"),
        default="project",
        help="project (default) writes to this directory; user, to every session",
    )
    parser.add_argument(
        "--client",
        choices=("claude-code", "claude-desktop"),
        default="claude-code",
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path.cwd(),
        help="the project directory (default: the working directory)",
    )


def build_parser() -> argparse.ArgumentParser:
    """The ``pyphi-mcp`` command line. With no subcommand, the server runs."""
    parser = argparse.ArgumentParser(
        prog="pyphi-mcp",
        description=(
            "Run the PyPhi Model Context Protocol server, or set it up in a "
            "project. With no subcommand, the server runs over stdio."
        ),
    )
    sub = parser.add_subparsers(dest="command")

    install_parser = sub.add_parser(
        "install", help="register the server and write the PyPhi instruction block"
    )
    _add_common(install_parser)
    install_parser.add_argument(
        "--from",
        dest="spec",
        default=None,
        help=(
            "launch through `uvx` with this package specification, instead of "
            "through the interpreter running this command"
        ),
    )
    install_parser.add_argument(
        "--print",
        dest="print_only",
        action="store_true",
        help="write nothing; print the registration and the block instead",
    )
    install_parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing, different pyphi registration",
    )

    uninstall_parser = sub.add_parser(
        "uninstall", help="remove what install wrote, leaving the rest alone"
    )
    _add_common(uninstall_parser)
    return parser


def run(args: argparse.Namespace) -> int:
    """Carry out an ``install`` or ``uninstall`` and report what happened."""
    if args.command == "install":
        if args.print_only:
            config = {"mcpServers": {"pyphi": registration(args.spec)}}
            print(f"{config_path(args.directory, args.scope, args.client)}:")
            print(json.dumps(config, indent=2))
            print(f"\n{args.directory / INSTRUCTIONS_FILE}:")
            print(block())
            return 0
        try:
            actions = install(
                args.directory,
                scope=args.scope,
                client=args.client,
                spec=args.spec,
                force=args.force,
            )
        except FileExistsError as error:
            print(error)
            return 1
    else:
        actions = uninstall(args.directory, scope=args.scope, client=args.client)
    for action in actions:
        print(action)
    return 0
