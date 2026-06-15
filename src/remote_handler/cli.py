"""Command-line interface for remote_handler configuration.

Subcommands:
  configure              build the global config and persist it
  configure cluster NAME build/edit a per-cluster ClusterConfig JSON
  check [--connect]      validate config and diagnose remote setup
  show [--json]          print the resolved configuration

Stdlib only; interactive prompts live in ``cli_prompts``.
"""

import argparse
import json
import os
import socket
from pathlib import Path
from typing import Callable, Optional, Sequence

from remote_handler import __version__
from remote_handler.cli_prompts import (
    prompt_choice,
    prompt_text,
    prompt_yes_no,
)
from remote_handler.config import (
    get_config,
    get_persistent_config_dir,
    get_persistent_config_file,
    reset_config,
)
from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.core.ssh_connection import (
    ConfigBasedSSHClientProvider,
    SSHConfigParser,
)
from remote_handler.exceptions import ConfigurationError, RemoteConnectionError

Reporter = Callable[[str, str], None]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _list_header_templates() -> list[str]:
    """Header ``.sh.j2`` templates on the search path (excl. ``body_*``)."""
    names: set[str] = set()
    for template_dir in get_config().get_template_dirs():
        for path in Path(template_dir).glob("*.sh.j2"):
            if not path.name.startswith("body_"):
                names.add(path.name)
    return sorted(names)


def _list_cluster_names(config_dir: Path) -> list[str]:
    """Per-cluster config names (``*.json`` stems, excl. ``config.json``)."""
    if not config_dir.is_dir():
        return []
    return sorted(
        p.stem for p in config_dir.glob("*.json") if p.name != "config.json"
    )


def _write_json(path: Path, data: dict) -> None:
    """Write ``data`` as pretty JSON, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def _resolved_config_dir() -> Path:
    """Config dir from the resolved config, falling back to the XDG dir.

    Reads the attribute directly: ``get_config_dir`` raises when the dir is
    set but does not exist yet, which is exactly the state ``configure``
    needs to handle.
    """
    config_dir = get_config().config_dir
    if config_dir is not None:
        return config_dir
    return get_persistent_config_dir()


# --------------------------------------------------------------------------
# configure cluster
# --------------------------------------------------------------------------
def cmd_configure_cluster(args: argparse.Namespace) -> int:
    """Interactively create/edit a per-cluster ClusterConfig JSON."""
    cfg = get_config()
    config_dir = _resolved_config_dir()

    name = args.name or prompt_text("Cluster config name")
    if not name:
        print("A config name is required.")
        return 1
    target = config_dir / f"{name}.json"

    existing: dict = {}
    if target.exists():
        try:
            existing = json.loads(target.read_text())
        except (OSError, json.JSONDecodeError):
            existing = {}
        if not prompt_yes_no(
            f"{target} already exists. Edit/overwrite it?", default=True
        ):
            print("Aborted.")
            return 1

    hosts = SSHConfigParser().get_available_hosts()
    remote_machine = prompt_choice(
        "Remote SSH alias (from ~/.ssh/config)",
        hosts,
        default=existing.get("remote_machine"),
    )
    partition = prompt_text(
        "SLURM partition",
        default=existing.get("partition", cfg.default_partition),
    )
    conda_env = prompt_text(
        "Conda environment",
        default=existing.get("conda_env", cfg.default_conda_env),
    )
    conda_source = prompt_text(
        "Conda activation source",
        default=existing.get("conda_source", cfg.default_conda_source),
    )
    savepath = prompt_text(
        "Remote savepath (where computations are stored)",
        default=existing.get("savepath"),
    )
    template = prompt_choice(
        "SLURM header template",
        _list_header_templates(),
        default=existing.get("template"),
    )

    data = {
        "remote_machine": remote_machine,
        "partition": partition,
        "conda_env": conda_env,
        "conda_source": conda_source,
        "savepath": savepath,
        "template": template,
    }

    empty = [key for key, value in data.items() if not str(value).strip()]
    if empty:
        print(f"Refusing to write: empty required field(s): {empty}")
        return 1
    try:
        ClusterConfig.from_dict(data)
    except (TypeError, ConfigurationError) as exc:
        print(f"Refusing to write invalid config: {exc}")
        return 1

    _write_json(target, data)
    print(f"Wrote {target}")
    return 0


# --------------------------------------------------------------------------
# configure (global)
# --------------------------------------------------------------------------
def cmd_configure(args: argparse.Namespace) -> int:
    """Interactively build the global config and persist it."""
    current = get_config()

    config_dir = prompt_text(
        "Config directory (where cluster JSONs live)",
        default=str(current.config_dir or get_persistent_config_dir()),
    )
    templates = prompt_text(
        "Custom SLURM templates dir (blank = built-in only)",
        default=(
            str(current.slurm_templates_dir)
            if current.slurm_templates_dir
            else ""
        ),
    )
    partition = prompt_text(
        "Default SLURM partition", default=current.default_partition
    )
    conda_env = prompt_text(
        "Default conda environment", default=current.default_conda_env
    )
    conda_source = prompt_text(
        "Default conda activation source",
        default=current.default_conda_source,
    )

    # Which per-cluster config the `remote` test tier targets. Offer the
    # known cluster names as a hint; blank falls back to remote.json.
    clusters = _list_cluster_names(_resolved_config_dir())
    hint = f" (known: {', '.join(clusters)})" if clusters else ""
    test_cluster = prompt_text(
        f"Cluster for remote tests, name without .json{hint}",
        default=current.test_cluster or "",
    )

    data = {
        "config_dir": config_dir or None,
        "slurm_templates_dir": templates or None,
        "default_partition": partition,
        "default_conda_env": conda_env,
        "default_conda_source": conda_source,
        "test_cluster": test_cluster or None,
    }
    pfile = get_persistent_config_file()
    _write_json(pfile, data)
    reset_config()
    print(f"Wrote {pfile}")
    print(
        "Configuration persisted; REMOTE_HANDLER_CONFIG_DIR is no longer "
        "required."
    )
    return 0


# --------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------
def _check_connect(cc: ClusterConfig, name: str, report: Reporter) -> None:
    """Best-effort live SSH probe (sbatch + conda env). Never raises."""
    provider = ConfigBasedSSHClientProvider(cc.remote_machine)
    try:
        ssh, sftp = provider.create_connection()
    except (RemoteConnectionError, socket.gaierror) as exc:
        report("WARN", f"{name}: cannot connect ({exc})")
        return
    except AttributeError as exc:
        if "public_blob" in str(exc):
            report(
                "WARN",
                f"{name}: paramiko ssh-agent certificate bug -- run "
                "'ssh-add -D' and retry (see DirectSSHConnection note).",
            )
        else:
            report("WARN", f"{name}: connection error ({exc})")
        return
    except Exception as exc:  # noqa: BLE001 - diagnostic, never crash
        report("WARN", f"{name}: connection error ({exc})")
        return

    try:
        _in, out, _err = ssh.exec_command("which sbatch")
        sbatch = out.read().decode().strip()
        if sbatch:
            report("PASS", f"{name}: sbatch at {sbatch}")
        else:
            report("WARN", f"{name}: sbatch not found on remote")

        src = cc.conda_source.rstrip("/")
        root = (
            src[: -len("/bin/activate")]
            if src.endswith("/bin/activate")
            else src
        )
        if root.startswith("~"):
            root = "$HOME" + root[1:]
        probe = f"test -d {root}/envs/{cc.conda_env} && echo OK"
        _in2, out2, _err2 = ssh.exec_command(probe)
        if out2.read().decode().strip() == "OK":
            report("PASS", f"{name}: conda env '{cc.conda_env}' present")
        else:
            report(
                "WARN",
                f"{name}: conda env '{cc.conda_env}' not found under "
                f"{root}/envs",
            )
    finally:
        try:
            sftp.close()
            ssh.close()
        finally:
            provider.cleanup()


def cmd_check(args: argparse.Namespace) -> int:
    """Validate config and cluster JSONs; optional live SSH probe."""
    ok = True

    def report(status: str, msg: str) -> None:
        print(f"[{status}] {msg}")

    # 1. Config resolves.
    try:
        cfg = get_config()
    except ConfigurationError as exc:
        report("FAIL", f"Configuration error: {exc}")
        return 1
    report("PASS", "Configuration resolved")
    pfile = get_persistent_config_file()
    report(
        "INFO",
        f"Persistent file: {pfile} "
        f"({'present' if pfile.exists() else 'absent'})",
    )
    if os.environ.get("REMOTE_HANDLER_CONFIG_DIR"):
        report("INFO", "REMOTE_HANDLER_CONFIG_DIR is set (overrides file)")

    # 2. config_dir exists.
    config_dir = cfg.config_dir
    if config_dir is None or not config_dir.is_dir():
        report(
            "FAIL",
            f"Config dir missing: {config_dir}. Run 'remote_handler "
            "configure' or create the directory.",
        )
        return 1
    report("PASS", f"Config dir: {config_dir}")

    # 3. Cluster JSONs present.
    cluster_files = sorted(
        p for p in config_dir.glob("*.json") if p.name != "config.json"
    )
    if not cluster_files:
        report("WARN", "No cluster configs found (remote tests will skip)")
        return 0
    report("PASS", f"Found {len(cluster_files)} cluster config(s)")

    # 4-7. Per-cluster checks.
    for path in cluster_files:
        try:
            cc = ClusterConfig.from_file(str(path.resolve()))
        except (ConfigurationError, TypeError) as exc:
            report("FAIL", f"{path.name}: cannot load ({exc})")
            ok = False
            continue
        report("PASS", f"{path.name}: loads")

        try:
            SSHConfigParser().parse_host_config(cc.remote_machine)
            report(
                "PASS",
                f"{path.name}: alias '{cc.remote_machine}' resolves",
            )
        except (RemoteConnectionError, FileNotFoundError) as exc:
            report(
                "FAIL",
                f"{path.name}: alias '{cc.remote_machine}' unresolved "
                f"({exc})",
            )
            ok = False
            continue

        if cfg.find_template(cc.template) is None:
            report(
                "WARN",
                f"{path.name}: template '{cc.template}' not on search path",
            )

        if args.connect:
            _check_connect(cc, path.name, report)

    return 0 if ok else 1


# --------------------------------------------------------------------------
# show
# --------------------------------------------------------------------------
def cmd_show(args: argparse.Namespace) -> int:
    """Print the resolved configuration and its provenance."""
    cfg = get_config()
    if args.as_json:
        print(json.dumps(cfg.to_dict(), indent=2))
        return 0

    pfile = get_persistent_config_file()
    print(f"remote_handler {__version__}")
    print(
        f"persistent config file: {pfile} "
        f"({'present' if pfile.exists() else 'absent'})"
    )
    env_cd = os.environ.get("REMOTE_HANDLER_CONFIG_DIR")
    if env_cd:
        print(f"REMOTE_HANDLER_CONFIG_DIR={env_cd} (overrides file)")
    print(f"config_dir: {cfg.config_dir}")
    print(f"default_partition: {cfg.default_partition}")
    print(f"default_conda_env: {cfg.default_conda_env}")
    print(f"default_conda_source: {cfg.default_conda_source}")
    print(f"test_cluster: {cfg.test_cluster or '(remote.json)'}")
    print("template search dirs:")
    for template_dir in cfg.get_template_dirs():
        print(f"  - {template_dir}")
    config_dir = cfg.config_dir
    clusters: list[str] = []
    if config_dir and config_dir.is_dir():
        clusters = sorted(
            p.stem
            for p in config_dir.glob("*.json")
            if p.name != "config.json"
        )
    print(f"clusters: {', '.join(clusters) if clusters else '(none)'}")
    return 0


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="remote_handler",
        description="Configure and diagnose remote_handler.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"remote_handler {__version__}",
    )
    sub = parser.add_subparsers(dest="command")

    p_conf = sub.add_parser(
        "configure", help="set up global or per-cluster configuration"
    )
    conf_sub = p_conf.add_subparsers(dest="target")
    p_cluster = conf_sub.add_parser(
        "cluster", help="create/edit a per-cluster config"
    )
    p_cluster.add_argument(
        "name", nargs="?", default=None, help="cluster config name"
    )

    p_check = sub.add_parser("check", help="validate / diagnose config")
    p_check.add_argument(
        "--connect",
        action="store_true",
        help="also open a live SSH connection and probe sbatch",
    )

    p_show = sub.add_parser("show", help="print the resolved config")
    p_show.add_argument(
        "--json", action="store_true", dest="as_json", help="emit JSON"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "configure":
        if getattr(args, "target", None) == "cluster":
            return cmd_configure_cluster(args)
        return cmd_configure(args)
    if args.command == "check":
        return cmd_check(args)
    if args.command == "show":
        return cmd_show(args)

    parser.print_help()
    return 1
