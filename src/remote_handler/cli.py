"""Command-line interface for remote_handler configuration.

Subcommands:
  configure              build the global config and persist it
  configure cluster NAME build/edit a per-cluster ClusterConfig JSON
  configure test [NAME]  set the cluster used by the remote test tier
  check [--connect]      validate config and diagnose remote setup
  show [--json]          print the resolved configuration
  show templates         list registered SLURM templates + paths
  show clusters          list registered cluster configs + paths
  show extras [M ...]    probe extras locally, or per-machine (local/clusters)

Light imports only at module load; ``show extras`` imports the ML stacks
(local probe) or paramiko (remote probe) lazily, inside the command.
Interactive prompts live in ``cli_prompts``.
"""

import argparse
import json
import os
import socket
import sys
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
# configure test
# --------------------------------------------------------------------------
def cmd_configure_test(args: argparse.Namespace) -> int:
    """Set only the remote-test target cluster, preserving other config.

    Updates ``test_cluster`` in the persistent config file in place, without
    re-running the full ``configure`` flow.
    """
    config_dir = _resolved_config_dir()
    name = args.name
    if name is None:
        clusters = _list_cluster_names(config_dir)
        hint = f" (known: {', '.join(clusters)})" if clusters else ""
        name = prompt_text(
            f"Cluster for remote tests, name without .json{hint}",
            default=get_config().test_cluster or "",
        )
    name = name.strip()

    # Soft check: warn (don't block) if the cluster JSON isn't there yet.
    if name and not (config_dir / f"{name}.json").exists():
        print(
            f"Warning: {name}.json not found in {config_dir}; "
            "create it with 'remote_handler configure cluster "
            f"{name}'."
        )

    pfile = get_persistent_config_file()
    data: dict = {}
    if pfile.exists():
        try:
            data = json.loads(pfile.read_text())
        except (OSError, json.JSONDecodeError):
            data = {}
    data["test_cluster"] = name or None

    _write_json(pfile, data)
    reset_config()
    print(
        f"Set test_cluster to {data['test_cluster'] or '(remote.json)'} "
        f"in {pfile}"
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

        def _sh(path: str) -> str:
            return "$HOME" + path[1:] if path.startswith("~") else path

        root = _sh(root)
        env_dirs = [f"{root}/envs", "$HOME/.conda/envs"]
        if cc.conda_env_dir:
            env_dirs.insert(0, _sh(cc.conda_env_dir.rstrip("/")))
        tests = " || ".join(f"test -d {d}/{cc.conda_env}" for d in env_dirs)
        probe = f"{{ {tests}; }} && echo OK"
        _in2, out2, _err2 = ssh.exec_command(probe)
        if out2.read().decode().strip() == "OK":
            report("PASS", f"{name}: conda env '{cc.conda_env}' present")
        else:
            report(
                "WARN",
                f"{name}: conda env '{cc.conda_env}' not found under "
                + ", ".join(env_dirs),
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


def cmd_show_templates(args: argparse.Namespace) -> int:
    """List the registered SLURM templates and their resolved file paths.

    Walks the template search path in priority order; when the same template
    name exists in several dirs, the first (highest priority) wins -- the same
    resolution ``find_template`` uses.
    """
    cfg = get_config()
    resolved: dict[str, Path] = {}
    for template_dir in cfg.get_template_dirs():
        template_dir = Path(template_dir)
        if not template_dir.is_dir():
            continue
        for path in sorted(template_dir.glob("*.j2")):
            resolved.setdefault(path.name, path)

    if not resolved:
        print("No templates found on the search path.")
        return 0
    for name in sorted(resolved):
        print(f"{name}\t{resolved[name]}")
    return 0


# Each optional extra mapped to the import(s) that prove it is usable. Keep in
# sync with ``[project.optional-dependencies]`` in pyproject.toml. ``mace`` and
# ``mace_freeze`` install the *same* distribution, so they are told apart by
# capability: the fork ships ``mace.tools.freeze``, which stock mace lacks.
_EXTRA_IMPORTS: dict[str, list[str]] = {
    "dev": ["pytest"],
    "docs": ["sphinx"],
    "mace": ["mace"],
    "torch_sim": ["torch_sim", "mace"],
    "mace_freeze": ["mace.tools.freeze"],
    "clinamen": ["clinamen2"],
    "quests": ["quests"],
    "neuralil": ["neuralil"],
    "mace_jax": ["mace_jax"],
}

# External (non-Python) applications a cluster sets up via its
# ``ClusterConfig.environment[<key>]`` entry, mapped to the launch binary to
# look for after that setup. Probed by checking the binary is on PATH (a
# ``command -v`` after the module loads) -- presence is the signal we need.
_BINARY_PROBES: dict[str, str] = {
    "vasp": "vasp_std",
    "orca": "orca",
}

# Reserved keys under which ``_probe_machine`` stashes the installed
# ``remote_handler`` version and the conda env it was read from. Neither is an
# extra/binary, so they never become grid rows -- they are pulled out and shown
# separately above the table.
_RH_VERSION_KEY = "__rh_version__"
_RH_ENV_KEY = "__rh_env__"


def _local_rh_version() -> str:
    """Installed ``remote_handler`` version in this process, or ``unknown``."""
    import importlib.metadata as md

    try:
        return md.version("remote_handler")
    except Exception:  # noqa: BLE001 - report rather than crash the command
        return "unknown"


def _local_conda_env() -> str:
    """The active conda env name for this process, or ``""`` if not in one."""
    import os

    return os.environ.get("CONDA_DEFAULT_ENV", "")


def _rh_version_remote(remote_fs, cluster_config) -> str:
    """Installed ``remote_handler`` version in a cluster's conda env, over SSH.

    Reuses the already-open ``remote_fs`` connection. Returns ``unknown`` if it
    is not importable / not installed there.
    """
    from remote_handler.core import remote_env

    activation = remote_env.conda_activation_script(
        cluster_config.conda_source, cluster_config.conda_env
    )
    cmd = (
        f"{activation}\n"
        'python -c "import importlib.metadata as m; '
        "print('RH_VERSION:' + m.version('remote_handler'))\"\n"
    )
    result = remote_fs.run_command(cmd)
    for line in (result.stdout or "").splitlines():
        if line.startswith("RH_VERSION:"):
            return line[len("RH_VERSION:") :].strip()
    return "unknown"


def _declared_extras() -> set:
    """The optional-dependency groups declared in this package's metadata."""
    import importlib.metadata as md

    return {
        e.replace("-", "_")
        for e in (
            md.metadata("remote_handler").get_all("Provides-Extra") or []
        )
    }


def cmd_show_extras(args: argparse.Namespace) -> int:
    """Probe which optional extras / applications are usable on a machine.

    With no target, probes the **local** environment (imports each extra's
    modules). With one or more targets (``local`` and/or registered cluster
    names) it prints a machine-by-machine table -- each cluster is probed over
    SSH inside its conda env, and any binary applications declared in the
    cluster's ``environment`` map (e.g. ``vasp``) are checked on PATH after
    their setup runs.
    """
    targets = getattr(args, "targets", None) or []
    if not targets:
        return _show_extras_local()
    return _show_extras_table(targets)


def _show_extras_local() -> int:
    """Single-machine view: import each extra's modules in this process."""
    import importlib
    import warnings

    env = _local_conda_env()
    where = f"local, env: {env}" if env else "local"
    print(f"remote_handler {_local_rh_version()} ({where})\n")
    print(
        "Probing optional extras (imports torch / jax / ...; may be slow):\n"
    )
    for extra in sorted(_declared_extras()):
        modules = _EXTRA_IMPORTS.get(extra)
        if modules is None:
            print(f"  [??] {extra}  (no probe defined)")
            continue
        missing = []
        for module in modules:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    importlib.import_module(module)
            except Exception:  # noqa: BLE001 - any import failure -> missing
                missing.append(module)
        if missing:
            print(f"  [ ] {extra}  (missing: {', '.join(missing)})")
        else:
            print(f"  [x] {extra}")

    binaries = _probe_binaries_local()
    if binaries:
        print()
        for app in sorted(binaries):
            binary = _BINARY_PROBES[app]
            mark = "x" if binaries[app] else " "
            where = "on PATH" if binaries[app] else "not on PATH"
            print(f"  [{mark}] {app}  ({binary} {where})")
    return 0


def _probe_extras_local() -> dict:
    """Map each extra -> importable? in this process."""
    import importlib
    import warnings

    result = {}
    for extra, modules in _EXTRA_IMPORTS.items():
        ok = True
        for module in modules:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    importlib.import_module(module)
            except Exception:  # noqa: BLE001
                ok = False
                break
        result[extra] = ok
    return result


def _remote_extras_snippet() -> str:
    """A python script (for ``python - <<EOF``) that import-probes the extras
    and prints ``RH_EXTRAS:{json}`` -- run inside the remote conda env."""
    return (
        "import importlib, json, warnings\n"
        f"mods = {_EXTRA_IMPORTS!r}\n"
        "res = {}\n"
        "for _x, _ms in mods.items():\n"
        "    _ok = True\n"
        "    for _m in _ms:\n"
        "        try:\n"
        "            with warnings.catch_warnings():\n"
        "                warnings.simplefilter('ignore')\n"
        "                importlib.import_module(_m)\n"
        "        except Exception:\n"
        "            _ok = False\n"
        "            break\n"
        "    res[_x] = _ok\n"
        "print('RH_EXTRAS:' + json.dumps(res))\n"
    )


def _probe_extras_remote(remote_fs, cluster_config) -> dict:
    """Import-probe the extras in a cluster's conda env over SSH."""
    import json

    from remote_handler.core import remote_env

    activation = remote_env.conda_activation_script(
        cluster_config.conda_source, cluster_config.conda_env
    )
    cmd = (
        f"{activation}\n"
        f"python - <<'RH_PY'\n{_remote_extras_snippet()}RH_PY\n"
    )
    result = remote_fs.run_command(cmd)
    for line in (result.stdout or "").splitlines():
        if line.startswith("RH_EXTRAS:"):
            return json.loads(line[len("RH_EXTRAS:") :])
    raise RuntimeError("remote extras probe returned no parseable result")


def _probe_binaries_local() -> dict:
    """Map each known binary app -> on PATH locally?

    There is no ``environment`` setup to source for ``local``, so this reflects
    the *current* local PATH -- which is exactly right when ``local`` is itself
    a cluster node where the modules are already loaded.
    """
    import shutil

    return {
        app: shutil.which(binary) is not None
        for app, binary in _BINARY_PROBES.items()
    }


def _probe_binaries_remote(remote_fs, cluster_config) -> dict:
    """For each binary app declared in the cluster's ``environment`` map, run
    its setup (in a login shell, so ``module`` works) and check the launch
    binary is on PATH. Returns ``{app: bool}``."""
    environment = getattr(cluster_config, "environment", {}) or {}
    apps = [(a, b) for a, b in _BINARY_PROBES.items() if a in environment]
    if not apps:
        return {}
    lines = ["bash -l <<'RH_BIN'"]
    for app, binary in apps:
        lines.append(f"( {environment[app]}")
        lines.append(
            f"command -v {binary} >/dev/null 2>&1 "
            f"&& echo {app}=yes || echo {app}=no )"
        )
    lines.append("RH_BIN")
    result = remote_fs.run_command("\n".join(lines) + "\n")
    keys = {a for a, _ in apps}
    out = {}
    for line in (result.stdout or "").splitlines():
        key, sep, val = line.partition("=")
        if sep and key.strip() in keys:
            out[key.strip()] = val.strip() == "yes"
    return out


def _probe_machine(target: str):
    """Probe one target. Returns a ``{row: bool}`` dict, or ``None`` if the
    machine could not be reached/resolved (its column renders as unknown)."""
    if target == "local":
        status = _probe_extras_local()
        status.update(_probe_binaries_local())
        status[_RH_VERSION_KEY] = _local_rh_version()
        status[_RH_ENV_KEY] = _local_conda_env()
        return status

    from remote_handler.core.cluster_handler import ClusterConfig
    from remote_handler.core.file_handler import RemoteFileSystem

    try:
        cc = ClusterConfig.from_name(target)
        remote_fs = RemoteFileSystem(remote_alias=cc.remote_machine)
    except Exception as exc:  # noqa: BLE001 - unresolved/unreachable cluster
        print(f"[WARN] {target}: cannot connect ({exc})", file=sys.stderr)
        return None
    try:
        status = _probe_extras_remote(remote_fs, cc)
        status.update(_probe_binaries_remote(remote_fs, cc))
        status[_RH_VERSION_KEY] = _rh_version_remote(remote_fs, cc)
        status[_RH_ENV_KEY] = getattr(cc, "conda_env", "") or ""
        return status
    except Exception as exc:  # noqa: BLE001 - probe failure
        print(f"[WARN] {target}: probe failed ({exc})", file=sys.stderr)
        return None


def _show_extras_table(targets) -> int:
    """Probe each target and render a machine-by-machine availability table."""
    print(f"Probing {', '.join(targets)} (SSH per cluster; may be slow):\n")
    columns = [(t, _probe_machine(t)) for t in targets]

    # Installed remote_handler version per machine (not a ✓/✗ row -- a version
    # string), with the conda env it was read from. Shown as its own block so a
    # stale install -- or a probe hitting a different env than you expect -- is
    # easy to spot.
    def _version(status):
        if status is None:
            return "?"  # unreachable
        return status.get(_RH_VERSION_KEY, "unknown")

    def _env(status):
        if status is None:
            return ""
        return status.get(_RH_ENV_KEY, "")

    name_w = max(len(t) for t, _ in columns)
    ver_w = max(len(_version(s)) for _, s in columns)
    print("remote_handler version")
    for t, status in columns:
        env = _env(status)
        suffix = f"  (env: {env})" if env else ""
        print(f"  {t.ljust(name_w)}  {_version(status).ljust(ver_w)}{suffix}")
    print()

    extra_rows = sorted(_declared_extras())
    # Binary-app rows: only those that surfaced on at least one machine.
    binary_rows = sorted(
        {
            app
            for _, status in columns
            if status
            for app in status
            if app in _BINARY_PROBES
        }
    )

    def cell(status, row):
        if status is None:
            return "?"
        if row not in status:
            return "—"  # em dash: not applicable on this machine
        return "✓" if status[row] else "✗"  # check / cross

    label_w = max([len("extra")] + [len(r) for r in extra_rows + binary_rows])
    col_w = {t: max(len(t), 3) for t, _ in columns}

    header = (
        "extra".ljust(label_w)
        + "  "
        + "  ".join(t.center(col_w[t]) for t, _ in columns)
    )
    print(header)
    print("-" * len(header))

    def emit(rows):
        for row in rows:
            line = (
                row.ljust(label_w)
                + "  "
                + "  ".join(
                    cell(status, row).center(col_w[t]) for t, status in columns
                )
            )
            print(line)

    emit(extra_rows)
    if binary_rows:
        print("-" * len(header))
        emit(binary_rows)

    # Non-zero if any requested machine could not be probed (scriptable).
    return 0 if all(status is not None for _, status in columns) else 1


def cmd_show_clusters(args: argparse.Namespace) -> int:
    """List the registered per-cluster configs and their file paths."""
    cfg = get_config()
    config_dir = cfg.config_dir
    if config_dir is None or not config_dir.is_dir():
        print(f"No config directory: {config_dir}")
        return 0
    clusters = sorted(
        p for p in config_dir.glob("*.json") if p.name != "config.json"
    )
    if not clusters:
        print(f"No cluster configs in {config_dir}")
        return 0
    for path in clusters:
        print(f"{path.stem}\t{path}")
    return 0


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="remote_handler",
        description="Configure and diagnose remote_handler.",
        epilog=(
            "all commands:\n"
            "  configure                 build & persist the global config\n"
            "  configure cluster [NAME]  create/edit a per-cluster config\n"
            "  configure test [NAME]     set the remote-test target cluster\n"
            "  check [--connect]         validate / diagnose config\n"
            "  show [--json]             print the resolved config\n"
            "  show templates            list registered templates + paths\n"
            "  show clusters             list registered clusters + paths\n"
            "  show extras               probe which optional extras are "
            "installed\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
    p_test = conf_sub.add_parser(
        "test", help="set the cluster used for remote tests"
    )
    p_test.add_argument(
        "name",
        nargs="?",
        default=None,
        help="cluster name without .json (blank prompts; empty clears)",
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
    show_sub = p_show.add_subparsers(dest="show_target")
    show_sub.add_parser(
        "templates", help="list registered SLURM templates + their paths"
    )
    show_sub.add_parser(
        "clusters", help="list registered cluster configs + their paths"
    )
    p_extras = show_sub.add_parser(
        "extras", help="probe which optional extras are installed"
    )
    p_extras.add_argument(
        "targets",
        nargs="*",
        help=(
            "machines to probe: 'local' and/or registered cluster names "
            "(none = local only; multiple = comparison table)"
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns a process exit code.

    Ctrl-C / Ctrl-D at any interactive prompt aborts quietly (no traceback)
    with exit code 130, the conventional SIGINT code.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "configure":
            target = getattr(args, "target", None)
            if target == "cluster":
                return cmd_configure_cluster(args)
            if target == "test":
                return cmd_configure_test(args)
            return cmd_configure(args)
        if args.command == "check":
            return cmd_check(args)
        if args.command == "show":
            show_target = getattr(args, "show_target", None)
            if show_target == "templates":
                return cmd_show_templates(args)
            if show_target == "clusters":
                return cmd_show_clusters(args)
            if show_target == "extras":
                return cmd_show_extras(args)
            return cmd_show(args)

        parser.print_help()
        return 1
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        return 130
    except EOFError:
        print("\nAborted (EOF).", file=sys.stderr)
        return 130
