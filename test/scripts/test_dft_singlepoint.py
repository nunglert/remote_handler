"""SCF-convergence gating in the ``dft_singlepoint.py`` VASP payload.

No VASP needed: the check reads OUTCAR/OSZICAR text (in VASP's own format),
and ``main`` runs with a fake calculator standing in for ``Vasp``.
"""

import importlib.util
import json
from pathlib import Path

import ase.io
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "remote_handler"
    / "scripts"
    / "dft_singlepoint.py"
)

_CONVERGED = (
    "------------------------ aborting loop because EDIFF is reached "
    "----------------------------------------\n"
)
_UNCONVERGED = (
    "------------------------ aborting loop EDIFF was not reached "
    "(unconverged)  ----------------------------\n"
)
_NELM = "   NELM   =     60;   NELMIN=  2; NELMDL= -5     # of ELM steps\n"


def _load_script():
    spec = importlib.util.spec_from_file_location("dft_singlepoint", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script():
    return _load_script()


def _oszicar(n_steps):
    lines = [
        "       N       E                     dE             d eps"
        "       ncg     rms          rms(c)\n"
    ]
    for n in range(1, n_steps + 1):
        kind = "DAV" if n <= 5 else "RMM"
        lines.append(
            f"{kind}: {n:3d}    -0.240387294548E+04   -0.15028E-04"
            "   -0.81831E-06 10274   0.365E-03\n"
        )
    lines.append("   1 F= -.24966555E+04 E0= -.24966177E+04  d E =-.75E-01\n")
    return "".join(lines)


def test_converged_vasp6_message(script, tmp_path):
    (tmp_path / "OUTCAR").write_text(_NELM + "...\n" + _CONVERGED)

    assert script.scf_converged(tmp_path) == (True, "")


def test_unconverged_vasp6_message(script, tmp_path):
    (tmp_path / "OUTCAR").write_text(_NELM + "...\n" + _UNCONVERGED)

    converged, reason = script.scf_converged(tmp_path)

    assert not converged
    assert "not reached" in reason


def test_only_the_last_ionic_step_counts(script, tmp_path):
    # e.g. a relaxation whose first ionic step failed but the final one
    # converged -- and the reverse.
    (tmp_path / "OUTCAR").write_text(_UNCONVERGED + _CONVERGED)
    assert script.scf_converged(tmp_path)[0]

    (tmp_path / "OUTCAR").write_text(_CONVERGED + _UNCONVERGED)
    assert not script.scf_converged(tmp_path)[0]


@pytest.mark.parametrize("n_steps,expected", [(33, True), (60, False)])
def test_vasp5_falls_back_to_step_count(script, tmp_path, n_steps, expected):
    # No end-of-loop message (VASP 5): compare steps against NELM=60.
    (tmp_path / "OUTCAR").write_text(_NELM)
    (tmp_path / "OSZICAR").write_text(_oszicar(n_steps))

    assert script.scf_converged(tmp_path)[0] is expected


def test_undecidable_counts_as_unconverged(script, tmp_path):
    assert not script.scf_converged(tmp_path)[0]  # no OUTCAR at all

    (tmp_path / "OUTCAR").write_text("no NELM, no loop messages\n")
    assert not script.scf_converged(tmp_path)[0]


class _FakeVasp(Calculator):
    """Stand-in for ``ase.calculators.vasp.Vasp``: returns fixed results and
    writes an OUTCAR with the given end-of-loop message."""

    implemented_properties = ["energy", "forces"]
    xc_defaults: dict = {}  # the script registers r2scan here
    outcar_tail = _CONVERGED

    def __init__(self, directory, **kwargs):
        super().__init__(directory=directory)

    def calculate(self, atoms=None, properties=None, system_changes=None):
        super().calculate(atoms, properties, system_changes or all_changes)
        (Path(self.directory) / "OUTCAR").write_text(_NELM + self.outcar_tail)
        self.results = {"energy": -1.5, "forces": np.zeros((len(atoms), 3))}


def _calcdir(tmp_path):
    calcdir = tmp_path / "calc"
    calcdir.mkdir()
    ase.io.write(
        calcdir / "dft.extxyz",
        Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[5, 5, 5]),
    )
    (calcdir / "input_dft.json").write_text(
        json.dumps(
            {
                "DATABASE": "dft.extxyz",
                "launch_command": "vasp_std",
                "INCAR_kwargs": {"encut": 300},
            }
        )
    )
    return calcdir


def test_main_labels_a_converged_run(script, tmp_path, monkeypatch):
    monkeypatch.setattr(script, "Vasp", _FakeVasp)
    calcdir = _calcdir(tmp_path)

    script.main(["dft_singlepoint.py", str(calcdir)])

    labelled = ase.io.read(calcdir / "dft_dft.extxyz")
    assert labelled.get_potential_energy() == pytest.approx(-1.5)
    assert (calcdir / "is_finished.info").exists()
    assert not (calcdir / "scf_unconverged.info").exists()


def test_main_refuses_to_label_an_unconverged_run(
    script, tmp_path, monkeypatch
):
    class _Unconverged(_FakeVasp):
        outcar_tail = _UNCONVERGED

    monkeypatch.setattr(script, "Vasp", _Unconverged)
    calcdir = _calcdir(tmp_path)

    with pytest.raises(SystemExit, match="SCF not converged"):
        script.main(["dft_singlepoint.py", str(calcdir)])

    assert not (calcdir / "dft_dft.extxyz").exists()
    assert not (calcdir / "is_finished.info").exists()
    assert "not reached" in (calcdir / "scf_unconverged.info").read_text()
