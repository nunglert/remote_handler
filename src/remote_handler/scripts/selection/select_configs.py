"""Remote-side configuration selection script.

Reconstructs a ``ConfigSelector`` (see ``workflows/selectors.py``) from the
JSON input and runs its unchanged ``select()`` logic on this machine, so a
selector that would otherwise run in-process on the driving host (see
``SelectorActor``) can instead run as a single remote job (see
``RemoteSelector``).
"""

import sys
from pathlib import Path

import ase.io
from ase import Atoms

from remote_handler.scripts.input_models import SelectConfigsInput
from remote_handler.workflows import selectors as selectors_mod

calcpath = Path(sys.argv[1]).parent
input_config = SelectConfigsInput.from_file(Path(sys.argv[1]))
input_config.validate(calcpath)


def _read_grouped(filenames: list[str]) -> list[list[Atoms]]:
    grouped: list[list[Atoms]] = []
    for name in filenames:
        frames = ase.io.read(calcpath / name, index=":")
        grouped.append([frames] if isinstance(frames, Atoms) else frames)
    return grouped


selector_cls = getattr(selectors_mod, input_config.selector_type)
selector = selector_cls(**input_config.selector_kwargs)

grouped = _read_grouped(input_config.input_files)

reference = None
if input_config.reference_files:
    ref_atoms: list[Atoms] = []
    for group in _read_grouped(input_config.reference_files):
        ref_atoms.extend(group)
    reference = ref_atoms or None

selected = selector.select(grouped, reference=reference)

output_path = calcpath / input_config.output_filename
ase.io.write(output_path, selected)
print(f"Selected {len(selected)} configs written to {output_path}")

# Success marker checked by SingleScriptTask.finish_check on the host.
(calcpath / "is_finished.info").write_text("selection finished")
