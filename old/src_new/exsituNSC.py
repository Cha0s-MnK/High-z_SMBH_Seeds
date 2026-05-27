"""Ex-situ satellite-NSC preprocessing helpers for the NSC formation stage.

The helpers in this module intentionally stop at catalogue construction. They
infer branch-release times from the fixed merger tree, classify non-MPB GCs by
whether they can sink to their own satellite centre before release, and expose
small audited pieces of metadata to ``main_spatial.py``. The downstream
``evo.py`` solver still receives a normal 13-column GCini-compatible table.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import *

@dataclass(frozen=True)
class TreeTable:
    log_mh: np.ndarray
    mass_msun: np.ndarray
    first_prog_id: np.ndarray
    subhalo_id: np.ndarray
    branch_id: np.ndarray
    redshift: np.ndarray
    spin_norm: np.ndarray
    mpb_branch_id: int
    msub_z0_msun: float

def _empty_tree() -> TreeTable:
    empty_float = np.array([], dtype=float)
    empty_int = np.array([], dtype=int)
    return TreeTable(
        log_mh=empty_float,
        mass_msun=empty_float,
        first_prog_id=empty_int,
        subhalo_id=empty_int,
        branch_id=empty_int,
        redshift=empty_float,
        spin_norm=empty_float,
        mpb_branch_id=-1,
        msub_z0_msun=0.0,
    )

def _make_tree_table(
    *,
    log_mh: np.ndarray,
    first_prog_id: np.ndarray,
    subhalo_id: np.ndarray,
    branch_id: np.ndarray,
    redshift: np.ndarray,
    spin_norm: np.ndarray,
    mpb_branch_id: int | None = None,
    msub_z0_msun: float | None = None,
) -> TreeTable:
    if len(log_mh) == 0:
        return _empty_tree()

    log_mh = np.asarray(log_mh, dtype=float)
    mass_msun = np.power(10.0, log_mh)
    first_prog_id = np.asarray(first_prog_id, dtype=int)
    subhalo_id = np.asarray(subhalo_id, dtype=int)
    branch_id = np.asarray(branch_id, dtype=int)
    redshift = np.asarray(redshift, dtype=float)
    spin_norm = np.asarray(spin_norm, dtype=float)

    if mpb_branch_id is None:
        mpb_branch_id = int(branch_id[np.argmax(log_mh)])
    if msub_z0_msun is None:
        main_mask = branch_id == int(mpb_branch_id)
        if np.any(main_mask):
            msub_z0_msun = float(np.max(mass_msun[main_mask]))
        else:
            msub_z0_msun = float(np.max(mass_msun))

    return TreeTable(
        log_mh=log_mh,
        mass_msun=mass_msun,
        first_prog_id=first_prog_id,
        subhalo_id=subhalo_id,
        branch_id=branch_id,
        redshift=redshift,
        spin_norm=spin_norm,
        mpb_branch_id=int(mpb_branch_id),
        msub_z0_msun=float(msub_z0_msun),
    )


def read_full_tree(tree_path: Path) -> TreeTable:
    """Read the full fixed tree without final-redshift truncation."""

    log_mh: list[float] = []
    first_prog_id: list[int] = []
    subhalo_id: list[int] = []
    branch_id: list[int] = []
    redshift: list[float] = []
    spin_norm: list[float] = []

    with Path(tree_path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            if line_no == 0:
                continue
            cols = line.split()
            if len(cols) < 9:
                continue
            log_mh.append(float(cols[0]))
            first_prog_id.append(int(cols[1]))
            subhalo_id.append(int(cols[2]))
            branch_id.append(int(cols[3]))
            redshift.append(float(cols[5]))
            sx = float(cols[6])
            sy = float(cols[7])
            sz = float(cols[8])
            spin_norm.append(float(np.sqrt(sx * sx + sy * sy + sz * sz)))

    return _make_tree_table(
        log_mh=np.asarray(log_mh, dtype=float),
        first_prog_id=np.asarray(first_prog_id, dtype=int),
        subhalo_id=np.asarray(subhalo_id, dtype=int),
        branch_id=np.asarray(branch_id, dtype=int),
        redshift=np.asarray(redshift, dtype=float),
        spin_norm=np.asarray(spin_norm, dtype=float),
    )

def filter_tree_for_formation(full: TreeTable, final_redshift: float) -> TreeTable:
    """Return the current formation-view tree while preserving full-tree metadata."""

    if len(full.log_mh) == 0:
        return _empty_tree()

    keep = full.redshift >= float(final_redshift)

    return _make_tree_table(
        log_mh=full.log_mh[keep],
        first_prog_id=full.first_prog_id[keep],
        subhalo_id=full.subhalo_id[keep],
        branch_id=full.branch_id[keep],
        redshift=full.redshift[keep],
        spin_norm=full.spin_norm[keep],
        mpb_branch_id=full.mpb_branch_id,
        msub_z0_msun=full.msub_z0_msun,
    )