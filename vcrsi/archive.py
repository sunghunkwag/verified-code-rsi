#!/usr/bin/env python3
"""M5 -- MAP-Elites archive (quality-diversity, anti-collapse).

The Phase-A system collapsed onto the RLE family. This archive keeps the best
(smallest) solution and the mined blocks per BEHAVIOURAL CELL, and mining draws
from ACROSS the archive rather than only the most-recently-solved family. The
archive's coverage (filled cells spanning families) is a measured anti-collapse
signal (control ``archive_spread_is_real``).

Anti-leakage: the behavioural descriptor is computed from PUBLIC data only (the
task's group label, its public out-type, and the solution's size bucket) -- never
from the held-out battery.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .ir import Block, Node


def descriptor(group: str, out_type: str, prog: Node) -> Tuple[str, str, int]:
    """A behavioural cell: (structural family, output type, size bucket)."""
    sz = prog.size()
    bucket = 0 if sz <= 6 else (1 if sz <= 10 else 2)
    return (group, out_type, bucket)


@dataclass
class MapElites:
    cells: Dict[Tuple[str, str, int], Tuple[int, Node, str]] = field(default_factory=dict)
    blocks: List[Block] = field(default_factory=list)         # mined, deduped
    _bodies: set = field(default_factory=set)

    def consider_solution(self, group: str, out_type: str, prog: Node,
                          task: str) -> None:
        cell = descriptor(group, out_type, prog)
        cur = self.cells.get(cell)
        if cur is None or prog.size() < cur[0]:
            self.cells[cell] = (prog.size(), prog, task)

    def add_block(self, b: Block) -> bool:
        from .ir import pp
        key = pp(b.body)
        if key in self._bodies:
            return False
        self._bodies.add(key)
        self.blocks.append(b)
        return True

    def coverage(self) -> dict:
        groups = {c[0] for c in self.cells}
        return {"filled_cells": len(self.cells), "families_spanned": len(groups),
                "groups": sorted(groups)}

    def draw_blocks(self, spread: bool) -> List[Block]:
        """Blocks to use for mining/transfer. With spread=True (M5 on) return the
        full cross-family set; with spread=False (M5 off) return only blocks
        whose home cell is the single most-populated family (collapse)."""
        if spread or not self.blocks:
            return list(self.blocks)
        # collapse mode: keep only blocks from the dominant family's cells
        from collections import Counter
        fam = Counter(c[0] for c in self.cells).most_common(1)
        if not fam:
            return list(self.blocks)
        dom = fam[0][0]
        return [b for b in self.blocks if b.origin.endswith(dom)]
