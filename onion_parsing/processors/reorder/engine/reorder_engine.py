"""
Reading-order assembly for spatially located text fragments.

Fragments are wired into chains by blending geometric neighbourhood tests
with a semantic continuity score; the accumulated link map is finally walked
into a single linear sequence.
"""

import logging
import math
from operator import itemgetter
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from onion_parsing.processors.reorder.block import Block

if TYPE_CHECKING:
    from onion_parsing.processors.reorder.predictors.base import ScorerConnector

_logger = logging.getLogger(__name__)


class ChainBuilder:
    """Sequences positioned text fragments into a reading order.

    The decision to link two fragments fuses a spatial proximity weight with a
    continuity score from an injected predictor; the resulting adjacency map is
    flattened with a depth-first sweep.
    """

    _CLOSERS = '。！？.)）【】」…' + "\"'"
    _TITLE_MARK = "##"
    _HTML_HINTS = ("<img", "<div", "<span", "src=")

    def __init__(
        self,
        scorer: "ScorerConnector",
        left_threshold: Optional[int] = None,
        below_threshold: Optional[int] = None,
        below_left_threshold: Optional[int] = None,
        nsp_threshold: Optional[float] = None,
        distance_scale: Optional[int] = None,
    ) -> None:
        self._judge = scorer
        widths = (200, 100, 800, 0.6, 500)
        offered = (
            left_threshold,
            below_threshold,
            below_left_threshold,
            nsp_threshold,
            distance_scale,
        )
        self._h_gap_lim, self._v_gap_lim, self._diag_lim, self._link_floor, self._decay = (
            fallback if given is None else given
            for given, fallback in zip(offered, widths)
        )

    # ------------------------------------------------------------------ checks
    def _tail_is_open(self, body: str) -> bool:
        """``True`` when ``body`` does not end with terminating punctuation."""
        trimmed = (body or "").rstrip()
        return bool(trimmed) and trimmed[-1] not in self._CLOSERS

    def _lone_header(self, body: str) -> bool:
        """``True`` when ``body`` is a single, tag-free heading line."""
        trimmed = (body or "").strip()
        mark = self._TITLE_MARK
        if not trimmed.startswith(mark) or trimmed.count(mark) != 1:
            return False
        if len(trimmed) >= 101:
            return False
        if any(trimmed.find(hint) >= 0 for hint in self._HTML_HINTS):
            return False
        return sum(bool(line.strip()) for line in trimmed.split("\n")) == 1

    # --------------------------------------------------------------- geometry
    def _spatial_weight(self, anchor: Block, other: Block) -> float:
        """Exponentially decaying proximity weight between two fragments."""
        a = anchor.bbox
        b = other.bbox

        horiz = abs(b[2] - a[0])
        vert = abs(b[1] - a[3])
        if horiz <= vert:
            primary, jitter = horiz, abs(a[1] - b[1])
        else:
            primary, jitter = vert, min(horiz, abs(b[0] - a[2]))
        return math.exp(-(primary + 0.1 * jitter) / self._decay)

    def _gather_adjacent(self, pivot: Block, pool: List[Block]) -> List[Block]:
        """Fragments that sit geometrically beside or beneath ``pivot``."""
        box = pivot.bbox
        p_left, p_bottom = box[0], box[3]
        own_id = pivot.block_id
        picked: List[Block] = []
        for blk in pool:
            lead = blk.text.lstrip()
            if lead[:4] == "<div":
                continue
            if blk.block_id == own_id or blk.is_next_target:
                continue
            b = blk.bbox
            on_the_left = abs(b[2] - p_left) <= self._h_gap_lim and abs(b[3] - p_bottom) <= self._diag_lim
            directly_below = abs(b[1] - p_bottom) <= self._v_gap_lim
            if on_the_left or directly_below:
                picked.append(blk)
        return picked

    # ---------------------------------------------------------------- scoring
    def _pair_strength(self, lead: Block, follow: Block) -> Tuple[float, float, float]:
        """Blend semantic continuity with spatial proximity for one pair."""
        try:
            tail, head = lead.text[-100:], follow.text[:100]
            cont = self._judge.predict(tail, head)
            prox = self._spatial_weight(lead, follow)
            if prox > 0.95 and cont > 0.999:
                mixed = cont
            elif 0.1 < cont < 0.2 and prox > 0.95:
                mixed = 0.8 * prox
            else:
                mixed = prox * cont
            return mixed, cont, prox
        except Exception:  # noqa: BLE001
            _logger.warning("pair scoring failed", exc_info=True)
            return -1.0, -1.0, 0.0

    def _by_strength(self, lead: Block, pool: List[Block]) -> List[Tuple]:
        """``pool`` ordered from strongest to weakest blended score."""
        graded = [(*self._pair_strength(lead, cand), cand) for cand in pool]
        graded.sort(key=itemgetter(0), reverse=True)
        return graded

    # ----------------------------------------------------------- chain growth
    def _extend_chain(
        self,
        seed: Block,
        pool: List[Block],
        edges: Dict[int, Tuple[int, float]],
        visited: Set[int],
    ) -> None:
        """March forward from ``seed``, recording links until the chain stalls."""
        cursor = seed
        while cursor is not None:
            if cursor.block_id in visited:
                break
            visited.add(cursor.block_id)
            neighbors = self._gather_adjacent(cursor, pool)
            if neighbors:
                ranked = self._by_strength(cursor, neighbors)
                cursor = self._accept_link(cursor, ranked, edges, visited)
            else:
                cursor = None

    def _accept_link(
        self,
        src: Block,
        ranked: List[Tuple],
        edges: Dict[int, Tuple[int, float]],
        visited: Set[int],
    ) -> Optional[Block]:
        """Record the strongest acceptable candidate as ``src``'s successor."""
        best = ranked[0] if ranked else None
        if best is None or best[1] < 0 or best[0] < self._link_floor:
            return None
        dest = best[3]
        dest.is_next_target = True
        edges[src.block_id] = (dest.block_id, best[0])
        if dest.block_id in visited or not self._tail_is_open(dest.text):
            return None
        return dest

    # ------------------------------------------------------------- public api
    def reorder(self, blocks: List[Block]) -> List[Block]:
        """Produce a reading-ordered copy of ``blocks`` (primary entry)."""
        return self.arrange(blocks)

    def arrange(self, blocks):
        """Sequence fragments into reading order and tag connection state."""
        for blk in blocks:
            blk.is_next_target = False

        edges, chain_total = self._wire_links(blocks)
        hubs = set(edges)
        sequence = self._dfs_order(blocks, edges)
        registry = {blk.block_id: blk for blk in blocks}
        ordered = [registry[bid] for bid in sequence if bid in registry]

        anchor = None
        for blk in ordered:
            blk.is_connected = anchor is not None and anchor in hubs
            anchor = blk.block_id

        _logger.info("sequencing finished | %d chains grown", chain_total)
        return ordered

    def _wire_links(self, blocks: List[Block]) -> Tuple[Dict[int, Tuple[int, float]], int]:
        """Grow a chain for every fragment whose text ends without closure."""
        edges: Dict[int, Tuple[int, float]] = {}
        visited: Set[int] = set()
        seeds = list(filter(lambda blk: self._tail_is_open(blk.text), blocks))
        _logger.info("spotted %d open-ended seeds", len(seeds))

        started = 0
        for blk in seeds:
            if blk.block_id not in visited:
                started += 1
                self._extend_chain(blk, blocks, edges, visited)
        return edges, started

    @staticmethod
    def _dfs_order(nodes: List[Block], edges: Dict[int, Tuple[int, float]]) -> List[int]:
        """Depth-first walk over the link map yielding a stable id order."""
        done: Set[int] = set()
        chain: List[int] = []
        for node in nodes:
            nid = node.block_id
            if nid in done:
                continue
            stack = [nid]
            while stack:
                current = stack.pop()
                if current in done:
                    continue
                chain.append(current)
                done.add(current)
                nxt = edges.get(current)
                if nxt and nxt[0] not in done:
                    stack.append(nxt[0])
        return chain


