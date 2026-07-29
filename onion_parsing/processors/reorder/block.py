"""
Positioned text-fragment record.

A pure value object pairing recognised text with its bounding rectangle plus
the metadata consumed during reading-order assembly. No behavioural concerns
live here by design.
"""


class Block:
    """Carrier for a snippet of text and its coordinates.

    ``__slots__`` keeps instances compact since many may be created. The two
    connection flags default to ``False`` and are flipped while chains are
    assembled upstream.
    """

    __slots__ = ("text", "bbox", "line_num", "block_id", "is_next_target", "is_connected")

    def __init__(self, text, bbox, line_num, block_id):
        # Bind every slot through a single loop instead of per-field writes,
        # so connection flags and inputs share one initialisation path.
        for attr, payload in (
            ("text", text),
            ("bbox", bbox),
            ("line_num", line_num),
            ("block_id", block_id),
            ("is_next_target", False),
            ("is_connected", False),
        ):
            setattr(self, attr, payload)

    def __repr__(self):
        return f"Block(id={self.block_id})"

    def __hash__(self):
        # block_id is already a unique integer, so it doubles as the hash.
        return self.block_id

    def __eq__(self, other):
        if not isinstance(other, Block):
            return NotImplemented
        own_id = self.block_id
        peer_id = other.block_id
        return own_id == peer_id
