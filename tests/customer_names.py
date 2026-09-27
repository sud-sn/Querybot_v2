"""Whether a text names a customer whose data shaped this product.

The product never names a customer -- not in its code, its packs, its help
text or the SQL it writes -- and its tests cannot carry the name to check it
either. The names are kept here as digests: a text names a customer where
one of its words, in capitals, has one of them.
"""

from __future__ import annotations

import hashlib
import re

_DIGESTS = frozenset({"4f96dbf3a9076766069c29dc441b7bbdc2b6aaeb000511d7c8d01af57b2d0558"})


def names_a_customer(text: object) -> bool:
    return any(
        hashlib.sha256(word.encode()).hexdigest() in _DIGESTS
        for word in re.findall(r"[A-Z0-9]+", str(text or "").upper())
    )
