"""
src/label_utils.py
===================
Shared, negation-aware classifier-label matching.

WHY THIS EXISTS: a plain substring check like `"drowsy" in label.lower()`
silently matches BOTH "Drowsy" and "Non Drowsy" (or "Not Drowsy",
"No-Yawn", etc.) -- because "drowsy" is a substring of "non drowsy" too.
That bug was firing a DROWSINESS_ALERT off frames the classifier itself
labeled "Non Drowsy" at 94%+ confidence (i.e. exactly backwards), which is
what produced the near-constant alert stream on live streams: with a
2-class model, almost every frame above the confidence threshold matched
the "drowsy" hint one way or the other, independent of which class the
model actually picked.

This checks hints against individual tokens (split on space/underscore/
hyphen) and rejects a match when the token immediately preceding the
matched token is a negation word ("non", "not", "no").
"""

import re

_NEGATION_TOKENS = {"non", "not", "no"}


def label_matches_hints(label: str, hints) -> bool:
    """
    True if `label` positively matches one of `hints` (substrings), and is
    NOT immediately preceded by a negation token ("non"/"not"/"no").

    Examples (hints=["drowsy"]):
        "Drowsy"              -> True
        "Non Drowsy"          -> False
        "Not-Drowsy"          -> False
        "No_Drowsy"           -> False
        "Eyes Closed"         -> True   (hints=["closed"])
        "Not Yawning"         -> False  (hints=["yawn"])
    """
    if not label:
        return False
    label_lower = label.lower().strip()
    tokens = re.split(r"[\s_\-]+", label_lower)

    for i, token in enumerate(tokens):
        if any(hint in token for hint in hints):
            prev_token = tokens[i - 1] if i > 0 else ""
            if prev_token in _NEGATION_TOKENS:
                continue  # negated -- e.g. "non drowsy", "not yawning"
            return True
    return False
