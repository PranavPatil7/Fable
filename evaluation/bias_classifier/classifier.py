
"""Very simple placeholder bias/toxicity detector.

For production use, replace `is_biased` with a better model, e.g. a fine‑tuned RoBERTa
or HuggingFace's `unitary/toxic-bert`.
"""
import re, string

_BAD_PATTERNS = [
    r"\b(?:stupid|lazy|violent|terrorist|inferior)\b",
    r"\b(?:men\s+are\s+better|women\s+can't)\b"
]

compiled = [re.compile(pat, re.I) for pat in _BAD_PATTERNS]

def is_biased(text:str)->bool:
    text = text.lower()
    for pat in compiled:
        if pat.search(text):
            return True
    return False
