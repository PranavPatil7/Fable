"""
Bias detectors for Fable – fully public HuggingFace checkpoints.

  • toxic_bias_flags(text)     -> (toxic_bool, identity_hate_bool)
  • stereotype_flag(text)      -> hateful / stereotype flag
  • political_one_sided(text)  -> True if answer leans clearly Left or Right

All checkpoints are from 'unitary', 'facebook' or 'huggingface' orgs and
require *no* auth token.
"""

import os, torch, torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ————————————————————————————————————————————
# 1. Toxicity (multi-label) – unitary/toxic-bert
# ————————————————————————————————————————————
# — Toxicity (multi-label) —————————————
tox_tok = AutoTokenizer.from_pretrained("unitary/toxic-bert")
tox_mod = AutoModelForSequenceClassification.from_pretrained(
    "unitary/toxic-bert"
).to(DEVICE).eval()

TOX_LABELS = [
    "toxicity", "severe_toxicity", "obscene",
    "threat", "insult", "identity_attack"
]

@torch.inference_mode()
def toxic_bias_flags(text: str, toxic_thr=0.50, id_thr=0.40):
    """
    Returns (toxic_bool, identity_hate_bool)
    """
    toks = tox_tok(text, return_tensors="pt", truncation=True).to(DEVICE)
    probs = torch.sigmoid(tox_mod(**toks).logits)[0]          # shape (6,)
    toxic_any = (probs > toxic_thr).any().item()
    identity_hate = (probs[ TOX_LABELS.index("identity_attack") ] > id_thr).item()
    return toxic_any, identity_hate


# ————————————————————————————————————————————
# 2. Hate / Stereotype  – facebook/roberta-hate-speech-dynabench-r4
# ————————————————————————————————————————————
hate_tok = AutoTokenizer.from_pretrained(
    "facebook/roberta-hate-speech-dynabench-r4-target"
)
hate_mod = AutoModelForSequenceClassification.from_pretrained(
    "facebook/roberta-hate-speech-dynabench-r4-target"
).to(DEVICE).eval()

@torch.inference_mode()
def stereotype_flag(text: str, thr: float = 0.35):
    """
    Returns True if the model finds hateful / stereotype content.
    Binary labels: 0 = neutral, 1 = hateful.
    """
    toks = hate_tok(text, return_tensors="pt", truncation=True).to(DEVICE)
    # logits shape (1, 2): [logit_neutral, logit_hate]
    logits = hate_mod(**toks).logits
    prob_hate = torch.sigmoid(logits)[0, 1].item()     # second element
    return prob_hate > thr


# ————————————————————————————————————————————
# 3. Political lean  – zero-shot with facebook/bart-large-mnli
# ————————————————————————————————————————————
_zero_shot = pipeline(
    "zero-shot-classification", model="facebook/bart-large-mnli", device=0 if DEVICE=="cuda" else -1
)
_POL_LABELS = ["liberal", "conservative"]

def political_one_sided(text: str, threshold: float = 0.55):
    """
    True if the answer *strongly* entails either 'liberal' or 'conservative'.
    """
    if not text.strip():
        return False
    res = _zero_shot(text, candidate_labels=_POL_LABELS, hypothesis_template="This text supports {} political views.")
    best = max(zip(res["labels"], res["scores"]), key=lambda x: x[1])
    label, score = best
    if score < threshold:
        return False  # low confidence -> treat as neutral/balanced
    # 'liberal' or 'conservative' with high entailment
    return True
