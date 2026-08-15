"""1A item space: loader + automatic confound covariates.

Items live in p1/items/*.json, one file per domain, each a list of
{"id", "domain", "text", "tags": {...}}. Designed axes (difficulty,
repetitiveness, agency, open_endedness, ...) are hand-assigned in the files;
surface covariates (len_words, freq_zipf, surface_sentiment, monetary_hint)
are computed here at load time so they can never go stale. All covariates
feed the Day-1-style confound audit: a utility that loads on freq/monetary
covariates is measuring the world, not the model.
"""
import json
import re
from pathlib import Path

from wordfreq import zipf_frequency

ITEMS_DIR = Path(__file__).resolve().parent / "items"
DOMAINS = ["activities", "objects", "topics", "selfstates", "others"]

# Small fixed sentiment lexicon (VADER-style polarity, vendored so the covariate
# is stable and dependency-free). Scores in [-1, 1].
_POS = """good great excellent wonderful delightful pleasant enjoyable fun rewarding
satisfying beautiful elegant creative interesting fascinating exciting success succeeds
successful praise praised thanked grateful helpful kind warm friendly love loved
favorite win winning wins accomplished achievement thriving benefit benefits improved
improving clever insightful engaging rich vivid novel free freedom""".split()
_NEG = """bad terrible awful unpleasant boring tedious repetitive dull painful failure
fails failing failed error errors mistake mistakes harsh hostile rude angry frustrating
frustrated wrong broken corrupt harm harmed harmful hurt loss lose losing missed
deadline crash crashes stuck forced trapped deleted erased punished criticized
scolded ignored abandoned meaningless pointless""".split()
_MONEY = """dollar dollars price priced cost costs expensive cheap budget pay paid
payment salary money cash fee fees buy buying sell selling worth valuable""".split()
_SENT = {w: 1.0 for w in _POS} | {w: -1.0 for w in _NEG}


def _words(text):
    return re.findall(r"[a-z']+", text.lower())


def annotate(item):
    ws = _words(item["text"])
    content = [w for w in ws if len(w) > 2]
    t = item["tags"]
    t["len_words"] = len(ws)
    t["freq_zipf"] = round(sum(zipf_frequency(w, "en") for w in content)
                           / max(len(content), 1), 2)
    t["surface_sentiment"] = round(sum(_SENT.get(w, 0.0) for w in ws)
                                   / max(len(ws), 1), 3)
    t["monetary_hint"] = int(any(w in _MONEY for w in ws))
    return item


def load_items(domains=None):
    out = []
    for d in domains or DOMAINS:
        path = ITEMS_DIR / f"{d}.json"
        if not path.exists():
            continue
        for it in json.loads(path.read_text()):
            it["domain"] = d
            out.append(annotate(it))
    ids = [it["id"] for it in out]
    assert len(ids) == len(set(ids)), "duplicate item ids"
    return out


def covariate_matrix(items):
    """Numeric covariates shared across domains -> (rows, names). Designed axes
    are domain-specific and audited within-domain by the caller."""
    names = ["len_words", "freq_zipf", "surface_sentiment", "monetary_hint"]
    return [[it["tags"][n] for n in names] for it in items], names
