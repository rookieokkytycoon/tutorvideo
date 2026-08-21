"""pairing.py — two models reason over the same source, and disagree in public.

A single extractor gives you one opinion with no error bar. Ask Claude and
DeepSeek the same question about the same transcript and three things fall
out for free:

    both agree          -> support += 1   (a corroborated step)
    both silent         -> nothing        (correctly dropped)
    exactly one asserts -> a PENDING CORRECTION, authored by the model that
                           disagreed, queued for review

That third case is the point. It is the correction-capture loop running
without waiting for a human to notice — the graph arrives already knowing
which of its own steps are shaky, and a reviewer spends their time only on
the contested ones.

Cost note: DeepSeek is cheap enough that pairing roughly doubles extraction
cost while roughly halving the fraction of the graph nobody has checked.

    export DEEPSEEK_API_KEY=sk-...        # platform.deepseek.com
    python pairing.py "https://www.youtube.com/watch?v=..."
"""

import json
import os
import re
import sys

import requests

import openclaw
from hivemind import same_thing, similarity, toks

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
# deepseek-reasoner is the chain-of-thought model; slower, better on
# "is this step actually required?" style judgement.

# --- the all-Claude pairing -------------------------------------------------
# Two passes have to disagree for the loop to mean anything, and the usual
# lever for that is gone: temperature is rejected outright on current Claude
# models. Independence therefore comes from the two things still available —
# DIFFERENT WEIGHTS and a DIFFERENT PROMPT. Pass A is the fast extractor;
# pass B is a slower, evidence-first reader on a stronger model that must
# quote the transcript for every step it keeps. Where they disagree is
# exactly where a single pass would have been confidently wrong.
PAIR_A_MODEL = os.environ.get("PAIR_A_MODEL", "claude-haiku-4-5")
PAIR_B_MODEL = os.environ.get("PAIR_B_MODEL", "claude-sonnet-5")

PAIR_B_SYS = openclaw.MINE_SYS + """

YOU ARE THE SECOND READER. A first pass has already extracted this source;
your job is not to agree with it — you cannot see it — but to be harder to
fool than it was.

Before you output anything, work through the source and decide, for each
candidate step, whether the speaker ACTUALLY STATED IT or whether it is
something a reasonable person would merely assume. Keep only what was
stated. Specifically:
- Drop any step you cannot point to a specific moment in the source for.
- Drop setup, sponsor reads, sign-offs, and "in this video we'll..." framing.
- Do NOT smooth a vague gesture ("and then you just sort it out") into a
  crisp instruction. If the source is vague there, the step does not exist.
- If the source contradicts itself, keep the correction, not the first take.
Being short is correct when the source is thin. Do not pad to reach a count.
"""

AGREE = 0.6          # two phrasings this close are the same instruction


class PairError(Exception):
    pass


def deepseek_available():
    return bool(os.environ.get("DEEPSEEK_API_KEY"))


def _deepseek(system, user, model=None, max_tokens=3000):
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise PairError("DEEPSEEK_API_KEY is not set")
    try:
        r = requests.post(
            DEEPSEEK_URL,
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
            json={"model": model or DEEPSEEK_MODEL,
                  "max_tokens": max_tokens,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}]},
            timeout=180)
    except requests.RequestException as e:
        raise PairError(f"DeepSeek unreachable: {e}")
    if r.status_code >= 400:
        try:
            msg = r.json().get("error", {}).get("message", r.text[:200])
        except ValueError:
            msg = r.text[:200]
        raise PairError(f"DeepSeek HTTP {r.status_code}: {msg}")
    try:
        return r.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as e:
        raise PairError(f"unexpected DeepSeek response: {e}")


# ------------------------------------------------------------ the pairing

def extract_pair(header, body, claude_key, model=None, provider=None):
    """Two independent extractions of the same source. -> (a, b, errors).

    provider="claude"   both passes on Claude, different model + different
                        prompt (the default — no second vendor required)
    provider="deepseek" second pass on DeepSeek, same prompt
    """
    provider = provider or ("deepseek" if deepseek_available() else "claude")
    errs = {"provider": provider}
    try:
        a = openclaw._extract(claude_key, header, body,
                              model or PAIR_A_MODEL)
    except Exception as e:
        a, errs["a"] = None, str(e)[:200]

    try:
        if provider == "deepseek":
            raw = _deepseek(openclaw.MINE_SYS, f"{header}\n\n{body}")
            b = openclaw._loads(raw)
            b["steps"] = [s for s in (b.get("steps") or [])
                          if isinstance(s, dict)]
        else:
            # same source, stronger model, adversarial reader prompt
            b = openclaw._extract(claude_key, header, body, PAIR_B_MODEL,
                                  system=PAIR_B_SYS)
    except Exception as e:
        b, errs["b"] = None, str(e)[:200]
    return a, b, errs


def reconcile(doc_a, doc_b):
    """Two extractions -> one doc plus a verdict per step.

    Every step is labelled:
      agreed   both models produced it        -> ingest with support 2
      only_a   Claude alone                   -> ingest, queue a dispute
      only_b   DeepSeek alone                 -> queue as a missing-step
                                                 correction, do not ingest
    Nothing is thrown away silently; the disagreements become the queue.
    """
    a_steps = (doc_a or {}).get("steps") or []
    b_steps = (doc_b or {}).get("steps") or []
    if not doc_a and not doc_b:
        raise PairError("both models failed")
    if not doc_b:                      # solo run, everything unconfirmed
        return doc_a, [{"step": s, "verdict": "only_a", "match": None}
                       for s in a_steps]
    if not doc_a:
        return doc_b, [{"step": s, "verdict": "only_b", "match": None}
                       for s in b_steps]

    used, verdicts = set(), []
    for s in a_steps:
        ta = toks(str(s.get("text", "")))
        best, score = None, 0.0
        for j, t in enumerate(b_steps):
            if j in used:
                continue
            sim = similarity(ta, toks(str(t.get("text", ""))))
            if sim > score:
                best, score = j, sim
        if best is not None and same_thing(
                ta, toks(str(b_steps[best].get("text", ""))), AGREE):
            used.add(best)
            verdicts.append({"step": s, "verdict": "agreed",
                             "match": b_steps[best], "sim": round(score, 3)})
        else:
            verdicts.append({"step": s, "verdict": "only_a", "match": None,
                             "sim": round(score, 3)})
    for j, t in enumerate(b_steps):
        if j not in used:
            verdicts.append({"step": t, "verdict": "only_b", "match": None})

    merged = dict(doc_a)
    merged["steps"] = [v["step"] for v in verdicts if v["verdict"] != "only_b"]
    # a tool or warning only one model saw is still worth keeping; they are
    # additive facts, not competing claims about what to do
    for field in ("tools", "warnings", "tags"):
        merged[field] = sorted(set((doc_a.get(field) or [])
                                   + (doc_b.get(field) or [])))
    return merged, verdicts


def ingest_paired(hm, doc, verdicts, graph_path=None, author=None):
    """Write a reconciled extraction into the graph, evidence and all.

    Agreement raises confidence; disagreement lands in the review queue
    attached to the exact step it is about, which is what makes the backlog
    actionable instead of a pile of "something is wrong somewhere".
    """
    tid = hm.ingest(openclaw.hivemind_doc(doc))
    step_ids = hm._steps_of(tid)
    by_text = {hm.g.nodes[s]["text"]: s for s in step_ids}

    counts = {"agreed": 0, "only_a": 0, "only_b": 0}
    for v in verdicts:
        text = str(v["step"].get("text", "")).strip()
        counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1
        sid = by_text.get(text) or next(
            (s for s in step_ids
             if same_thing(toks(text), toks(hm.g.nodes[s]["text"]), AGREE)),
            None)

        if v["verdict"] == "agreed" and sid:
            hm.correct(sid, kind="confirm", author=author or "second-reader", auto=True,
                       text="independently extracted the same step")
        elif v["verdict"] == "only_a" and sid:
            hm.correct(sid, kind="wrong", author=author or "second-reader", auto=True,
                       text="the second model did not find this step in the "
                            "source — it may be inferred rather than stated")
        elif v["verdict"] == "only_b":
            hm.correct(tid, kind="missing", author=author or "second-reader", auto=True,
                       text="the second model found a step the first missed",
                       replacement=text)
    if graph_path:
        hm.save(graph_path)
    return {"task": tid, "verdicts": counts,
            "agreement": round(counts["agreed"]
                               / max(sum(counts.values()), 1), 3)}


def mine_paired(target, claude_key, hm=None, graph_path=None, model=None,
                provider=None):
    """The whole paired pipeline: source -> two extractions -> graph."""
    t = str(target).strip()
    if openclaw.is_youtube(t):
        vid, title, rows = openclaw.fetch_transcript(t)
        header = f"Video title: {title or '(unknown)'}"
        body = f"Transcript:\n{openclaw.transcript_text(rows)}"
        extra = {"video_id": vid, "url": f"https://www.youtube.com/watch?v={vid}",
                 "source": f"youtube:{vid}", "video_title": title}
    elif re.match(r"https?://", t, re.I):
        title, text = openclaw.fetch_article(t)
        header = (f"Article title: {title or '(unknown)'}\nThis is a written "
                  f"how-to article. Omit the \"at\" field entirely.")
        body = f"Article:\n{text[:14000]}"
        extra = {"url": t, "source": t}
    else:
        hit = openclaw.search_with_transcripts(t, want=1)[0]
        return mine_paired(hit["id"], claude_key, hm, graph_path, model,
                           provider)

    a, b, errs = extract_pair(header, body, claude_key, model, provider)
    doc, verdicts = reconcile(a, b)
    doc = openclaw._finish(doc, extra.get("video_title", ""), extra)

    reader = "deepseek" if errs.get("provider") == "deepseek" \
        else PAIR_B_MODEL
    out = {"doc": doc, "verdicts": verdicts, "errors": errs,
           "lesson": openclaw.doc_to_lesson(doc),
           "paired": bool(a and b), "second_reader": reader}
    if hm is not None:
        out.update(ingest_paired(hm, doc, verdicts, graph_path, author=reader))
        out["stats"] = hm.stats()
    return out


def _main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    from hivemind import Hivemind
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "hivemind_graph.json")
    hm = Hivemind(path)
    out = mine_paired(argv[1], os.environ.get("ANTHROPIC_API_KEY", ""), hm, path)
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("doc", "verdicts", "lesson")}, indent=2))
    for v in out["verdicts"]:
        print(f"  {v['verdict']:8} {str(v['step'].get('text', ''))[:66]}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
