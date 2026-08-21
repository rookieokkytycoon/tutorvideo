"""
hivemind.py — the how-to knowledge graph ("hivemind") behind the tutor.

What it does
------------
- INGEST   : structured how-to docs -> typed graph
             task --has_step--> step --precedes--> step
             task --requires--> tool
             task --warns--> warning
             task <--shares_tool--> task        (the "hivemind" cross-links)
- EXTRACT  : raw article text -> structured doc, via Claude (optional)
             or a plain-text heuristic parser (works offline)
- RETRIEVE : question -> best tasks + their steps/tools/related tasks,
             as a text block ready to inject into an LLM prompt
- PERSIST  : save/load the whole graph as JSON

No heavy dependencies: networkx + stdlib. LLM calls are optional and
only used in extract_with_llm() / answer_with_llm().

Quick start
-----------
    from hivemind import Hivemind, SEED_DOCS
    hm = Hivemind()
    for d in SEED_DOCS: hm.ingest(d)
    print(hm.stats())
    print(hm.retrieve("my bicycle chain slipped off, what do I do?"))

Feeding it real data
--------------------
- wikiHow: the open dataset on Hugging Face ("wikihow" summarization
  dataset, CC-licensed w/ attribution) — map each article through
  parse_plain_howto() or extract_with_llm().
- YouTube: youtube-transcript-api -> transcript text -> extract_with_llm().
- Instructables: respect robots.txt; same extraction path.
"""

from __future__ import annotations
import json
import math
import os
import re
import time
from collections import Counter

import networkx as nx
from networkx.readwrite import json_graph

# --------------------------------------------------------------------------
# tokenizing / scoring (dependency-free retrieval)
# --------------------------------------------------------------------------

_STOP = set("""a an and are as at be by can do for from how i in is it my of on
or that the this to what when where which with you your
yang di ke dari dan atau saya aku itu ini apa apakah bagaimana gimana kalau
untuk pada dengan bisa harus sudah lagi nya dong ya sih""".split())

# The graph is indexed in English, but the tutor gets asked in whatever
# language the lesson is running in. Mapping the handful of words that
# actually carry meaning in a how-to question is enough to make Indonesian
# retrieval hit the same nodes — no translation call, no second index.
_SYN = {"bike": "bicycle", "bikes": "bicycle", "cycle": "bicycle",
        "tyre": "tire", "plant": "houseplant",
        # id -> en
        "sepeda": "bicycle", "rantai": "chain", "gir": "cog", "gigi": "cog",
        "ban": "tire", "roda": "wheel", "pedal": "pedal", "sadel": "saddle",
        "bocor": "puncture", "tambal": "patch", "tambalan": "patch",
        "lepas": "slip", "copot": "slip", "putus": "slip", "macet": "stuck",
        "pelumas": "lube", "oli": "lube", "minyak": "lube", "gemuk": "lube",
        "lumasi": "lube", "melumasi": "lube",
        "bersihkan": "clean", "membersihkan": "clean", "cuci": "clean",
        "perbaiki": "repair", "memperbaiki": "repair", "betulkan": "repair",
        "benerin": "repair", "servis": "repair", "rusak": "repair",
        "pasang": "fit", "memasang": "fit", "pompa": "pump",
        "tanaman": "houseplant", "pohon": "houseplant", "daun": "leaf",
        "akar": "root", "akarnya": "root", "pot": "pot", "tanah": "soil",
        "air": "water", "menyiram": "water", "siram": "water",
        "kebanyakan": "overwatering", "layu": "wilt", "busuk": "rot",
        "alat": "tool", "kain": "rag", "lap": "rag", "gunting": "scissors"}


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > 4 and w.endswith(suf):
            w = w[: -len(suf)]
            break
    if len(w) > 3 and w.endswith("e"):
        w = w[:-1]
    return w


def toks(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in _STOP or len(w) < 2:
            continue
        out.append(_stem(_SYN.get(w, w)))
    return out


def jaccard(a, b) -> float:
    """Token overlap, order-free. The similarity everything here is built on:
    it is cheap, needs no model, and is stable as the graph grows."""
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


MIN_SHARED = 3          # below this, agreement is coincidence, not paraphrase


def same_thing(a, b, thresh=0.6) -> bool:
    """Are these two token lists the same instruction, written twice?

    Jaccard alone says no far too often, because paraphrase is mostly a
    LENGTH difference: "Go to jetbrains.com and download the Community
    edition" and "Visit jetbrains.com/pycharm and download the free PyCharm
    Community Edition" share almost everything the shorter one says, yet
    every extra word in the longer one pushes their union up and their
    Jaccard down. Containment measures the thing that actually matters —
    how much of the SHORTER instruction the longer one already contains —
    and the absolute floor stops two three-word stubs matching on "the".
    """
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return False
    shared = len(sa & sb)
    if jaccard(a, b) >= thresh:
        return True
    return (shared >= MIN_SHARED
            and shared / min(len(sa), len(sb)) >= thresh)


def similarity(a, b) -> float:
    """The score behind same_thing(), for reporting and ranking."""
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    contain = len(sa & sb) / min(len(sa), len(sb))
    return round(max(jaccard(a, b), contain), 3)


# Two titles this similar are the same procedure written twice. Set high
# enough that "unclog a sink" and "unclog a shower" stay separate, low enough
# that "fix a slipped bike chain" and "how to fix a dropped bicycle chain"
# collapse into one node instead of two fragments.
TASK_MERGE = 0.55
STEP_MERGE = 0.6          # same threshold logic, one step at a time

_NUM = re.compile(r"^\d+$")

# Two steps can be word-for-word similar and mean opposite things. Merging
# "shift to the smallest cog" into "shift to the largest cog" would leave one
# well-supported instruction that is wrong half the time — the failure mode a
# correctness-critical graph cannot have. A disagreeing term from this table
# vetoes a merge and opens a dispute instead.
ANTONYMS = [
    {"smallest", "largest"}, {"small", "large"}, {"small", "big"},
    {"min", "max"}, {"minimum", "maximum"}, {"least", "most"},
    {"tighten", "loosen"}, {"tight", "loose"}, {"lock", "unlock"},
    {"open", "close"}, {"on", "off"}, {"up", "down"},
    {"left", "right"}, {"front", "rear"}, {"front", "back"},
    {"forward", "backward"}, {"forward", "reverse"},
    {"push", "pull"}, {"raise", "lower"}, {"add", "drain"},
    {"install", "remove"}, {"attach", "detach"}, {"connect", "disconnect"},
    {"increase", "decrease"}, {"hot", "cold"}, {"positive", "negative"},
    {"inward", "outward"}, {"above", "below"}, {"before", "after"},
]
# A prefix that inverts whatever follows it: "counter-clockwise" tokenizes to
# "counter" + "clockwise", so the reversal lives in a separate token.
NEGATORS = {"counter", "anti", "reverse", "opposite", "without", "never",
            "not", "avoid", "unplug", "undo"}


def _conflict(a, b):
    """Do these two token lists assert opposite things? -> the pair, or None."""
    sa, sb = set(a), set(b)
    for pair in ANTONYMS:
        x, y = tuple(pair)
        if (x in sa and y in sb) or (y in sa and x in sb):
            return (x, y)
    na, nb = sa & NEGATORS, sb & NEGATORS
    if na != nb:                       # one says "counter-", the other doesn't
        return tuple(sorted(na ^ nb))[:2] or None
    return None


def _pagerank(g, alpha=0.85, iters=80, tol=1e-9):
    """Weighted PageRank by power iteration, in pure Python.

    networkx's own pagerank routes through scipy, which this project does not
    depend on — and the module's promise is "networkx + stdlib". Twenty lines
    here keeps that true and keeps the analytics working on a bare install.
    """
    nodes = list(g.nodes())
    n = len(nodes)
    if not n:
        return {}
    out_w = {x: sum(float(e.get("weight") or 1.0)
                    for _, _, e in g.out_edges(x, data=True)) for x in nodes}
    rank = {x: 1.0 / n for x in nodes}
    for _ in range(iters):
        nxt = {x: (1.0 - alpha) / n for x in nodes}
        dangling = 0.0
        for x in nodes:
            if out_w[x] <= 0:
                dangling += rank[x]
                continue
            for _, v, e in g.out_edges(x, data=True):
                nxt[v] += alpha * rank[x] * float(e.get("weight") or 1.0) / out_w[x]
        if dangling:
            share = alpha * dangling / n
            for x in nodes:
                nxt[x] += share
        if sum(abs(nxt[x] - rank[x]) for x in nodes) < tol:
            rank = nxt
            break
        rank = nxt
    return rank


def _identifiers(tokens):
    """Numerals in a procedure title are identifiers, not adjectives.

    "Replace the pump on a 2019 model" and "…on a 2021 model" are 90% the
    same words and describe different procedures. Merging them would quietly
    tell a technician to do the wrong thing to the wrong machine, so a
    disagreeing numeral vetoes a merge no matter how similar the prose.
    """
    return {t for t in tokens if _NUM.match(t)}


def confidence(support: int, disputes: int) -> float:
    """Laplace-smoothed agreement. One unchallenged source is 0.67, not 1.0 —
    the graph should never sound certain because exactly one video said so.
    """
    return round((support + 1.0) / (support + disputes + 2.0), 3)


# --------------------------------------------------------------------------
# the graph
# --------------------------------------------------------------------------

class Hivemind:
    def __init__(self, path: str | None = None):
        self.g = nx.DiGraph()
        self.path = path
        if path and os.path.exists(path):
            self.load(path)

    # ---------------- ingest ----------------

    def _tasks(self):
        return [(n, d) for n, d in self.g.nodes(data=True)
                if d.get("kind") == "task"]

    def find_task(self, title: str, tags=None):
        """The existing node for this procedure, if the graph already has it.

        Without this the graph fragments: every source phrases the same repair
        differently, so 40 videos about a slipped chain would become 40
        disconnected tasks and nothing would ever accumulate evidence. Merging
        is what turns ingestion into a ratchet instead of a pile.
        """
        want = toks(title + " " + " ".join(tags or []))
        want_ids = _identifiers(want)
        best, score = None, 0.0
        for n, d in self._tasks():
            have = d.get("tokens") or []
            if _identifiers(have) != want_ids:      # different model/version
                continue
            # containment-aware, for the same reason step matching is:
            # "fix a slipped bicycle chain" and "how to fix a dropped bike
            # chain" are one procedure, and plain Jaccard says otherwise
            s = similarity(want, have)
            if s > score and same_thing(want, have, TASK_MERGE):
                best, score = n, s
        return (best, score) if best else (None, score)

    def _steps_of(self, tid):
        out = [(self.g.nodes[v].get("order", 0), v)
               for _, v in self.g.out_edges(tid)
               if self.g.nodes[v].get("kind") == "step"]
        return [v for _, v in sorted(out)]

    def ingest(self, doc: dict) -> str:
        """doc = {title, steps:[str], tools:[str], warnings:[str],
                  tags:[str], source:str}. Returns the task node id.

        Ingesting a procedure the graph already knows does NOT create a second
        copy: matching steps gain a source and a point of support, genuinely
        new steps are appended. Re-mining the same repair from ten videos
        therefore produces one well-evidenced procedure, not ten guesses.
        """
        title = doc["title"].strip()
        src = doc.get("source", "")
        tags = [t.lower() for t in doc.get("tags", [])]
        tid, _sim = self.find_task(title, tags)
        merged = tid is not None
        if not merged:
            tid = "task:" + re.sub(r"\W+", "_", title.lower()).strip("_")

        if merged:
            d = self.g.nodes[tid]
            d["tokens"] = sorted(set(d.get("tokens", []))
                                 | set(toks(title + " " + " ".join(tags))))
            d["tags"] = sorted(set(d.get("tags", [])) | set(tags))
            d["sources"] = sorted(set(d.get("sources", [])
                                      + ([src] if src else [])))
            d["support"] = int(d.get("support", 1)) + 1
        else:
            self.g.add_node(tid, kind="task", title=title, source=src,
                            sources=[src] if src else [], tags=tags,
                            support=1, disputes=0,
                            tokens=toks(title + " " + " ".join(tags)))

        for s in doc.get("steps", []):
            self._add_step(tid, s, src)
        self._reorder(tid)
        for sym in doc.get("symptoms", []):
            self._add_symptom(tid, sym)

        # The identifier veto keeps a 2019 and a 2021 procedure apart, which
        # is correct — but leaving them unlinked loses the fact that they are
        # the same job on different hardware. variant_of restores it without
        # letting their evidence pool.
        if not merged:
            want_ids = _identifiers(toks(title + " " + " ".join(tags)))
            for other, od in self._tasks():
                if other == tid:
                    continue
                if _identifiers(od.get("tokens") or []) == want_ids:
                    continue
                if same_thing([t for t in self.g.nodes[tid]["tokens"]
                               if not _NUM.match(t)],
                              [t for t in (od.get("tokens") or [])
                               if not _NUM.match(t)], TASK_MERGE):
                    self.g.add_edge(tid, other, rel="variant_of")
                    self.g.add_edge(other, tid, rel="variant_of")

        for t in doc.get("tools", []):
            tool_id = "tool:" + re.sub(r"\W+", "_", t.lower()).strip("_")
            if tool_id not in self.g:
                self.g.add_node(tool_id, kind="tool", name=t.strip())
            self.g.add_edge(tid, tool_id, rel="requires")
        for w in doc.get("warnings", []):
            self._add_warning(tid, w, src)
        self._crosslink(tid)
        self._idf = None                      # the corpus changed
        return tid

    def _add_step(self, tid: str, text: str, src: str = ""):
        """Merge a step into a task, or append it. -> (step_id, was_new)."""
        text = str(text).strip()
        if not text:
            return None, False
        tk = toks(text)
        clash = None
        for sid in self._steps_of(tid):
            sd = self.g.nodes[sid]
            if not same_thing(tk, toks(sd["text"]), STEP_MERGE):
                continue
            bad = _conflict(tk, toks(sd["text"]))
            if bad:
                # same instruction, opposite direction — keep both and mark
                # the disagreement rather than silently picking a winner
                clash = (sid, bad)
                break
            sd["support"] = int(sd.get("support", 1)) + 1
            if src and src not in sd.get("sources", []):
                sd["sources"] = sd.get("sources", []) + [src]
            sd["confidence"] = confidence(sd["support"],
                                          int(sd.get("disputes", 0)))
            return sid, False
        n = len(self._steps_of(tid)) + 1
        sid = f"{tid}/step{n}"
        while sid in self.g:                  # merged tasks can collide
            n += 1
            sid = f"{tid}/step{n}"
        self.g.add_node(sid, kind="step", text=text, order=n,
                        support=1, disputes=0, sources=[src] if src else [],
                        confidence=confidence(1, 0))
        self.g.add_edge(tid, sid, rel="has_step")
        if clash:
            other, (x, y) = clash
            why = (f'sources disagree: one says "{x}", the other says "{y}" '
                   f'for what is otherwise the same step')
            self.g.add_edge(sid, other, rel="contradicts", terms=[x, y])
            self.g.add_edge(other, sid, rel="contradicts", terms=[x, y])
            for node in (sid, other):
                self.correct(node, kind="wrong", author="ingest", auto=True,
                             text=why)
        return sid, True

    def _symptoms_of(self, tid):
        return [v for _, v in self.g.out_edges(tid)
                if self.g.nodes[v].get("kind") == "symptom"]

    def _add_symptom(self, tid: str, text: str):
        """Index a task by the PROBLEM it solves, in the words a person uses.

        Procedures are named by their fix ("fix a slipped chain"); people
        describe their symptom ("the chain came off"). Indexing only on the
        title means the query that matters most — someone saying what went
        wrong — matches on nothing more specific than "chain", and ties break
        arbitrarily against any other task mentioning one. A symptom is that
        missing surface, shared across tasks so two procedures can answer the
        same complaint.
        """
        text = re.sub(r"\s+", " ", str(text)).strip().lower()[:160]
        if not text:
            return None
        sid = "symptom:" + re.sub(r"\W+", "_", text).strip("_")[:80]
        if sid not in self.g:
            self.g.add_node(sid, kind="symptom", text=text, tokens=toks(text))
        self.g.add_edge(tid, sid, rel="presents_as")
        self.g.add_edge(sid, tid, rel="resolved_by")
        return sid

    def _add_warning(self, tid: str, text: str, src: str = ""):
        text = str(text).strip()
        if not text:
            return None
        tk = toks(text)
        for _, v in self.g.out_edges(tid):
            nd = self.g.nodes[v]
            if nd.get("kind") == "warning" and same_thing(tk, toks(nd["text"]),
                                                          STEP_MERGE):
                nd["support"] = int(nd.get("support", 1)) + 1
                nd["confidence"] = confidence(nd["support"],
                                              int(nd.get("disputes", 0)))
                return v
        wid = f"{tid}/warn{abs(hash(text)) & 0xffff}"
        self.g.add_node(wid, kind="warning", text=text, support=1, disputes=0,
                        sources=[src] if src else [], confidence=confidence(1, 0))
        self.g.add_edge(tid, wid, rel="warns")
        return wid

    def _reorder(self, tid: str):
        """Rebuild the precedes chain after steps were added or removed."""
        sids = self._steps_of(tid)
        for a, b, e in list(self.g.edges(data=True)):
            if e.get("rel") == "precedes" and a in sids:
                self.g.remove_edge(a, b)
        for i, sid in enumerate(sids, 1):
            self.g.nodes[sid]["order"] = i
            if i > 1:
                self.g.add_edge(sids[i - 2], sid, rel="precedes")

    def _crosslink(self, tid: str):
        """The hivemind part: tasks that share a RARE tool point at each other.

        Weighting by rarity is what stops this degenerating into a hairball —
        every repair on earth uses a rag, so sharing one means nothing, while
        two procedures sharing a torque wrench are genuinely related.
        """
        my_tools = {v for _, v in self.g.out_edges(tid)
                    if self.g.nodes[v].get("kind") == "tool"}
        n_tasks = max(len(self._tasks()), 1)
        for tool in my_tools:
            users = [o for o, _ in self.g.in_edges(tool)
                     if self.g.nodes[o].get("kind") == "task"]
            # a tool more than half the corpus uses carries no information
            if len(users) > max(2, n_tasks // 2):
                continue
            weight = round(1.0 - len(users) / float(n_tasks + 1), 3)
            for other in users:
                if other == tid:
                    continue
                self.g.add_edge(tid, other, rel="shares_tool", via=tool,
                                weight=weight)
                self.g.add_edge(other, tid, rel="shares_tool", via=tool,
                                weight=weight)

    # ---------------- retrieval ----------------

    def _idf_table(self) -> dict:
        """Inverse document frequency over tasks, cached until ingest.

        Raw overlap scoring rewards whichever task happens to repeat a common
        word most often. With IDF, "derailleur" outweighs "the wheel", which
        is the difference between retrieval that works at 20 tasks and
        retrieval that still works at 20,000.
        """
        if getattr(self, "_idf", None) is not None:
            return self._idf
        df, n_docs = Counter(), 0
        for n, d in self._tasks():
            n_docs += 1
            seen = set(d.get("tokens") or [])
            for sid in self._steps_of(n):
                seen |= set(toks(self.g.nodes[sid]["text"]))
            df.update(seen)
        self._idf = {w: math.log(1.0 + n_docs / (1.0 + c))
                     for w, c in df.items()}
        self._idf_default = math.log(1.0 + n_docs)
        return self._idf

    def _w(self, word: str) -> float:
        return self._idf_table().get(word, getattr(self, "_idf_default", 1.0))

    def _score_tasks(self, question: str) -> list[tuple[str, float]]:
        q = Counter(toks(question))
        if not q:
            return []
        scored = []
        for n, d in self._tasks():
            body = Counter(d.get("tokens") or [])
            for sid in self._steps_of(n):
                body.update(toks(self.g.nodes[sid]["text"]))
            overlap = sum(min(q[w], body[w]) * self._w(w) for w in q)
            title_bonus = sum(3 * self._w(w) for w in q
                              if w in (d.get("tokens") or []))
            tag_bonus = sum(2 * self._w(w) for w in q
                            if w in [_stem(t) for t in d.get("tags", [])])
            # A symptom match is the strongest signal there is: the user just
            # told us their problem in the same words a source used to
            # describe it. Weighted above the title so "the chain came off"
            # beats any task that merely mentions a chain.
            sym_bonus = 0.0
            for sy in self._symptoms_of(n):
                st = self.g.nodes[sy].get("tokens") or []
                hit = sum(self._w(w) for w in q if w in st)
                if hit and same_thing(list(q), st, 0.45):
                    hit *= 2.0            # whole complaint matches, not a word
                sym_bonus = max(sym_bonus, hit * 2.5)
            raw = overlap + title_bonus + tag_bonus + sym_bonus
            if raw <= 0:
                continue
            # a well-corroborated procedure outranks a single-source guess
            # at equal relevance, but evidence never manufactures relevance
            ev = confidence(int(d.get("support", 1)), int(d.get("disputes", 0)))
            scored.append((n, round(raw * (0.75 + 0.5 * ev), 4)))
        return sorted(scored, key=lambda x: -x[1])

    def retrieve(self, question: str, k: int = 2) -> str:
        """Best-matching tasks + steps + tools + warnings + related tasks,
        as a compact text block for prompt injection. Empty string if the
        hivemind knows nothing relevant (so callers can say 'not covered')."""
        hits = self._score_tasks(question)[:k]
        if not hits:
            return ""
        out = []
        for tid, score in hits:
            d = self.g.nodes[tid]
            steps, tools, warns, related = [], [], [], set()
            for _, v, e in self.g.out_edges(tid, data=True):
                nd = self.g.nodes[v]
                if nd.get("kind") == "step":
                    steps.append((nd.get("order", 0), v, nd))
                elif nd.get("kind") == "tool":
                    tools.append(nd["name"])
                elif nd.get("kind") == "warning":
                    warns.append(nd)
                elif nd.get("kind") == "task" and e.get("rel") == "shares_tool":
                    related.add(nd["title"])
            steps.sort(key=lambda x: x[0])
            srcs = d.get("sources") or ([d.get("source")] if d.get("source") else [])
            block = [f"## {d['title']}  (sources: {len(srcs) or 1}"
                     f" — {', '.join(srcs[:3]) or 'seed'})"]
            if tools:
                block.append("Tools: " + ", ".join(sorted(set(tools))))
            for i, sid, nd in steps:
                line = f"{i}. {nd['text']}"
                # the tutor must be able to hedge on a step the field
                # disputes, so the evidence travels with the instruction
                sup = int(nd.get("support", 1))
                dis = int(nd.get("disputes", 0))
                if dis:
                    line += (f"   [DISPUTED — {sup} source(s) assert this, "
                             f"{dis} correction(s) say it is wrong; "
                             f"confidence {confidence(sup, dis)}]")
                    # A physics verdict is deterministic and carries the law
                    # it applied, so it is shown while still pending: waiting
                    # for a human to approve "torque = F x r" would leave the
                    # tutor teaching the wrong thing in the meantime.
                    for c in self.corrections_on(sid):
                        if c.get("status") != "accepted" \
                                and c.get("author") != "physics":
                            continue
                        line += f"\n     -> correction: {c['text']}"
                        if c.get("replacement"):
                            line += f"\n        instead: {c['replacement']}"
                elif sup >= 3:
                    line += f"   [confirmed by {sup} sources]"
                block.append(line)
            for w in warns:
                block.append(f"! Warning: {w['text']}")
            if related:
                block.append("Related tasks in the hivemind: "
                             + "; ".join(sorted(related)[:5]))
            out.append("\n".join(block))
        return "\n\n".join(out)

    def retrieve_ids(self, question: str, k: int = 2) -> list:
        """Same retrieval, but addressable.

        retrieve() returns prose for a prompt; this returns the node ids
        behind it, which is what lets a student's "that's wrong" attach to
        the exact step instead of to the lesson in general.
        """
        out = []
        for tid, score in self._score_tasks(question)[:k]:
            d = self.g.nodes[tid]
            steps = []
            for sid in self._steps_of(tid):
                sd = self.g.nodes[sid]
                steps.append({"id": sid, "text": sd["text"],
                              "order": sd.get("order", 0),
                              "support": int(sd.get("support", 1)),
                              "disputes": int(sd.get("disputes", 0)),
                              "confidence": self.confidence_of(sid)})
            out.append({"task": tid, "title": d["title"],
                        "score": score, "steps": steps,
                        "sources": d.get("sources", []),
                        "confidence": self.confidence_of(tid)})
        return out

    # ---------------- reasoning ----------------
    #
    # Retrieval answers "which procedure best matches these words". None of
    # the methods below could be built on that alone: they are the difference
    # between a lookup table and something that can be asked a question.

    def _profile(self, tid):
        """Every token that characterises a task — title, steps, symptoms."""
        d = self.g.nodes[tid]
        bag = set(d.get("tokens") or [])
        for sid in self._steps_of(tid):
            bag |= set(toks(self.g.nodes[sid]["text"]))
        for sy in self._symptoms_of(tid):
            bag |= set(self.g.nodes[sy].get("tokens") or [])
        return bag

    def discriminate(self, candidates):
        """What single question separates these candidate procedures?

        A symptom rarely identifies one repair — "the chain came off" and
        "the chain is jammed" share almost every word a person would use.
        Returning the best match and hoping is how a lookup tool behaves; a
        tutor asks the one question that halves the space. The discriminator
        is the highest-IDF token each candidate has that the others lack.
        """
        if len(candidates) < 2:
            return None
        profiles = {t: self._profile(t) for t in candidates}
        opts = []
        for t, own in profiles.items():
            others = set().union(*(p for k, p in profiles.items() if k != t))
            unique = own - others
            if not unique:
                continue
            best = max(unique, key=self._w)
            opts.append({"task": t, "title": self.g.nodes[t]["title"],
                         "term": best, "weight": round(self._w(best), 3)})
        if len(opts) < 2:
            return None
        opts.sort(key=lambda o: -o["weight"])
        opts = opts[:3]
        return {"question": "Which is closer to what you are seeing — "
                            + ", or ".join(f'"{o["term"]}"' for o in opts)
                            + "?",
                "options": opts}

    def diagnose(self, question, k: int = 3):
        """Symptom in -> ranked candidates plus the question that separates
        them. This is the retrieval a tutor needs, not the one a search box
        needs."""
        hits = self._score_tasks(question)[:k]
        if not hits:
            return {"candidates": [], "discriminator": None, "confident": False}
        top = hits[0][1]
        cands = [{"task": t, "title": self.g.nodes[t]["title"],
                  "score": round(s, 3),
                  "margin": round(s / top, 3) if top else 0,
                  "confidence": self.confidence_of(t)} for t, s in hits]
        # a clear winner needs no question; a near-tie is exactly when asking
        # one is worth more than guessing
        close = [c["task"] for c in cands if c["margin"] >= 0.6]
        return {"candidates": cands,
                "discriminator": self.discriminate(close) if len(close) > 1
                else None,
                "confident": len(close) == 1}

    def prerequisites(self, tid, depth: int = 2):
        """Walk requires/shares_tool outward — the edges retrieval ignores.

        The graph has always built these; nothing ever read them, so
        "what do I need before I start?" was unanswerable despite the answer
        being two hops away.
        """
        seen, tools, related = {tid}, [], []
        frontier, d = [tid], 0
        while frontier and d < depth:
            nxt = []
            for n in frontier:
                for _, v, e in self.g.out_edges(n, data=True):
                    nd = self.g.nodes[v]
                    if nd.get("kind") == "tool" and v not in seen:
                        seen.add(v)
                        tools.append({"id": v, "name": nd["name"],
                                      "via": self.g.nodes[n].get("title", n),
                                      "hops": d + 1})
                        nxt.append(v)
                    elif nd.get("kind") == "task" and v not in seen \
                            and e.get("rel") in ("shares_tool", "variant_of"):
                        seen.add(v)
                        related.append({"id": v, "title": nd["title"],
                                        "rel": e.get("rel"),
                                        "weight": e.get("weight"),
                                        "hops": d + 1})
                        nxt.append(v)
            frontier, d = nxt, d + 1
        return {"tools": tools, "related": related}

    def find_steps(self, question, k: int = 5):
        """Step-level retrieval. Retrieval has only ever worked at task level,
        so a question about one action pulled in a whole procedure."""
        q = Counter(toks(question))
        if not q:
            return []
        out = []
        for n, d in self.g.nodes(data=True):
            if d.get("kind") != "step":
                continue
            st = toks(d["text"])
            score = sum(min(q[w], st.count(w)) * self._w(w) for w in q)
            if score <= 0:
                continue
            owner = next((u for u, _ in self.g.in_edges(n)
                          if self.g.nodes[u].get("kind") == "task"), None)
            out.append({"id": n, "text": d["text"],
                        "score": round(score, 3),
                        "confidence": self.confidence_of(n),
                        "task": self.g.nodes[owner]["title"] if owner else ""})
        return sorted(out, key=lambda x: -x["score"])[:k]

    def central_tasks(self, k: int = 10):
        """PageRank over the task graph — which procedures are load-bearing.

        At scale this is how you find the canonical procedure among near
        duplicates, which is the one worth curating and the right merge
        target."""
        sub = self.g.subgraph([n for n, d in self.g.nodes(data=True)
                               if d.get("kind") == "task"])
        if sub.number_of_nodes() < 2:
            return []
        pr = _pagerank(sub)
        return [{"task": t, "title": self.g.nodes[t]["title"],
                 "rank": round(v, 4)}
                for t, v in sorted(pr.items(), key=lambda x: -x[1])[:k]]

    def communities(self):
        """Which clusters the corpus actually contains — an empirical answer
        to 'what vertical do I have', rather than an assumed one."""
        sub = self.g.subgraph([n for n, d in self.g.nodes(data=True)
                               if d.get("kind") == "task"]).to_undirected()
        if sub.number_of_nodes() < 2:
            return []
        try:
            from networkx.algorithms.community import louvain_communities
            groups = louvain_communities(sub, seed=1)
        except Exception:
            try:
                from networkx.algorithms.community import \
                    greedy_modularity_communities
                groups = greedy_modularity_communities(sub)
            except Exception:
                return []
        return [sorted(self.g.nodes[t]["title"] for t in grp)
                for grp in sorted(groups, key=len, reverse=True)]

    # ---------------- corrections ----------------
    #
    # The compounding half of the graph. Ingestion gives you what the world
    # already published, which any competitor can also scrape. Corrections
    # are generated by the people actually doing the work, on the equipment
    # they actually touch, and exist nowhere else. Every one of them moves a
    # step's confidence, so the graph gets more truthful with use rather than
    # merely larger.

    CORRECTION_KINDS = ("wrong", "missing", "clarify", "confirm")

    def correct(self, target: str, kind: str = "wrong", text: str = "",
                author: str = "anon", replacement: str = "",
                auto: bool = False) -> str:
        """Record that someone disagreed with, or confirmed, a node.

        target      a step / warning / task id already in the graph
        kind        wrong | missing | clarify | confirm
        replacement for "wrong" or "missing": what it should say instead

        Nothing is silently rewritten. A correction is evidence; applying it
        is a separate, reviewable decision (see apply_correction), because a
        graph that any passer-by can edit in place is not an asset.
        """
        if target not in self.g:
            raise KeyError(f"no such node: {target}")
        if kind not in self.CORRECTION_KINDS:
            raise ValueError(f"kind must be one of {self.CORRECTION_KINDS}")
        td = self.g.nodes[target]

        cid = f"correction:{abs(hash((target, kind, text, replacement, author))) & 0xffffffff}"
        self.g.add_node(cid, kind="correction", ckind=kind,
                        text=str(text)[:500], replacement=str(replacement)[:500],
                        author=str(author)[:80], at=time.time(),
                        auto=bool(auto),
                        status="accepted" if kind == "confirm" else "pending")

        if kind == "confirm":
            self.g.add_edge(cid, target, rel="confirms")
            td["support"] = int(td.get("support", 1)) + 1
        else:
            self.g.add_edge(cid, target, rel="disputes" if kind == "wrong"
                            else "corrects")
            if kind == "wrong":
                td["disputes"] = int(td.get("disputes", 0)) + 1
        td["confidence"] = confidence(int(td.get("support", 1)),
                                      int(td.get("disputes", 0)))
        return cid

    def corrections_on(self, target: str, status: str | None = None):
        """Every correction pointing at a node, newest first."""
        out = []
        for c, _ in self.g.in_edges(target):
            d = self.g.nodes[c]
            if d.get("kind") != "correction":
                continue
            if status and d.get("status") != status:
                continue
            out.append({"id": c, **{k: v for k, v in d.items() if k != "kind"}})
        return sorted(out, key=lambda x: -x.get("at", 0))

    def pending_corrections(self, limit: int = 50):
        """The review queue — what a human or a second model should rule on."""
        out = []
        for n, d in self.g.nodes(data=True):
            if d.get("kind") != "correction" or d.get("status") != "pending":
                continue
            tgt = next((v for _, v in self.g.out_edges(n)), None)
            out.append({"id": n, "target": tgt,
                        "target_text": self.g.nodes[tgt].get("text", "")
                        if tgt in self.g else "",
                        "ckind": d.get("ckind"), "text": d.get("text"),
                        "replacement": d.get("replacement"),
                        "author": d.get("author"), "auto": d.get("auto", False),
                        "at": d.get("at", 0)})
        return sorted(out, key=lambda x: -x["at"])[:limit]

    def apply_correction(self, cid: str, accept: bool = True) -> dict:
        """Rule on a pending correction. -> what changed.

        Accepting a "wrong" with a replacement rewrites the step and resets
        its evidence: the old support counted votes for text that is no
        longer there, so carrying it over would launder a corrected claim
        into a well-supported one.
        """
        if cid not in self.g or self.g.nodes[cid].get("kind") != "correction":
            raise KeyError(f"no such correction: {cid}")
        cd = self.g.nodes[cid]
        tgt = next((v for _, v in self.g.out_edges(cid)), None)
        if not accept:
            cd["status"] = "rejected"
            if cd.get("ckind") == "wrong" and tgt in self.g:
                td = self.g.nodes[tgt]
                td["disputes"] = max(0, int(td.get("disputes", 0)) - 1)
                td["confidence"] = confidence(int(td.get("support", 1)),
                                              td["disputes"])
            return {"status": "rejected", "target": tgt}

        cd["status"] = "accepted"
        if tgt not in self.g:
            return {"status": "accepted", "target": None}
        td = self.g.nodes[tgt]
        changed = {"status": "accepted", "target": tgt, "rewrote": False}

        if cd.get("ckind") == "wrong" and cd.get("replacement"):
            td["text"] = cd["replacement"]
            td["support"], td["disputes"] = 1, 0
            td["confidence"] = confidence(1, 0)
            td["sources"] = sorted(set(td.get("sources", [])
                                       + [f"correction:{cd.get('author')}"]))
            changed["rewrote"] = True
        elif cd.get("ckind") == "missing" and cd.get("replacement"):
            owner = tgt if td.get("kind") == "task" else \
                next((u for u, _ in self.g.in_edges(tgt)
                      if self.g.nodes[u].get("kind") == "task"), None)
            if owner:
                sid, _new = self._add_step(owner, cd["replacement"],
                                           f"correction:{cd.get('author')}")
                self._reorder(owner)
                changed["added_step"] = sid
        self._idf = None
        return changed

    def confidence_of(self, node: str) -> float:
        d = self.g.nodes[node]
        return confidence(int(d.get("support", 1)), int(d.get("disputes", 0)))

    def disputed(self, limit: int = 50):
        """Everything the field disagrees with, worst first — the work queue
        for whoever is responsible for the graph being right."""
        out = []
        for n, d in self.g.nodes(data=True):
            if d.get("kind") not in ("step", "warning", "task"):
                continue
            if not int(d.get("disputes", 0)):
                continue
            out.append({"id": n, "kind": d.get("kind"),
                        "text": d.get("text") or d.get("title", ""),
                        "support": int(d.get("support", 1)),
                        "disputes": int(d.get("disputes", 0)),
                        "confidence": self.confidence_of(n)})
        return sorted(out, key=lambda x: x["confidence"])[:limit]

    # ---------------- stats / persistence ----------------

    def stats(self) -> dict:
        kinds = Counter(d.get("kind") for _, d in self.g.nodes(data=True))
        rels = Counter(e.get("rel") for _, _, e in self.g.edges(data=True))
        steps = [d for _, d in self.g.nodes(data=True)
                 if d.get("kind") == "step"]
        conf = [confidence(int(d.get("support", 1)), int(d.get("disputes", 0)))
                for d in steps]
        pending = sum(1 for _, d in self.g.nodes(data=True)
                      if d.get("kind") == "correction"
                      and d.get("status") == "pending")
        return {"nodes": dict(kinds), "edges": dict(rels),
                "crosslinks": rels.get("shares_tool", 0) // 2,
                # the numbers that say whether the graph is compounding or
                # merely growing: corroboration, dispute, and review backlog
                "avg_confidence": round(sum(conf) / len(conf), 3) if conf else 0,
                "corroborated": sum(1 for d in steps
                                    if int(d.get("support", 1)) >= 2),
                "disputed": sum(1 for d in steps
                                if int(d.get("disputes", 0)) > 0),
                "pending_corrections": pending}

    def save(self, path: str | None = None):
        path = path or self.path
        with open(path, "w") as f:
            json.dump(json_graph.node_link_data(self.g, edges="links"), f)

    def load(self, path: str):
        with open(path) as f:
            self.g = json_graph.node_link_graph(json.load(f), edges="links")
        self._idf = None          # a loaded corpus needs its own IDF table


# --------------------------------------------------------------------------
# extraction: raw text -> structured doc
# --------------------------------------------------------------------------

def backfill_symptoms(hm: "Hivemind") -> int:
    """Add seed symptoms to a graph persisted before symptoms existed.

    Seeding only runs on an empty graph, so an existing hivemind_graph.json
    keeps whatever schema it was written with — the symptom index would stay
    empty forever on any deployment that has already saved a graph. Adding
    only symptom nodes (never support) makes this safe to run on every boot.
    """
    added = 0
    for doc in SEED_DOCS:
        if not doc.get("symptoms"):
            continue
        tid, _ = hm.find_task(doc["title"], doc.get("tags"))
        if not tid:
            continue
        have = {hm.g.nodes[s]["text"] for s in hm._symptoms_of(tid)}
        for sym in doc["symptoms"]:
            if str(sym).strip().lower() not in have:
                hm._add_symptom(tid, sym)
                added += 1
    if added:
        hm._idf = None
    return added


def parse_plain_howto(title: str, text: str, source: str = "") -> dict:
    """Heuristic parser for wikiHow-style plain text: numbered/bulleted
    lines become steps; 'you will need' lines become tools. Offline."""
    steps, tools, warnings = [], [], []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        low = ln.lower()
        m = re.match(r"^(?:step\s*)?\d+[.)]\s*(.+)|^[-*•]\s*(.+)", ln, re.I)
        if low.startswith(("you will need", "you'll need", "materials:", "tools:")):
            tools += [t.strip() for t in
                      re.split(r"[,;]", ln.split(":", 1)[-1]) if t.strip()]
        elif low.startswith(("warning", "caution", "be careful")):
            warnings.append(re.sub(r"^(warning|caution)\s*[:.]?\s*", "", ln, flags=re.I))
        elif m:
            steps.append((m.group(1) or m.group(2)).strip())
    return {"title": title, "steps": steps, "tools": tools,
            "warnings": warnings, "tags": [], "source": source}


EXTRACT_SYS = """You turn how-to articles or video transcripts into JSON.
Respond with ONLY this JSON, no fences:
{"title":"...","steps":["..."],"tools":["..."],"warnings":["..."],"tags":["..."]}
Steps are short imperative sentences in order. Tools are physical items.
Tags are 3-6 topic words."""


def extract_with_llm(text: str, source: str = "", model="claude-haiku-4-5-20251001") -> dict:
    """Raw article/transcript -> structured doc via Claude. Needs
    ANTHROPIC_API_KEY. Haiku by default: extraction is cheap work."""
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model, max_tokens=1500, system=EXTRACT_SYS,
        messages=[{"role": "user", "content": text[:12000]}])
    raw = "".join(b.text for b in resp.content if b.type == "text")
    doc = json.loads(re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.M))
    doc["source"] = source
    return doc


def answer_with_llm(hm: Hivemind, question: str, model="claude-sonnet-4-6") -> str:
    """Compose a grounded answer from retrieved context. Refuses politely
    when the hivemind has nothing (no hallucinated how-tos)."""
    ctx = hm.retrieve(question, k=2)
    if not ctx:
        return ("The hivemind has no article covering that yet. "
                "Crawl a source for it and ask again.")
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model, max_tokens=800,
        system=("Answer using ONLY the how-to knowledge provided. Cite the "
                "source titles. If the knowledge does not cover the question, "
                "say so plainly."),
        messages=[{"role": "user",
                   "content": f"Knowledge:\n{ctx}\n\nQuestion: {question}"}])
    return "".join(b.text for b in resp.content if b.type == "text")


# --------------------------------------------------------------------------
# seed how-tos so everything is testable offline
# --------------------------------------------------------------------------

SEED_DOCS = [
    {"title": "Fix a slipped bicycle chain",
     "steps": ["Shift to the smallest rear cog and stop pedaling",
               "Push the rear derailleur forward to slacken the chain",
               "Seat the chain onto the bottom of the chainring teeth",
               "Rotate the pedals slowly forward until the chain seats fully",
               "Wipe your hands and test-ride at low speed"],
     "tools": ["degreaser", "rag", "gloves"],
     "warnings": ["Keep fingers clear of the chainring while rotating pedals"],
     "tags": ["bicycle", "chain", "repair"],
     "symptoms": ["the chain came off", "chain fell off the bike",
                  "chain dropped off the chainring",
                  "chain is dangling loose", "drivetrain detached"],
     "source": "seed:wikihow-style"},
    {"title": "Lubricate a bicycle chain",
     "steps": ["Clean the chain with degreaser and a rag",
               "Apply one drop of chain lube per roller while backpedaling",
               "Let the lube sit for a few minutes",
               "Wipe off all excess lube with a clean rag"],
     "tools": ["degreaser", "rag", "chain lube"],
     "warnings": [],
     "tags": ["bicycle", "chain", "maintenance"],
     "symptoms": ["the chain is squeaking", "chain sounds dry and noisy",
                  "chain is rusty", "gears grind when pedaling"],
     "source": "seed:wikihow-style"},
    {"title": "Patch a bicycle inner tube",
     "steps": ["Remove the wheel and pry the tire off with tire levers",
               "Pull out the tube and inflate it slightly to find the leak",
               "Rough the area around the hole with sandpaper",
               "Apply vulcanizing glue and press the patch on firmly",
               "Refit the tube and tire, then inflate to pressure"],
     "tools": ["tire levers", "patch kit", "pump", "sandpaper"],
     "warnings": ["Check the tire inside for the thorn or glass that caused it"],
     "tags": ["bicycle", "tire", "puncture", "repair"],
     "symptoms": ["the tire is flat", "tyre keeps going down",
                  "wheel lost all its air", "puncture in the inner tube"],
     "source": "seed:wikihow-style"},
    {"title": "Revive an overwatered houseplant",
     "steps": ["Take the plant out of its pot and inspect the roots",
               "Trim off brown mushy roots with clean scissors",
               "Repot in fresh dry soil with drainage holes",
               "Water lightly and keep out of direct sun for a week"],
     "tools": ["scissors", "fresh potting soil", "pot with drainage"],
     "warnings": ["Sterilize scissors so root rot does not spread"],
     "tags": ["houseplant", "overwatering", "roots"],
     "source": "seed:wikihow-style"},
]


if __name__ == "__main__":
    hm = Hivemind()
    for d in SEED_DOCS:
        hm.ingest(d)
    print("stats:", hm.stats())
    print()
    print(hm.retrieve("my bike chain came off while riding"))
