"""openclaw.py — mine a how-to video into a lesson the rigs can perform.

The hivemind can already be grown from articles. This closes the other half:
point it at a YouTube instructional video and it comes back as a lesson made
of *motion*, not prose — every step classified as something the fingers do,
something the whole body does, or both, and phrased so the hand rig
(handform.py) and the skeleton rig (motion.py) can actually perform it.

    transcript  ->  Claude extraction  ->  steps + modality + rig phrasing
                ->  hivemind.ingest (so it is retrievable forever)
                ->  chapters index.html can play, using compose for "both"

Transcripts come from youtube_transcript_api when it is installed and from
the watch page's own caption tracks when it is not, so the miner works on a
bare `pip install -r requirements.txt`.

    python openclaw.py "https://www.youtube.com/watch?v=..."      # one video
    python openclaw.py urls.txt                                   # a crawl
"""

import json
import os
import re
import sys

import requests

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
EXTRACT_MODEL = "claude-haiku-4-5"            # extraction is cheap work

# The rigs recognise these verbs. Steering the model toward them is the
# difference between a clip that demonstrates the step and a clip that
# waves vaguely, so they are pasted into the prompt rather than hoped for.
BODY_VERBS = ("grasp", "lift", "push", "pull", "twist", "crank", "pour",
              "wipe", "tap", "point", "cut", "inspect", "explain")
HAND_POSES = ("pinch", "grip", "flat", "fist", "point", "tripod", "spread",
              "open palm", "thumbs up", "ok", "peace", "gun")


class MineError(Exception):
    """Anything that stops a video becoming a lesson, phrased for a user."""


# ------------------------------------------------------------- transcript

def video_id(url):
    """Accept a full watch/share/embed URL or a bare id."""
    url = str(url).strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url
    m = re.search(r"(?:v=|/embed/|/shorts/|youtu\.be/|/v/)([A-Za-z0-9_-]{11})", url)
    if not m:
        raise MineError(f"no YouTube video id in {url!r}")
    return m.group(1)


def _via_library(vid):
    """Preferred path: youtube_transcript_api, when it is installed."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        return None
    try:                                     # 1.x instance API, then 0.6.x
        api = YouTubeTranscriptApi()
        rows = api.fetch(vid, languages=["en", "en-US", "en-GB"]).to_raw_data()
    except AttributeError:
        rows = YouTubeTranscriptApi.get_transcript(
            vid, languages=["en", "en-US", "en-GB"])
    except Exception:
        return None
    return [(float(r["start"]), str(r["text"]).replace("\n", " ").strip())
            for r in rows if str(r.get("text", "")).strip()]


def _via_watch_page(vid):
    """Fallback: read the caption track the watch page itself advertises."""
    r = requests.get(f"https://www.youtube.com/watch?v={vid}&hl=en",
                     headers={"User-Agent": UA, "Accept-Language": "en-US,en"},
                     cookies={"CONSENT": "YES+1"}, timeout=30)
    if r.status_code >= 400:
        raise MineError(f"YouTube returned HTTP {r.status_code} for {vid}")
    m = re.search(r'"captionTracks":(\[.*?\])', r.text)
    if not m:
        raise MineError("this video has no caption track the miner can read "
                        "(auto-captions off, age-gated, or region blocked)")
    try:
        tracks = json.loads(m.group(1))
    except ValueError:
        raise MineError("could not parse the caption track list")
    track = next((t for t in tracks
                  if str(t.get("languageCode", "")).startswith("en")), tracks[0])
    base = track.get("baseUrl")
    if not base:
        raise MineError("caption track has no URL")
    c = requests.get(base + "&fmt=json3", headers={"User-Agent": UA}, timeout=30)
    try:
        events = c.json().get("events") or []
    except ValueError:
        raise MineError("caption download was not JSON")
    out = []
    for e in events:
        txt = "".join(s.get("utf8", "") for s in (e.get("segs") or []))
        txt = txt.replace("\n", " ").strip()
        if txt:
            out.append((e.get("tStartMs", 0) / 1000.0, txt))
    return out


def _page_title(vid):
    try:
        r = requests.get("https://www.youtube.com/oembed?format=json&url="
                         f"https://www.youtube.com/watch?v={vid}",
                         headers={"User-Agent": UA}, timeout=20)
        if r.ok:
            return str(r.json().get("title", ""))[:160]
    except (requests.RequestException, ValueError):
        pass
    return ""


def fetch_transcript(url):
    """-> (video_id, title, [(start_seconds, text), ...]). Raises MineError."""
    vid = video_id(url)
    rows = _via_library(vid) or _via_watch_page(vid)
    if not rows:
        raise MineError("the transcript came back empty")
    return vid, _page_title(vid), rows


def search(query, limit=8):
    """YouTube search -> [video ids], best match first.

    Scraped from the results page rather than the Data API, so mining by
    topic needs no second API key. Order is YouTube's own relevance order.
    """
    try:
        r = requests.get("https://www.youtube.com/results",
                         params={"search_query": query, "hl": "en"},
                         headers={"User-Agent": UA,
                                  "Accept-Language": "en-US,en"},
                         cookies={"CONSENT": "YES+1"}, timeout=30)
    except requests.RequestException as e:
        raise MineError(f"YouTube search unreachable: {e}")
    if r.status_code >= 400:
        raise MineError(f"YouTube search returned HTTP {r.status_code}")
    seen, out = set(), []
    for m in re.finditer(r'"videoId":"([A-Za-z0-9_-]{11})"', r.text):
        v = m.group(1)
        if v not in seen:
            seen.add(v)
            out.append(v)
        if len(out) >= limit:
            break
    if not out:
        raise MineError(f"no videos found for {query!r}")
    return out


def search_with_transcripts(query, want=1, tries=8):
    """Search, then keep only the hits that actually have a readable
    transcript — the first result is regularly one that does not."""
    good = []
    for vid in search(query, tries):
        try:
            _, title, rows = fetch_transcript(vid)
        except MineError:
            continue
        good.append({"id": vid, "title": title, "rows": len(rows)})
        if len(good) >= want:
            break
    if not good:
        raise MineError(f"no video for {query!r} had a usable transcript")
    return good


# ------------------------------------------------------- articles (wikiHow)

_TAG = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)
_STRIP = re.compile(r"<[^>]+>")


def _html_text(html):
    """Crude but dependency-free: drop scripts, tags and entities."""
    txt = _TAG.sub(" ", html)
    # wikiHow marks each step headline with <b class="whb">; keeping a line
    # break there is what turns the page back into a numbered list.
    txt = re.sub(r"<(b[^>]*class=\"[^\"]*whb|/?(p|div|li|h[1-6]|br))[^>]*>",
                 "\n", txt, flags=re.I)
    txt = _STRIP.sub(" ", txt)
    for a, b in (("&nbsp;", " "), ("&amp;", "&"), ("&quot;", '"'),
                 ("&#39;", "'"), ("&lt;", "<"), ("&gt;", ">")):
        txt = txt.replace(a, b)
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in txt.splitlines()]
    return "\n".join(ln for ln in lines if len(ln) > 2)


def fetch_article(url):
    """Any how-to page -> (title, plain text). wikiHow is the shape this was
    written against, but nothing here is wikiHow-specific."""
    try:
        r = requests.get(url, headers={"User-Agent": UA,
                                       "Accept-Language": "en-US,en"},
                         timeout=30)
    except requests.RequestException as e:
        raise MineError(f"could not fetch that page: {e}")
    if r.status_code >= 400:
        raise MineError(f"that page returned HTTP {r.status_code}")
    m = re.search(r"<title[^>]*>(.*?)</title>", r.text, re.S | re.I)
    title = _STRIP.sub("", m.group(1)).strip()[:160] if m else ""
    title = re.sub(r"\s*[-|]\s*wikiHow\s*$", "", title, flags=re.I)
    text = _html_text(r.text)
    if len(text) < 200:
        raise MineError("that page had almost no readable text")
    return title, text


def transcript_text(rows, max_chars=14000):
    """Timestamped plain text — the timestamps let the model keep the order
    and give each chapter an honest 'from 4:12 in the video' provenance."""
    out, n = [], 0
    for start, txt in rows:
        line = f"[{int(start)//60}:{int(start) % 60:02d}] {txt}"
        if n + len(line) > max_chars:
            break
        out.append(line)
        n += len(line) + 1
    return "\n".join(out)


# ----------------------------------------------------------- extraction

MINE_SYS = f"""You convert a how-to VIDEO TRANSCRIPT into a lesson that will be
performed by two physical rigs: a tracked 3D HAND (21 landmarks, five finger
chains) and a whole-body SKELETON (servo-chain limbs, gesture tracking).

Respond with ONLY this JSON, no markdown fences:
{{"title":"short how-to title, imperative",
  "summary":"one sentence on what the video teaches",
  "tools":["physical items used"],
  "warnings":["safety cautions actually stated"],
  "tags":["3-6 topic words"],
  "symptoms":["how someone with this problem would DESCRIBE it in their own words, before they know the fix — 2-5 short phrases, e.g. 'the chain came off', 'it squeaks when I pedal'. Not the name of the repair."],
  "steps":[
    {{"text":"the step as a full instruction, imperative",
      "narration":"1-2 conversational sentences a tutor says out loud while this is demonstrated",
      "modality":"hand" | "body" | "both",
      "hand_actions":["2-4 word imperative starting with a verb"],
      "body_actions":["2-4 word imperative starting with a verb"],
      "at":"m:ss timestamp this step starts in the video"}}]}}

Rules that matter:
- 3 to 6 steps. Real steps from the transcript only — never invent a step the
  speaker did not describe, and drop intros, sponsor reads and sign-offs.
- "modality" is the honest answer to "what does the student need to SEE?":
  "hand"  the fingers do the work (grip, pinch, twist a small part)
  "body"  posture, reach, lifting, whole-arm force
  "both"  the body positions while the fingers act — use this whenever the
          step has a bracing/holding component AND a fingers component.
- Fill hand_actions when modality is hand or both; body_actions when it is
  body or both. Both lists for "both". 1-2 entries each, never more.
- Phrase actions so a rig can perform them: START WITH A VERB and prefer
  these, because they are the motions the rigs actually know —
  body: {", ".join(BODY_VERBS)}
  hand: {", ".join(HAND_POSES)}
  Good: "Pinch the clip and pull", "Brace the panel with your knee".
  Bad: "Be careful here", "The next part is important".
- narration is spoken aloud, so no timestamps or "as you can see in the video".
"""


def _claude(api_key, system, user, model=EXTRACT_MODEL, max_tokens=3000):
    r = requests.post(ANTHROPIC_URL,
                      headers={"Content-Type": "application/json",
                               "x-api-key": api_key,
                               "anthropic-version": "2023-06-01"},
                      json={"model": model, "max_tokens": max_tokens,
                            "system": system,
                            "messages": [{"role": "user", "content": user}]},
                      timeout=120)
    if r.status_code >= 400:
        try:
            msg = r.json().get("error", {}).get("message", r.text[:200])
        except ValueError:
            msg = r.text[:200]
        raise MineError(f"extraction call failed: {msg}")
    body = r.json()
    return "".join(b.get("text", "") for b in body.get("content", [])
                   if b.get("type") == "text")


def _loads(raw):
    """Models occasionally fence their JSON despite being told not to."""
    txt = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    try:
        return json.loads(txt)
    except ValueError:
        m = re.search(r"\{.*\}", txt, re.S)
        if not m:
            raise MineError("the extractor did not return JSON")
        return json.loads(m.group(0))


def _clean_actions(vals, limit=2):
    out = []
    for v in vals or []:
        s = re.sub(r"\s+", " ", str(v)).strip(" .")
        if s:
            out.append(s[:90])
    return out[:limit]


def _extract(api_key, header, body, model, system=None):
    """Shared tail of every miner: text in, validated step list out.

    `system` overrides the extraction prompt — pairing.py passes a second,
    deliberately different reader prompt here so the two passes can disagree.
    """
    if not api_key:
        raise MineError("server has no ANTHROPIC_API_KEY set")
    doc = _loads(_claude(api_key, system or MINE_SYS,
                         f"{header}\n\n{body}", model))
    steps = []
    for s in (doc.get("steps") or [])[:6]:
        if not isinstance(s, dict):
            s = {"text": str(s)}
        mod = s.get("modality") if s.get("modality") in ("hand", "body", "both") \
            else "hand"
        hand = _clean_actions(s.get("hand_actions"))
        bodyacts = _clean_actions(s.get("body_actions"))
        # a modality with no actions to perform is a lie — demote it
        if mod == "both" and not (hand and bodyacts):
            mod = "hand" if hand else "body" if bodyacts else "hand"
        if mod == "hand" and not hand:
            hand = _clean_actions([s.get("text")]) or ["Grip it firmly"]
        if mod == "body" and not bodyacts:
            bodyacts = _clean_actions([s.get("text")]) or ["Reach for it"]
        steps.append({"text": re.sub(r"\s+", " ", str(s.get("text", ""))).strip()[:200],
                      "narration": re.sub(r"\s+", " ", str(s.get("narration", ""))).strip()[:400],
                      "modality": mod, "hand_actions": hand,
                      "body_actions": bodyacts, "at": str(s.get("at", ""))[:8]})
    steps = [s for s in steps if s["text"]]
    if not steps:
        raise MineError("no usable steps found in that source — it may be a "
                        "talking-head video or an article that never gets "
                        "around to the actual procedure")
    doc["steps"] = steps
    return doc


def _finish(doc, title, extra):
    """Common doc envelope, whatever the source was."""
    out = {"title": str(doc.get("title") or title or "mined how-to")[:120],
           "summary": str(doc.get("summary", ""))[:300],
           "steps": doc["steps"],
           "tools": [str(t)[:60] for t in (doc.get("tools") or [])][:12],
           "warnings": [str(w)[:160] for w in (doc.get("warnings") or [])][:6],
           "tags": [str(t)[:30] for t in (doc.get("tags") or [])][:8],
           "symptoms": [str(s)[:120] for s in (doc.get("symptoms") or [])][:6]}
    out.update(extra)
    return out


def mine(url, api_key, model=EXTRACT_MODEL):
    """Video URL -> structured how-to doc with per-step modality.

    Raises MineError with something a user can act on.
    """
    vid, title, rows = fetch_transcript(url)
    doc = _extract(api_key,
                   f"Video title: {title or '(unknown)'}",
                   f"Transcript:\n{transcript_text(rows)}", model)
    return _finish(doc, title, {
        "video_id": vid, "video_title": title,
        "url": f"https://www.youtube.com/watch?v={vid}",
        "source": f"youtube:{vid}"})


def mine_article(url, api_key, model=EXTRACT_MODEL):
    """A how-to PAGE (wikiHow and friends) -> the same doc shape.

    Steps have no timestamps, so a mined article can never be pixel-tracked
    — it becomes synthesised hand/body/compose chapters, which is the whole
    point: an article has no footage to track in the first place.
    """
    title, text = fetch_article(url)
    doc = _extract(api_key,
                   f"Article title: {title or '(unknown)'}\n"
                   f"This is a written how-to article, not a video "
                   f"transcript. Omit the \"at\" field entirely.",
                   f"Article:\n{text[:14000]}", model)
    for s in doc["steps"]:
        s["at"] = ""                      # nothing to seek to in an article
    return _finish(doc, title, {"url": url, "source": url})


# ------------------------------------------------- doc -> playable lesson

def hivemind_doc(doc):
    """The flat shape hivemind.ingest expects — steps as plain sentences."""
    return {"title": doc["title"], "steps": [s["text"] for s in doc["steps"]],
            "tools": doc.get("tools", []), "warnings": doc.get("warnings", []),
            "tags": doc.get("tags", []), "symptoms": doc.get("symptoms", []),
            "source": doc.get("source", "youtube")}


def _seconds_at(at):
    """"4:12" -> 252.0. Returns None for anything that is not a timestamp."""
    m = re.fullmatch(r"\s*(?:(\d+):)?(\d{1,2}):(\d{2})\s*", str(at or ""))
    if m:
        h, mi, s = m.group(1) or 0, m.group(2), m.group(3)
        return int(h) * 3600 + int(mi) * 60 + int(s)
    m = re.fullmatch(r"\s*(\d{1,4})\s*", str(at or ""))
    return float(m.group(1)) if m else None


def doc_to_lesson(doc, track=False):
    """Steps -> chapters index.html can play.

    "both" becomes a compose chapter, so the posture and the grip are on
    screen together instead of the student having to pick one.

    With track=True each chapter also carries a "track" block naming the
    window of the source video it came from. The page then asks /api/track
    for that window and plays the REAL tracked body and hands instead of the
    synthesised ones — same clip contract, different backend. A step whose
    timestamp the extractor could not give is left synthesised rather than
    tracked from the wrong moment.
    """
    chapters = []
    for i, s in enumerate(doc["steps"]):
        title = s["text"][:52]
        narr = s["narration"] or s["text"]
        n = max(len(s["hand_actions"]), len(s["body_actions"]), 1)
        seconds = max(9, min(18, 5 + n * 3))
        if s["modality"] == "both":
            ch = {"mode": "compose", "seconds": seconds + 2,
                  "parts": [{"kind": "skeleton", "title": "posture",
                             "steps": s["body_actions"]},
                            {"kind": "hand", "title": "grip",
                             "steps": s["hand_actions"]}]}
        elif s["modality"] == "body":
            ch = {"mode": "skeleton", "seconds": seconds,
                  "motion_steps": s["body_actions"]}
        else:
            ch = {"mode": "hand", "seconds": seconds,
                  "hand_steps": s["hand_actions"]}
        ch.update({"narration": narr, "title": title,
                   "mined_at": s["at"], "step_no": i + 1})
        at = _seconds_at(s["at"])
        if track and at is not None:
            ch["track"] = {"url": doc["url"], "start": at,
                           "seconds": min(seconds, 20),
                           "kind": {"both": "compose", "body": "skeleton"}
                                   .get(s["modality"], "hand"),
                           "steps": s["hand_actions"] + s["body_actions"]}
        chapters.append(ch)

    if doc.get("warnings"):
        chapters.append({
            "mode": "anim", "seconds": 9, "title": "before you start",
            "narration": "One thing the video was clear about: "
                         + doc["warnings"][0],
            "steps": [{"beat": "safety", "els": [
                {"type": "text", "x": 25, "y": 34, "text": "watch out",
                 "color": "red", "in": "fade"},
                {"type": "box", "x": 25, "y": 22, "w": 34, "h": 10,
                 "label": doc["warnings"][0][:46], "color": "red", "in": "pop"}]}]})
    return chapters


def is_youtube(s):
    return bool(re.search(r"youtube\.com|youtu\.be", str(s), re.I)) or \
        bool(re.fullmatch(r"[A-Za-z0-9_-]{11}", str(s).strip()))


def mine_any(target, api_key, model=EXTRACT_MODEL):
    """One entry point for every source OpenClaw understands.

    a YouTube URL or id   -> transcript miner
    any other http(s) URL -> article miner (wikiHow and friends)
    anything else         -> a search phrase; the first hit with a usable
                             transcript gets mined
    """
    t = str(target).strip()
    if is_youtube(t):
        return mine(t, api_key, model)
    if re.match(r"https?://", t, re.I):
        return mine_article(t, api_key, model)
    hit = search_with_transcripts(t, want=1)[0]
    return mine(hit["id"], api_key, model)


def mine_to_lesson(url, api_key, hm=None, graph_path=None, model=EXTRACT_MODEL,
                   track=False):
    """The whole pipeline: source -> doc -> hivemind -> playable chapters."""
    doc = mine_any(url, api_key, model)
    task, flagged = None, []
    if hm is not None:
        task = hm.ingest(hivemind_doc(doc))
        # a mined source is exactly where a physically impossible instruction
        # arrives, so the narrow expert runs before the graph is saved
        try:
            import physics
            flagged = physics.dispute_in_graph(hm, task)
        except ImportError:
            pass
        if graph_path:
            hm.save(graph_path)
    return {"doc": doc, "lesson": doc_to_lesson(doc, track=track), "task": task,
            "physics": flagged,
            "stats": hm.stats() if hm is not None else None}


# ------------------------------------------------------------------ CLI

def _main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    arg = argv[1]
    urls = ([ln.strip() for ln in open(arg, encoding="utf-8")
             if ln.strip() and not ln.startswith("#")]
            if os.path.exists(arg) else [arg])

    from hivemind import Hivemind
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "hivemind_graph.json")
    hm = Hivemind(path)
    ok = 0
    for u in urls:
        try:
            out = mine_to_lesson(u, key, hm, path)
        except MineError as e:
            print(f"  skip  {u}\n        {e}")
            continue
        d = out["doc"]
        ok += 1
        mods = ", ".join(f"{s['modality']}" for s in d["steps"])
        print(f"  mined {d['title']}\n"
              f"        {len(d['steps'])} steps [{mods}]  -> {out['task']}")
    print(f"\n{ok}/{len(urls)} mined. graph: {hm.stats()}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
