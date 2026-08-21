"""
server.py — Python backend for the Agentic Video Tutor.

Serves index.html and proxies the browser's Claude calls so your
API key never appears in the page source. Also proxies the two
diffusion endpoints (text-to-video, text-to-image) and streams the
generated media back same-origin so the board canvas stays untainted
and the Record button keeps working.

Run:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=sk-ant-...        # console.anthropic.com
    python server.py
    # open http://localhost:8000

Deploy anywhere that runs Python (Render, Railway, Fly.io, a VPS):
    gunicorn -w 2 -b 0.0.0.0:8000 server:app
"""

import math
import os
import re
import time
from urllib.parse import urlparse

from flask import Flask, Response, request, jsonify, send_from_directory
import requests

from hivemind import (Hivemind, SEED_DOCS, parse_plain_howto,
                      backfill_symptoms, toks)
import motion
import lineform
import handform
import world
import coach
import openclaw
import posetrack
import physics
import screenread

ROOT = os.path.dirname(os.path.abspath(__file__))
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
MAX_BODY = 60_000          # guardrail: nobody stuffs a novel through your key
ALLOWED_MODELS = {"claude-sonnet-4-6", "claude-haiku-4-5-20251001"}
GRAPH_PATH = os.path.join(ROOT, "hivemind_graph.json")

REPLICATE_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "")
VIDEO_MODEL = os.environ.get("VIDEO_MODEL", "wan-video/wan-2.2-t2v-fast")
# swap via env: VIDEO_MODEL=thudm/cogvideox-t2v  (the CogVideoX paper's model,
# slower ~6 min/clip but it's the real thing)
IMAGE_MODEL = os.environ.get("IMAGE_MODEL", "black-forest-labs/flux-schnell")
# flux-schnell is ~2s and cents-per-image; swap for flux-dev / sdxl / any
# Replicate text-to-image model that takes {"prompt": ...}.

app = Flask(__name__, static_folder=None)

HM = Hivemind(GRAPH_PATH)
if HM.g.number_of_nodes() == 0:          # first boot: seed it
    for _doc in SEED_DOCS:
        HM.ingest(_doc)
    HM.save(GRAPH_PATH)
# a graph saved before the symptom index existed has none — add them without
# touching support, so retrieval works on already-deployed data
if backfill_symptoms(HM):
    HM.save(GRAPH_PATH)


@app.get("/")
def home():
    return send_from_directory(ROOT, "index.html")


@app.get("/health")
def health():
    return jsonify({"ok": True, "key_configured": bool(API_KEY),
                    "video": bool(REPLICATE_TOKEN), "video_model": VIDEO_MODEL,
                    "image": bool(REPLICATE_TOKEN), "image_model": IMAGE_MODEL,
                    "hivemind": HM.stats(),
                    # motion and lineform are pure maths — always available,
                    # no key, no GPU, no network
                    "motion": True, "lineform": True, "hand": True,
                    "world": True, "world_actors": list(world.KINDS),
                    # live form coaching is the same maths run backwards —
                    # the tracker runs in the student's browser, so the only
                    # thing this needs is a camera on their side
                    "coach": True, "coach_measures": coach.MEASURES,
                    # replaying the student's own track needs nothing
                    # installed here — the browser did the tracking
                    "replay": True,
                    # mining needs the key (for extraction) and nothing else
                    "mine": bool(API_KEY),
                    # real pixel tracking needs mediapipe + opencv + yt-dlp
                    "track": posetrack.available()[0],
                    "track_why": posetrack.available()[1],
                    # the screen agent: OCR over frames, for the specs that
                    # are shown rather than spoken
                    "screenread": screenread.available()[0],
                    "physics_rules": len(physics.RULES),
                    # real hand tracking needs mediapipe; without it /api/hand
                    # synthesises, and every clip says which one it was
                    "hand_tracking": handform.tracker_available(),
                    "hand_poses": sorted(handform.HAND_POSES),
                    "devices": {k: v["name"] for k, v in lineform.DEVICES.items()},
                    "shapes": lineform.SHAPES,
                    "gestures": sorted(motion.PRIMITIVES),
                    "papers": motion.PAPERS})


# ---------------------------------------------------------------- diffusion

def _replicate(model, payload, budget_s):
    """POST a prediction, poll until it settles. -> (output, error_message)."""
    headers = {"Authorization": f"Bearer {REPLICATE_TOKEN}",
               "Content-Type": "application/json", "Prefer": "wait=60"}
    try:
        r = requests.post(
            f"https://api.replicate.com/v1/models/{model}/predictions",
            headers=headers, json={"input": payload}, timeout=90)
        if r.status_code >= 400:
            try:
                j = r.json()
                msg = j.get("detail") or j.get("error") or f"HTTP {r.status_code}"
            except ValueError:
                msg = f"Replicate HTTP {r.status_code}"
            return None, str(msg)
        pred = r.json()
        poll = (pred.get("urls") or {}).get("get")
        t0 = time.time()
        while pred.get("status") in ("starting", "processing") and poll \
                and time.time() - t0 < budget_s:
            time.sleep(2)
            pred = requests.get(poll, headers=headers, timeout=30).json()
        if pred.get("status") == "succeeded":
            out = pred.get("output")
            if isinstance(out, list):
                out = out[0] if out else None
            if out:
                return str(out), None
        return None, str(pred.get("error") or f"generation {pred.get('status')}")
    except requests.RequestException as e:
        return None, f"Upstream unreachable: {e}"


@app.post("/api/video")
def video_gen():
    """Diffusion text-to-video via Replicate. {"prompt": "..."} -> {"video": url}.
    Returns {"disabled": true} when no REPLICATE_API_TOKEN is set so the
    frontend can fall back to a still or a sketch-only scene."""
    if not REPLICATE_TOKEN:
        return jsonify({"disabled": True})
    prompt = (request.get_json(silent=True) or {}).get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": {"message": "prompt required"}}), 400
    out, err = _replicate(VIDEO_MODEL, {"prompt": prompt}, 420)
    if err:
        return jsonify({"error": {"message": err}}), 502
    return jsonify({"video": out})


MAX_STEP_CLIPS = 4         # cost ceiling: a chapter cannot quietly bill for 9


@app.post("/api/videos")
def videos_gen():
    """One clip per step, generated concurrently. -> {"videos": [url|null, ...]}

    {"prompts": ["...", "..."]}  ->  {"videos": [...], "errors": [...]}

    A room shows the rig performing step 3 while the screen behind it plays
    footage of step 3, so the clips have to be cut to the steps rather than
    one clip per chapter. Sending them one at a time would serialise 30-90s
    of generation per step and starve the worker pool on a 2-worker deploy,
    so they go out together and the whole batch waits once.

    Order is preserved and a failed clip comes back as null: the screen goes
    quiet for that step and the lesson carries on.
    """
    if not REPLICATE_TOKEN:
        return jsonify({"disabled": True})
    prompts = (request.get_json(silent=True) or {}).get("prompts")
    if not isinstance(prompts, list) or not prompts:
        return jsonify({"error": {"message": "prompts required"}}), 400
    prompts = [str(p).strip()[:1200] for p in prompts[:MAX_STEP_CLIPS]]
    if not all(prompts):
        return jsonify({"error": {"message": "empty prompt in list"}}), 400

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        results = list(pool.map(
            lambda p: _replicate(VIDEO_MODEL, {"prompt": p}, 420), prompts))
    return jsonify({"videos": [out for out, _ in results],
                    "errors": [err for _, err in results],
                    "model": VIDEO_MODEL})


@app.post("/api/image")
def image_gen():
    """Diffusion text-to-image via Replicate. {"prompt": "..."} -> {"image": url}.
    Same disabled-when-tokenless contract as /api/video. Seconds, not minutes,
    so a scene can show a real generated still even when a clip is too slow."""
    if not REPLICATE_TOKEN:
        return jsonify({"disabled": True})
    prompt = (request.get_json(silent=True) or {}).get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": {"message": "prompt required"}}), 400
    inp = {"prompt": prompt}
    if IMAGE_MODEL.startswith("black-forest-labs/"):   # flux takes a ratio
        inp.update({"aspect_ratio": "4:3", "output_format": "webp"})
    out, err = _replicate(IMAGE_MODEL, inp, 120)
    if err:
        return jsonify({"error": {"message": err}}), 502
    return jsonify({"image": out})


@app.get("/api/media")
def media_proxy():
    """Stream a Replicate result through this origin.

    Drawing a cross-origin clip or still onto the board canvas taints it,
    which makes canvas.captureStream() unrecordable — the Record button
    would die the moment a scene used generated media. Proxying keeps
    everything same-origin. Locked to Replicate's CDN so this can't be
    used as an open proxy."""
    url = request.args.get("u", "")
    p = urlparse(url)
    host = p.netloc.lower().split(":")[0]
    if p.scheme != "https" or not (host == "replicate.delivery"
                                   or host.endswith(".replicate.delivery")):
        return jsonify({"error": {"message": "host not allowed"}}), 400
    try:
        r = requests.get(url, stream=True, timeout=60)
    except requests.RequestException as e:
        return jsonify({"error": {"message": f"Upstream unreachable: {e}"}}), 502
    return Response(r.iter_content(64 * 1024), status=r.status_code,
                    content_type=r.headers.get("Content-Type",
                                               "application/octet-stream"))


# ------------------------------------------------- motion / actuated curve

# Clips are a pure function of their request, and generating one costs real
# CPU (per-frame curve fitting for six chains plus the flow field), so the
# same lesson replayed does not pay twice.
_CLIP_CACHE = {}
CLIP_CACHE_MAX = 48


def _cached(key, make):
    if key in _CLIP_CACHE:
        return _CLIP_CACHE[key]
    clip = make()
    if len(_CLIP_CACHE) >= CLIP_CACHE_MAX:
        _CLIP_CACHE.pop(next(iter(_CLIP_CACHE)))
    _CLIP_CACHE[key] = clip
    return clip


@app.post("/api/motion")
def motion_gen():
    """How-to text -> 3D skeleton clip driven by actuated curve chains.

    {"steps": [...] | "text": "...", "seconds": 12, "yaw": 0.38,
     "constraint": false, "title": "..."}  ->  the clip index.html renders.

    Set "question" instead and the steps come from the hivemind, so the
    motion shown is the motion the graph actually knows about.
    """
    p = request.get_json(silent=True) or {}
    steps = p.get("steps") or p.get("text") or ""
    title = str(p.get("title", ""))[:120]
    source = ""
    if not steps and p.get("question"):
        ctx = HM.retrieve(str(p["question"]), k=1)
        if not ctx:
            return jsonify({"error": {"message": "hivemind has nothing on that yet"}}), 404
        steps = [ln.lstrip("0123456789. ") for ln in ctx.splitlines()
                 if re.match(r"^\d+\.\s", ln)]
        head = next((ln for ln in ctx.splitlines() if ln.startswith("## ")), "")
        title = title or head[3:].split("  (source")[0]
        source = "hivemind"
    if not steps:
        return jsonify({"error": {"message": "steps, text or question required"}}), 400
    if isinstance(steps, list):
        steps = [str(s)[:200] for s in steps[:10]]
    else:
        steps = str(steps)[:4000]

    seconds = _num(p.get("seconds"), 4.0, 30.0)
    fps = int(_num(p.get("fps"), 6, 20) or 12)
    yaw = _num(p.get("yaw"), -3.14, 3.14)
    key = ("m", repr(steps), seconds, fps, yaw, bool(p.get("constraint")), title)
    try:
        clip = _cached(key, lambda: motion.text_to_motion(
            steps, seconds=seconds, fps=fps,
            cam={"yaw": yaw} if yaw is not None else None,
            title=title, source=source, constraint=bool(p.get("constraint"))))
    except Exception as e:                       # a bad step list must not 500
        return jsonify({"error": {"message": f"motion synthesis failed: {e}"}}), 400
    return jsonify(clip)


def _steps_from_hivemind(question):
    """Question -> (steps, title). The graph is the source of the repair."""
    ctx = HM.retrieve(str(question), k=1)
    if not ctx:
        return [], ""
    steps = [re.sub(r"^\d+\.\s*", "", ln) for ln in ctx.splitlines()
             if re.match(r"^\d+\.\s", ln)]
    head = next((ln for ln in ctx.splitlines() if ln.startswith("## ")), "")
    return steps, head[3:].split("  (source")[0]


@app.post("/api/hand")
def hand_gen():
    """How-to text -> a hand-gesture clip: five finger chains + recognition.

    {"steps": [...] | "text": "..." | "question": "how do I fix ..."}

    With "question", the steps come from the hivemind, so the hands
    demonstrate a repair the graph actually knows rather than one the model
    invented. Returns {"disabled": false} style errors, never a 500.
    """
    p = request.get_json(silent=True) or {}
    steps = p.get("steps") or p.get("text") or ""
    title = str(p.get("title", ""))[:120]
    source = ""
    if not steps and p.get("question"):
        steps, found = _steps_from_hivemind(p["question"])
        if not steps:
            return jsonify({"error": {"message":
                "the hivemind has no repair for that yet"}}), 404
        title, source = title or found, "hivemind"
    if not steps:
        return jsonify({"error": {"message": "steps, text or question required"}}), 400
    steps = [str(s)[:200] for s in steps[:10]] if isinstance(steps, list) \
        else str(steps)[:4000]
    seconds = _num(p.get("seconds"), 4.0, 30.0)
    fps = int(_num(p.get("fps"), 6, 20) or 12)
    yaw = _num(p.get("yaw"), -3.14, 3.14) or 0.0
    key = ("h", repr(steps), seconds, fps, yaw, title)
    try:
        clip = _cached(key, lambda: handform.text_to_hand(
            steps, seconds=seconds, fps=fps, yaw=yaw, title=title, source=source))
    except Exception as e:
        return jsonify({"error": {"message": f"hand synthesis failed: {e}"}}), 400
    return jsonify(clip)


@app.post("/api/lineform")
def lineform_gen():
    """Actuated curve interface clip — the paper's own display primitives.

    {"shapes": ["curve","phone","wristband"], "device": "large"|"small",
     "snap": false, "seconds": 9, "values": [...]}   -> clip

    Or hand it vector data directly:
    {"svg": "M 10 10 L 90 ...."}  /  {"path": [[x,y,z], ...]}
    """
    p = request.get_json(silent=True) or {}
    device = p.get("device") if p.get("device") in lineform.DEVICES else "large"
    seconds = _num(p.get("seconds"), 3.0, 30.0)
    snap = bool(p.get("snap"))
    try:
        if p.get("svg"):
            path = lineform.from_svg_path(str(p["svg"])[:8000])
            key = ("lsvg", str(p["svg"])[:8000], device, seconds, snap)
            clip = _cached(key, lambda: lineform.clip_from_paths(
                [path], labels=["CAD outline"], seconds=seconds,
                device=device, snap=snap, title="CAD vector outline"))
        elif p.get("path"):
            pts = [(float(q[0]), float(q[1]), float(q[2]) if len(q) > 2 else 0.0)
                   for q in p["path"][:2000]]
            key = ("lpath", repr(pts)[:4000], device, seconds, snap)
            clip = _cached(key, lambda: lineform.clip_from_paths(
                [pts], labels=["vector path"], seconds=seconds,
                device=device, snap=snap, title="vector path"))
        else:
            shapes = [s for s in (p.get("shapes") or ["curve", "phone", "wristband"])
                      if s in lineform.SHAPES][:6] or ["curve"]
            vals = p.get("values")
            key = ("lshape", tuple(shapes), device, seconds, snap, repr(vals)[:400])
            clip = _cached(key, lambda: lineform.clip_from_shapes(
                shapes, seconds=seconds, device=device, snap=snap, values=vals))
    except Exception as e:
        return jsonify({"error": {"message": f"curve fit failed: {e}"}}), 400
    return jsonify(clip)


# ------------------------------------------------------- composite scenes

# Panel rectangles in normalised board space (x, y, w, h), one row per part
# count. Two parts sit side by side; three give the first pane the wide half
# because it is the one the narration is usually about; four is a plain grid.
COMPOSE_LAYOUTS = {
    1: [(0.0, 0.0, 1.0, 1.0)],
    2: [(0.0, 0.0, 0.5, 1.0), (0.5, 0.0, 0.5, 1.0)],
    3: [(0.0, 0.0, 0.56, 1.0), (0.56, 0.0, 0.44, 0.5), (0.56, 0.5, 0.44, 0.5)],
    4: [(0.0, 0.0, 0.5, 0.5), (0.5, 0.0, 0.5, 0.5),
        (0.0, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)],
}
COMPOSE_KINDS = ("skeleton", "hand", "lineform")
COMPOSE_MAX_PARTS = 4


def _compose_part(spec, seconds, fps, question):
    """One pane of a composite clip -> (clip, error_message).

    Each pane is an ordinary skeleton / hand / lineform clip built by the
    same functions the single-mode endpoints use, so a pane can do anything
    its standalone mode can. They share `seconds` and `fps`, which is what
    lets one playhead scrub all of them.
    """
    kind = spec.get("kind")
    if kind not in COMPOSE_KINDS:
        return None, f"unknown pane kind {kind!r}"
    title = str(spec.get("title", ""))[:120]
    source = ""

    if kind == "lineform":
        device = spec.get("device") if spec.get("device") in lineform.DEVICES \
            else "large"
        shapes = [s for s in (spec.get("shapes") or ["curve", "phone", "wristband"])
                  if s in lineform.SHAPES][:6] or ["curve"]
        clip = lineform.clip_from_shapes(
            shapes, seconds=seconds, fps=fps, device=device,
            snap=bool(spec.get("snap")), values=spec.get("values"), title=title)
        return clip, None

    # skeleton / hand both want steps, and both can take them from the graph
    steps = spec.get("steps") or spec.get("text") or ""
    if not steps and (spec.get("question") or question):
        steps, found = _steps_from_hivemind(spec.get("question") or question)
        if not steps:
            return None, "the hivemind has no repair for that yet"
        title, source = title or found, "hivemind"
    if not steps:
        return None, f"{kind} pane needs steps, text or question"
    steps = [str(s)[:200] for s in steps[:10]] if isinstance(steps, list) \
        else str(steps)[:4000]
    yaw = _num(spec.get("yaw"), -3.14, 3.14)

    if kind == "hand":
        clip = handform.text_to_hand(steps, seconds=seconds, fps=fps,
                                     yaw=yaw or 0.0, title=title, source=source)
    else:
        clip = motion.text_to_motion(
            steps, seconds=seconds, fps=fps,
            cam={"yaw": yaw} if yaw is not None else None,
            title=title, source=source,
            constraint=bool(spec.get("constraint")))
    return clip, None


def _really_about(question, title):
    """Is this retrieved task actually the thing that was asked for?

    Most of the title's own content words have to appear in the question.
    That is a deliberately blunt test — it is here to reject a match, not to
    find one — and it is stricter than retrieval on purpose, because the
    cost of a wrong answer is different: a loose match in a prompt is noise
    the model discards, a loose match here is a person being corrected
    against the wrong movement and told their form is bad.
    """
    t, q = set(toks(title or "")), set(toks(question or ""))
    if not t:
        return False
    hit = len(t & q)
    return hit >= max(2, math.ceil(0.6 * len(t)))


@app.post("/api/coach")
def coach_open():
    """Open a live coaching session. The rig stops performing and watches.

    {"question": "how do I squat"} | {"steps": [...]} | {"text": "..."}
     with optional "seconds", "fps", "yaw"

    ->  {"clip": <the skeleton clip to copy>,
         "reference": {"steps": [{"targets": {...}, "key": [...]}, ...],
                       plus the joint-triple and phrasing spec the page
                       needs to compute the same numbers live}}

    The tracking happens in the student's browser (MediaPipe over their
    webcam) because thirty poses a second is not something to put on a
    network; what this endpoint owns is the DEFINITION of what is measured
    and what counts as right, so the page cannot quietly grade on something
    else. /api/coach/score and /api/coach/report re-run that same code
    server-side when an authoritative number is wanted.

    The reference is any clip motion.py can make, so "coach me" works on a
    hivemind repair, a typed how-to, or a real video tracked by posetrack.
    """
    p = request.get_json(silent=True) or {}
    steps = p.get("steps") or p.get("text") or ""
    title = str(p.get("title", ""))[:120]
    source = ""
    # Prefer the graph — a real procedure beats one synthesised from a
    # sentence — but only on a CONFIDENT match. Retrieval ranks, it does not
    # threshold, so a topic the hivemind has never heard of still comes back
    # with whatever scored least badly. Elsewhere that is harmless: the block
    # goes into a prompt and the model ignores it. Here it would silently
    # grade somebody's squat against "patch a bicycle inner tube".
    if p.get("question"):
        found, head = _steps_from_hivemind(p["question"])
        if found and _really_about(p["question"], head):
            steps, title, source = found, title or head, "hivemind"
        elif not steps:
            steps = str(p["question"])[:4000]
    if not steps:
        return jsonify({"error": {"message":
                                  "question, steps or text required"}}), 400
    steps = [str(s)[:200] for s in steps[:10]] if isinstance(steps, list) \
        else str(steps)[:4000]

    seconds = _num(p.get("seconds"), 4.0, 30.0)
    fps = int(_num(p.get("fps"), 6, 20) or 12)
    yaw = _num(p.get("yaw"), -3.14, 3.14)
    key = ("c", repr(steps), seconds, fps, yaw, title)
    try:
        clip = _cached(key, lambda: motion.text_to_motion(
            steps, seconds=seconds, fps=fps,
            cam={"yaw": yaw} if yaw is not None else None,
            title=title, source=source))
        ref = coach.reference(clip)
    except Exception as e:
        return jsonify({"error": {"message": f"coach setup failed: {e}"}}), 400
    return jsonify({"clip": clip, "reference": ref})


def _live_angles(p):
    """Request body -> one pose's angles, however the page chose to send it.

    Either pre-computed ("angles", what the live overlay already has) or raw
    17-joint landmarks ("frame", when the caller would rather this side did
    the maths). Both end up in the same nine numbers.
    """
    a = p.get("angles")
    if isinstance(a, dict) and a:
        return {k: float(v) for k, v in a.items()
                if k in coach.MEASURES and isinstance(v, (int, float))}
    f = p.get("frame")
    if isinstance(f, list) and len(f) == len(motion.JOINTS):
        return coach.angles_of(f)
    return {}


@app.post("/api/coach/score")
def coach_score():
    """One live pose against one step. {"reference": ..., "step": 0,
    "angles": {...} | "frame": [[x,y,z] x17], "mirror": false}

    Stateless on purpose: the page holds the reference it was given, so this
    survives a restart, a second gunicorn worker, or a session left open
    over lunch. The live overlay scores locally at frame rate and only comes
    here when it wants the authoritative number.
    """
    p = request.get_json(silent=True) or {}
    ref = p.get("reference") or {}
    steps = ref.get("steps") or []
    if not steps:
        return jsonify({"error": {"message": "reference with steps required"}}), 400
    live = _live_angles(p)
    if not live:
        return jsonify({"error": {"message": "angles or frame required"}}), 400
    i = int(_num(p.get("step"), 0, len(steps) - 1) or 0)
    return jsonify(coach.score(live, steps[i], mirror=bool(p.get("mirror"))))


@app.post("/api/coach/report")
def coach_report():
    """The session, after the fact. {"reference": ..., "samples": [...]}

    Samples are {"step": i, "t": seconds, "angles": {...}} — a few a second,
    not a 30 Hz log. Everything is re-scored here rather than trusting the
    numbers the page put on screen, so the verdict and the reference come
    out of the same file.
    """
    p = request.get_json(silent=True) or {}
    ref = p.get("reference") or {}
    samples = p.get("samples")
    if not (ref.get("steps") and isinstance(samples, list)):
        return jsonify({"error": {"message":
                                  "reference and samples required"}}), 400
    try:
        return jsonify(coach.report(ref, samples[:4000],
                                    mirror=bool(p.get("mirror"))))
    except Exception as e:
        return jsonify({"error": {"message": f"report failed: {e}"}}), 400


@app.post("/api/compose")
def compose_gen():
    """Several clips in one scene — the modes stop being mutually exclusive.

    {"parts": [{"kind": "skeleton", "steps": [...]},
               {"kind": "hand",     "steps": [...]},
               {"kind": "lineform", "shapes": ["curve", "phone"]}],
     "seconds": 12, "title": "...", "question": "..."}

    -> {"kind": "compose", "parts": [<clip>, ...], "layout": [[x,y,w,h], ...]}

    Every pane is a full clip of its own kind sharing one timebase, and
    "layout" says where each one goes in normalised board space. Panes with
    no steps of their own fall back to the top-level "question", so a single
    hivemind repair can be shown as body, hands and device at once.
    """
    p = request.get_json(silent=True) or {}
    specs = p.get("parts")
    if not isinstance(specs, list) or not specs:
        return jsonify({"error": {"message": "parts required"}}), 400
    specs = [s for s in specs if isinstance(s, dict)][:COMPOSE_MAX_PARTS]
    if not specs:
        return jsonify({"error": {"message": "no usable parts"}}), 400

    seconds = _num(p.get("seconds"), 4.0, 30.0) or 12.0
    fps = int(_num(p.get("fps"), 6, 20) or 12)
    question = str(p.get("question", ""))[:400]
    title = str(p.get("title", ""))[:120]

    layout = p.get("layout")
    if not (isinstance(layout, list) and len(layout) >= len(specs)):
        layout = COMPOSE_LAYOUTS[len(specs)]
    layout = [[float(v) for v in r[:4]] for r in layout[:len(specs)]]

    key = ("c", repr(specs)[:4000], seconds, fps, question, title, repr(layout))
    try:
        def build():
            clips = []
            for spec in specs:
                clip, err = _compose_part(spec, seconds, fps, question)
                if err:
                    raise ValueError(err)
                clips.append(clip)
            return {"kind": "compose", "version": 1, "seconds": seconds,
                    "fps": fps, "title": title,
                    "source": next((c.get("source") for c in clips
                                    if c.get("source")), ""),
                    "layout": layout, "parts": clips,
                    "papers": sorted({p for c in clips
                                      for p in (c.get("papers") or
                                                ([c["paper"]] if c.get("paper") else []))})}
        clip = _cached(key, build)
    except Exception as e:
        return jsonify({"error": {"message": f"compose failed: {e}"}}), 400
    return jsonify(clip)


# ----------------------------------------------------------- one shared room

@app.post("/api/world")
def world_gen():
    """Every rig in ONE 3D space, filmed by one moving camera.

    {"actors": [{"kind": "skeleton", "steps": [...]},
                {"kind": "hand",     "steps": [...]},
                {"kind": "lineform", "shapes": [...], "wrap": true}],
     "seconds": 14, "screen": true, "question": "...", "title": "..."}

    -> {"kind": "world", "actors": [...placed clips...],
        "stage": {"ground": ..., "screen": ...}, "shots": [...camera track...]}

    Unlike /api/compose this does not split the board: the actors are placed
    at real positions in one room, so they occlude each other, the generated
    footage is a surface hanging in that room rather than a backdrop behind
    everything, and an actor can be BOUND to another actor's joint — which
    is the only way to show LineFORM's constraint affordance honestly.
    """
    p = request.get_json(silent=True) or {}
    specs = p.get("actors") or p.get("parts")
    if not isinstance(specs, list) or not specs:
        return jsonify({"error": {"message": "actors required"}}), 400

    seconds = _num(p.get("seconds"), 5.0, 40.0) or 14.0
    fps = int(_num(p.get("fps"), 6, 20) or 12)
    question = str(p.get("question", ""))[:400]
    title = str(p.get("title", ""))[:120]
    screen = p.get("screen") is not False
    key = ("w", repr(specs)[:4000], seconds, fps, question, title, screen,
           repr(p.get("stage"))[:600], repr(p.get("shots"))[:1200])
    try:
        clip = _cached(key, lambda: world.build_world(
            specs, seconds=seconds, fps=fps, question=question, title=title,
            screen=screen, stage=p.get("stage"), shots=p.get("shots"),
            resolve=_steps_from_hivemind))
    except Exception as e:
        return jsonify({"error": {"message": f"world build failed: {e}"}}), 400
    return jsonify(clip)


def _num(v, lo, hi):
    """Clamp an optional numeric request field; None when absent or junk."""
    try:
        if v is None:
            return None
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- hivemind

@app.post("/api/graph")
def graph_retrieve():
    """Question in -> grounded how-to context out (empty if not covered)."""
    q = (request.get_json(silent=True) or {}).get("question", "")
    if not q.strip():
        return jsonify({"error": {"message": "question required"}}), 400
    # "nodes" carries the step ids behind the prose, so a student who says
    # "that's wrong" can be attached to the exact step they meant
    return jsonify({"context": HM.retrieve(q, k=2),
                    "nodes": HM.retrieve_ids(q, k=2), "stats": HM.stats()})


@app.post("/api/correct")
def graph_correct():
    """Capture a disagreement. This is the loop that compounds.

    {"target": "task:.../step2", "kind": "wrong"|"missing"|"clarify"|"confirm",
     "text": "why", "replacement": "what it should say", "author": "tech_42"}

    Nothing is rewritten on the spot — a correction is evidence, queued for
    review at /api/review. What changes immediately is the step's confidence,
    so the tutor starts hedging on a disputed instruction straight away
    instead of after someone gets round to the queue.
    """
    p = request.get_json(silent=True) or {}
    target = str(p.get("target", "")).strip()
    kind = str(p.get("kind", "wrong")).strip()
    if not target:
        return jsonify({"error": {"message": "target node id required"}}), 400
    try:
        cid = HM.correct(target, kind=kind,
                         text=str(p.get("text", ""))[:500],
                         replacement=str(p.get("replacement", ""))[:500],
                         author=str(p.get("author", "student"))[:80])
    except KeyError:
        return jsonify({"error": {"message": f"no such node: {target}"}}), 404
    except ValueError as e:
        return jsonify({"error": {"message": str(e)}}), 400
    HM.save(GRAPH_PATH)
    return jsonify({"correction": cid, "target": target,
                    "confidence": HM.confidence_of(target),
                    "stats": HM.stats()})


@app.post("/api/diagnose")
def graph_diagnose():
    """Differential retrieval: candidates + the question that separates them.

    {"question": "something is wrong with my chain"} ->
      {"candidates": [...], "discriminator": {...}, "confident": false}

    A symptom rarely identifies one repair. Returning the top hit and hoping
    is what a search box does; asking the one discriminating question is what
    a tutor does.
    """
    p = request.get_json(silent=True) or {}
    q = str(p.get("question", "")).strip()
    if not q:
        return jsonify({"error": {"message": "question required"}}), 400
    k = int(_num(p.get("k"), 1, 6) or 3)
    out = HM.diagnose(q, k=k)
    out["steps"] = HM.find_steps(q, k=4)
    return jsonify(out)


@app.get("/api/prereq")
def graph_prereq():
    """Multi-hop: what a task needs, and what it is related to. ?task=task:..."""
    tid = request.args.get("task", "")
    if tid not in HM.g:
        return jsonify({"error": {"message": f"no such task: {tid}"}}), 404
    depth = int(_num(request.args.get("depth"), 1, 4) or 2)
    return jsonify({"task": tid, "title": HM.g.nodes[tid].get("title", ""),
                    **HM.prerequisites(tid, depth=depth)})


@app.get("/api/structure")
def graph_structure():
    """What the corpus actually contains — centrality and clusters."""
    return jsonify({"central": HM.central_tasks(10),
                    "communities": HM.communities(),
                    "stats": HM.stats()})


@app.post("/api/screenread")
def screen_read():
    """OCR a video's frames — the specs shown on screen but never spoken.

    {"url": "...", "start": 0, "seconds": 60}
      -> {"readings": [...], "specs": [...]}

    Transcript mining discards every number that is displayed rather than
    said, which is most of the numbers that decide whether a repair is right.
    """
    p = request.get_json(silent=True) or {}
    url = str(p.get("url", "")).strip()
    if not url:
        return jsonify({"error": {"message": "url required"}}), 400
    ok, why = screenread.available()
    if not ok:
        return jsonify({"error": {"message": f"OCR unavailable — {why}"},
                        "disabled": True}), 503
    start = _num(p.get("start"), 0.0, 36000.0) or 0.0
    seconds = _num(p.get("seconds"), 5.0, float(screenread.MAX_SECONDS)) or 60.0
    try:
        out = screenread.read_url(url, start=start, seconds=seconds)
    except screenread.ScreenError as e:
        return jsonify({"error": {"message": str(e)}}), 502
    except Exception as e:
        return jsonify({"error": {"message": f"screen read failed: {e}"}}), 502
    return jsonify(out)


@app.post("/api/physics")
def physics_check():
    """The narrow expert, on demand. {"steps": [...]} or {"text": "..."}.

    Corroboration says how many sources agreed; this says whether they can
    all be right. Every finding carries the law it applied, so a reviewer can
    check the reasoning instead of trusting the verdict.
    """
    p = request.get_json(silent=True) or {}
    steps = p.get("steps") if isinstance(p.get("steps"), list) else None
    if steps is None:
        txt = str(p.get("text", "")).strip()
        if not txt:
            return jsonify({"error": {"message": "steps or text required"}}), 400
        steps = [ln for ln in txt.splitlines() if ln.strip()] or [txt]
    steps = [str(s)[:400] for s in steps[:40]]
    found = physics.audit_steps(steps)
    return jsonify({"checked": len(steps), "flagged": len(found),
                    "findings": found,
                    "rules": len(physics.RULES)})


@app.get("/api/review")
def graph_review():
    """The queue: what the field disputes, and what is waiting on a ruling."""
    return jsonify({"pending": HM.pending_corrections(50),
                    "disputed": HM.disputed(50), "stats": HM.stats()})


@app.post("/api/review")
def graph_review_apply():
    """Rule on a correction. {"id": "correction:...", "accept": true}"""
    p = request.get_json(silent=True) or {}
    cid = str(p.get("id", "")).strip()
    if not cid:
        return jsonify({"error": {"message": "correction id required"}}), 400
    try:
        out = HM.apply_correction(cid, accept=p.get("accept") is not False)
    except KeyError:
        return jsonify({"error": {"message": f"no such correction: {cid}"}}), 404
    HM.save(GRAPH_PATH)
    return jsonify({**out, "stats": HM.stats()})


@app.post("/api/ingest")
def graph_ingest():
    """Grow the hivemind: {"title": "...", "text": "raw article text"}.
    Uses the offline parser; swap in extract_with_llm for messy sources."""
    p = request.get_json(silent=True) or {}
    if not p.get("title") or not p.get("text"):
        return jsonify({"error": {"message": "title and text required"}}), 400
    doc = parse_plain_howto(p["title"], p["text"], p.get("source", "manual"))
    if not doc["steps"]:
        return jsonify({"error": {"message":
            "no steps found — use numbered lines like '1. Do this'"}}), 400
    tid = HM.ingest(doc)
    flagged = physics.dispute_in_graph(HM, tid)
    HM.save(GRAPH_PATH)
    return jsonify({"task": tid, "steps": len(doc["steps"]),
                    "physics": flagged, "stats": HM.stats()})


@app.post("/api/track")
def track_gen():
    """The pixels, not the words: a real video becomes a real tracked clip.

    {"url": "https://youtube.com/watch?v=...", "start": 42, "seconds": 10,
     "kind": "skeleton" | "hand" | "compose", "steps": [...], "title": "..."}

    -> the same clip contract /api/motion and /api/hand return, with the
       backend field saying "blazepose-tracked" / "mediapipe-tracked" so a
       tracked clip can never be mistaken for a synthesised one.

    "compose" tracks the body and the fingers from the same window and puts
    them in two panes, which is the honest version of the composite scene:
    both panes are the same person at the same moment.

    Tracking costs real CPU and a download, so results are cached like every
    other clip and the video file is kept between calls.
    """
    p = request.get_json(silent=True) or {}
    url = str(p.get("url", "")).strip()
    if not url:
        return jsonify({"error": {"message": "url required"}}), 400
    ok, why = posetrack.available()
    if not ok:
        return jsonify({"error": {"message": f"tracking unavailable — {why}"},
                        "disabled": True}), 503

    kind = p.get("kind") if p.get("kind") in ("skeleton", "hand", "compose") \
        else "compose"
    start = _num(p.get("start"), 0.0, 36000.0) or 0.0
    seconds = _num(p.get("seconds"), 3.0, float(posetrack.MAX_SECONDS)) or 10.0
    steps = p.get("steps") if isinstance(p.get("steps"), list) else None
    steps = [str(s)[:200] for s in (steps or [])[:10]] or None
    title = str(p.get("title", ""))[:120]

    key = ("t", url, kind, start, seconds, repr(steps), title)
    try:
        def build():
            want = ("body",) if kind == "skeleton" else \
                   ("hand",) if kind == "hand" else ("body", "hand")
            out = posetrack.track_url(url, start=start, seconds=seconds,
                                      want=want, steps=steps, title=title,
                                      source="tracked")
            errs = out.get("errors") or {}
            if kind == "skeleton":
                if not out.get("skeleton"):
                    raise ValueError(errs.get("skeleton", "no body tracked"))
                return out["skeleton"]
            if kind == "hand":
                if not out.get("hand"):
                    raise ValueError(errs.get("hand", "no hand tracked"))
                return out["hand"]
            panes = [c for c in (out.get("skeleton"), out.get("hand")) if c]
            if not panes:
                raise ValueError("; ".join(errs.values()) or "nothing tracked")
            secs = min(c["seconds"] for c in panes)
            return {"kind": "compose", "version": 1, "seconds": secs,
                    "fps": panes[0]["fps"], "title": title,
                    "source": "tracked", "tracked": True,
                    "missing": list(errs),
                    "layout": COMPOSE_LAYOUTS[len(panes)], "parts": panes,
                    "papers": sorted({q for c in panes
                                      for q in (c.get("papers") or [])})}
        clip = _cached(key, build)
    except posetrack.TrackError as e:
        return jsonify({"error": {"message": str(e)}}), 502
    except Exception as e:
        return jsonify({"error": {"message": f"tracking failed: {e}"}}), 502
    return jsonify(clip)


# A live session is seconds long like any other chapter, and the payload is
# raw landmarks, so this is both a sanity bound and a size guard.
MAX_REPLAY_FRAMES = posetrack.MAX_SECONDS * posetrack.TRACK_FPS
MIN_REPLAY_FRAMES = 6      # fewer than this is noise, not a demonstration
MIN_PANE_SHARE = 0.4       # a pane that saw less of the take than this is cut
                           # rather than allowed to truncate the others


def _landmark_frames(raw, n_points):
    """Untrusted landmark stream -> clean [[(x,y,z) x n_points], ...].

    Frames of the wrong width or with non-finite numbers are dropped rather
    than rejected: a tracker drops a hand for a few frames all the time, and
    losing the whole take because of it would be absurd.
    """
    out = []
    if not isinstance(raw, list):
        return out
    for f in raw[:MAX_REPLAY_FRAMES]:
        if not isinstance(f, list) or len(f) != n_points:
            continue
        pts = []
        for p in f:
            if not isinstance(p, list) or len(p) < 3:
                break
            try:
                x, y, z = float(p[0]), float(p[1]), float(p[2])
            except (TypeError, ValueError):
                break
            if not all(map(math.isfinite, (x, y, z))):
                break
            pts.append((x, y, z))
        if len(pts) == n_points:
            out.append(pts)
    return out


@app.post("/api/replay")
def replay_gen():
    """The student's OWN motion -> the clip the rigs perform.

    {"body": [[[x,y,z] x33], ...],   BlazePose world landmarks
     "hand": [[[x,y,z] x21], ...],   MediaPipe hand world landmarks, y up
     "fps": 12, "kind": "compose"|"skeleton"|"hand",
     "steps": [...], "title": "..."}

    -> the same clip contract /api/motion, /api/hand and /api/track return.

    This is the other end of /api/coach. The camera tracked the student in
    their own browser; this turns that stream into a clip, which means the
    thing they just did comes back performed by the servo chains, with the
    gestures recognised from THEIR frames rather than from a text prompt.

    Unlike /api/track it needs no mediapipe, no OpenCV, no yt-dlp and no
    download — the tracking already happened on the other side of the wire.
    A deployment where /api/track is unavailable can still do this, which is
    most deployments.

    The clip goes into the timeline as an ordinary chapter, so pausing,
    interrupting, asking and resuming all work on it exactly as they do on a
    synthesised one: nothing here is a special case for the player.
    """
    p = request.get_json(silent=True) or {}
    kind = p.get("kind") if p.get("kind") in ("skeleton", "hand", "compose") \
        else "compose"
    fps = int(_num(p.get("fps"), 4, 30) or posetrack.TRACK_FPS)
    title = str(p.get("title", ""))[:120] or "what you just did"
    steps = p.get("steps") if isinstance(p.get("steps"), list) else None
    steps = [str(s)[:200] for s in (steps or [])[:10]] or None

    # 33 in, 17 out: motion.from_landmarks applies BLAZEPOSE_MAP, the same
    # decimation posetrack does server-side, so the browser must send the
    # full BlazePose stream rather than a pre-reduced one.
    body = _landmark_frames(p.get("body"), 33)
    hand = _landmark_frames(p.get("hand"), len(handform.LANDMARKS))

    panes, errs = [], {}
    if kind in ("skeleton", "compose"):
        if len(body) >= MIN_REPLAY_FRAMES:
            try:
                panes.append(motion.clip_from_landmarks(
                    motion.from_landmarks(body), fps=fps, steps=steps,
                    title=title, source="live", backend="blazepose-live"))
            except Exception as e:
                errs["skeleton"] = f"body replay failed: {e}"
        else:
            errs["skeleton"] = (f"only {len(body)} usable body frames — "
                                f"stand back so your whole body is in shot")
    if kind in ("hand", "compose"):
        if len(hand) >= MIN_REPLAY_FRAMES:
            try:
                # handform's own planner, not a hand-rolled dict: it also
                # classifies a pose per step, and clip_from_frames requires
                # that key. Building the plan literally here is what breaks
                # posetrack.video_to_hand (KeyError: 'pose').
                plan = handform.plan_from_steps(
                    steps, len(hand) / float(fps))[0] if steps else None
                panes.append(handform.clip_from_frames(
                    hand, fps, title=title, source="live",
                    backend="mediapipe-live", plan=plan))
            except Exception as e:
                errs["hand"] = f"hand replay failed: {e}"
        else:
            errs["hand"] = (f"only {len(hand)} usable hand frames — "
                            f"hold a hand up where the camera can see it")

    if not panes:
        return jsonify({"error": {"message":
                        "; ".join(errs.values()) or "no usable frames"}}), 400

    # Panes share one playhead, so a composite can only run as long as its
    # SHORTEST pane. A hand that drifted in and out of shot for half a second
    # would therefore truncate a fifteen-second body replay to half a second —
    # the rig would twitch once and stop. A pane that saw far less of the take
    # than the others is dropped instead of allowed to dictate the length; it
    # is reported in "missing" so the page can say why it is not there.
    if len(panes) > 1:
        longest = max(c["seconds"] for c in panes)
        keep = []
        for c in panes:
            if c["seconds"] >= longest * MIN_PANE_SHARE:
                keep.append(c)
            else:
                errs[c["kind"]] = (f"{c['kind']} was only in shot for "
                                   f"{c['seconds']:.1f}s of {longest:.1f}s")
        panes = keep

    if len(panes) == 1:
        return jsonify({**panes[0], "missing": list(errs)})
    secs = min(c["seconds"] for c in panes)
    return jsonify({"kind": "compose", "version": 1, "seconds": secs,
                    "fps": panes[0]["fps"], "title": title,
                    "source": "live", "tracked": True, "missing": list(errs),
                    "layout": COMPOSE_LAYOUTS[len(panes)], "parts": panes,
                    "papers": sorted({q for c in panes
                                      for q in (c.get("papers") or [])})})


MAX_MINE_URLS = 8          # a crawl is a queue, not a denial-of-service


@app.post("/api/mine")
def mine_video():
    """OpenClaw: a how-to VIDEO becomes a lesson the rigs perform.

    {"url": "https://youtube.com/watch?v=..."}     a video (transcript)
    {"url": "https://www.wikihow.com/..."}         an article
    {"url": "how to fix a bike chain"}             a search phrase
    {"urls": ["...", "..."]}                       a small crawl
    {"url": "...", "track": true}                  track the real pixels
    {"url": "...", "ingest": false}                do not grow the graph

    -> {"title": ..., "lesson": [chapters], "doc": {...}, "stats": {...}}

    Every step comes back classified as something the fingers do, something
    the whole body does, or both — and "both" becomes a compose chapter, so
    the posture and the grip play side by side instead of the student having
    to pick one. The steps are also ingested into the hivemind, so the video
    answers questions long after it has finished playing.
    """
    p = request.get_json(silent=True) or {}
    urls = p.get("urls") if isinstance(p.get("urls"), list) else \
        ([p["url"]] if p.get("url") else [])
    urls = [str(u)[:300] for u in urls if str(u).strip()][:MAX_MINE_URLS]
    if not urls:
        return jsonify({"error": {"message": "url or urls required"}}), 400
    if not API_KEY:
        return jsonify({"error": {"message":
            "Server has no ANTHROPIC_API_KEY set — export it and restart"}}), 500
    do_ingest = p.get("ingest") is not False
    # tracking is opt-in per request and silently impossible without the deps
    do_track = bool(p.get("track")) and posetrack.available()[0]

    mined, failed = [], []
    for u in urls:
        try:
            out = openclaw.mine_to_lesson(
                u, API_KEY, HM if do_ingest else None,
                GRAPH_PATH if do_ingest else None, track=do_track)
        except openclaw.MineError as e:
            failed.append({"url": u, "message": str(e)})
            continue
        except Exception as e:                  # a bad video must not 500
            failed.append({"url": u, "message": f"mining failed: {e}"})
            continue
        mined.append(out)

    if not mined:
        return jsonify({"error": {"message": failed[0]["message"]
                                  if failed else "nothing mined"},
                        "failed": failed}), 502

    first = mined[0]
    return jsonify({"title": first["doc"]["title"], "tracked": do_track,
                    "lesson": first["lesson"], "doc": first["doc"],
                    "mined": [{"title": m["doc"]["title"], "task": m["task"],
                               "url": m["doc"]["url"],
                               "steps": len(m["doc"]["steps"]),
                               "modalities": [s["modality"]
                                              for s in m["doc"]["steps"]]}
                              for m in mined],
                    "failed": failed, "stats": HM.stats()})


@app.post("/api/claude")
def claude_proxy():
    if not API_KEY:
        return jsonify({"error": {"message":
            "Server has no ANTHROPIC_API_KEY set — export it and restart"}}), 500

    body = request.get_data()
    if len(body) > MAX_BODY:
        return jsonify({"error": {"message": "Request too large"}}), 413

    # optional sanity check: only allow the models the app actually uses
    payload = request.get_json(silent=True) or {}
    if payload.get("model") not in ALLOWED_MODELS:
        return jsonify({"error": {"message": "Model not allowed"}}), 400

    try:
        r = requests.post(
            ANTHROPIC_URL,
            headers={
                "Content-Type": "application/json",
                "x-api-key": API_KEY,
                "anthropic-version": "2023-06-01",
            },
            data=body,
            timeout=120,
        )
    except requests.RequestException as e:
        return jsonify({"error": {"message": f"Upstream unreachable: {e}"}}), 502

    return r.content, r.status_code, {"Content-Type": "application/json"}


if __name__ == "__main__":
    print("Agentic Video Tutor -> http://localhost:8000"
          + ("" if API_KEY else "   [WARNING: ANTHROPIC_API_KEY not set]"))
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=False)
