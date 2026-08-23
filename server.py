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

import hashlib
import json
import math
import os
import re
import sys
import threading
import time
from urllib.parse import urlparse, quote

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
import pairing
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
                    # the vendored MoneyPrinterTurbo pipeline: neural
                    # narration and real stock footage, live in the lesson
                    "mpt": mpt_status(),
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


# ------------------------------------------- MoneyPrinterTurbo (mpt/) bridge
#
# MoneyPrinterTurbo is vendored whole in mpt/ and imported as a library, not
# run as a second server. It supplies the two things the board could not get
# for itself, and both go INTO the lesson rather than into an export:
#
#   /api/tts     real neural narration WITH WORD TIMINGS, replacing the
#                browser's speechSynthesis. Edge's voices need no API key,
#                so unlike every other generated asset here this one is free.
#   /api/stock   footage that was really filmed, under any chapter and under
#                any interrupt answer — seconds instead of 30-90s, free
#                instead of paid, and a real pinch instead of a plausible one.
#
# There is deliberately no MP4 export. The point is not to produce a file at
# the end; it is that the explanation itself is real film, narrated by a real
# voice, with the annotator's marker strokes drawn on top of both — including
# when the student interrupts and the answer brings its own footage.
#
# Everything is imported LAZILY and never at boot. mpt/ pulls in moviepy,
# edge_tts, openai and pydantic; a deployment without them must still serve
# the board, so every route below degrades exactly the way /api/video does
# without REPLICATE_API_TOKEN — {"disabled": true, "why": ...} — and the
# frontend falls back rather than erroring.

MPT_DIR = os.path.join(ROOT, "mpt")
# MPT names voices "<edge voice>-<Gender>"; parse_voice_name strips the suffix.
MPT_VOICES = {"en": os.environ.get("MPT_VOICE_EN", "en-US-JennyNeural-Female"),
              "id": os.environ.get("MPT_VOICE_ID", "id-ID-GadisNeural-Female")}
MAX_TTS_CHARS = 1200       # a chapter's narration is 1-2 sentences, not a book
MAX_STOCK_CLIPS = 4        # same cost ceiling as MAX_STEP_CLIPS above

_MPT = {"tried": False, "mods": None, "why": ""}


def _mpt():
    """Import the vendored pipeline once. -> (modules dict or None, why)."""
    if not _MPT["tried"]:
        _MPT["tried"] = True
        if not os.path.isdir(MPT_DIR):
            _MPT["why"] = "mpt/ is not vendored next to server.py"
        else:
            try:
                if MPT_DIR not in sys.path:
                    # appended, not inserted: mpt/ has its own top-level `app`
                    # package and this file must keep resolving its own modules
                    sys.path.append(MPT_DIR)
                from app.config import config as _cfg
                from app.utils import utils as _utils
                from app.models import schema as _schema
                from app.services import voice as _voice
                from app.services import material as _material
                from app.services import video as _video
                _MPT["mods"] = {"config": _cfg, "utils": _utils,
                                "schema": _schema, "voice": _voice,
                                "material": _material, "video": _video}
            except Exception as e:          # missing dep, bad config, ...
                _MPT["why"] = f"{type(e).__name__}: {e}"
    return _MPT["mods"], _MPT["why"]


def mpt_status():
    """What /health reports: whether the pipeline imports, and which providers
    actually have keys. Voice is the interesting one — it is the only
    generated asset in this whole app that costs nothing."""
    mods, why = _mpt()
    if not mods:
        return {"ok": False, "why": why or "unavailable"}
    stock_why = ""
    try:
        mods["material"].get_api_key("pexels_api_keys")
    except Exception as e:
        stock_why = next((ln for ln in str(e).splitlines() if ln.strip()),
                         "no api key").strip()
    return {"ok": True, "tts": True, "voices": MPT_VOICES,
            "stock": not stock_why, "stock_why": stock_why,
            # text-to-video generation of the actions themselves — the
            # Make-A-Video lineage, via the same Replicate model /api/video
            # uses. Paid per clip, so reported separately from stock.
            "generate": bool(REPLICATE_TOKEN),
            "perceive": bool(API_KEY),
            # the look, not just the assets: the page draws its captions with
            # MoneyPrinterTurbo's own fonts and subtitle styling so a chapter
            # on the board reads like one of its rendered shorts
            "graphics": True, "fonts": mpt_fonts(), "subtitle": MPT_SUBTITLE,
            "transitions": MPT_TRANSITIONS}


# MoneyPrinterTurbo's subtitle look, as its VideoParams defaults define it:
# white fill, black outline, sat at the bottom of the frame. These are the
# numbers its renderer uses, kept here so the board and a rendered short would
# put the same words in the same place in the same face.
MPT_SUBTITLE = {"font": os.environ.get("MPT_FONT", "BeVietnamPro-Bold.ttf"),
                "size": 60, "fore": "#FFFFFF", "stroke": "#000000",
                "stroke_width": 1.5, "position": "bottom",
                "custom_position": 70.0,
                # its optional rounded plate, drawn under the text
                "background": False, "background_color": "#000000",
                "background_alpha": 140, "background_radius": 16,
                # which of its transitions a chapter enters on; "Shuffle"
                # picks per chapter the way its renderer picks per clip
                "transition": os.environ.get("MPT_TRANSITION", "FadeIn")}

# app/services/utils/video_effects.py — the transitions its renderer applies
# between clips. The board cuts between chapters, so the same names mean the
# same thing here.
MPT_TRANSITIONS = ["None", "FadeIn", "FadeOut", "SlideIn", "SlideOut",
                   "ZoomIn", "ZoomOut"]


def _font_dir():
    return os.path.join(MPT_DIR, "resource", "fonts")


def mpt_fonts():
    """The faces MoneyPrinterTurbo ships. -> [{"file","name","latin"}, ...]

    Reported so the page can @font-face them instead of falling back to
    whatever the operating system happens to have. `latin` matters because
    half of these are CJK faces: a lesson in English wants BeVietnamPro,
    not STHeiti, and picking blind would letterbox the captions in tofu."""
    try:
        files = sorted(f for f in os.listdir(_font_dir())
                       if f.lower().endswith((".ttf", ".otf", ".ttc")))
    except OSError:
        return []
    return [{"file": f, "name": os.path.splitext(f)[0],
             # .ttc collections here are the CJK ones; the .ttf faces are the
             # Latin-capable ones, and the browser cannot load .ttc at all
             "latin": f.lower().endswith((".ttf", ".otf"))} for f in files]


@app.get("/api/mpt/font")
def mpt_font():
    """Serve one of MoneyPrinterTurbo's bundled fonts to the page.

    Confined to resource/fonts by name match against the directory listing —
    not by string surgery on the request — so there is no path to walk."""
    want = request.args.get("f", "")
    if want not in {f["file"] for f in mpt_fonts()}:
        return jsonify({"error": {"message": "unknown font"}}), 404
    return send_from_directory(_font_dir(), want, conditional=True,
                               max_age=86400)


def _tutor_dir():
    """Everything this bridge writes lives under one directory, so
    /api/mpt/media can serve from it without exposing the rest of the disk."""
    d = os.path.join(MPT_DIR, "storage", "tutor")
    os.makedirs(d, exist_ok=True)
    return d


def _tutor_url(path):
    """A same-origin URL for a file inside the tutor directory."""
    rel = os.path.relpath(os.path.realpath(path), os.path.realpath(_tutor_dir()))
    return "/api/mpt/media?f=" + quote(rel.replace(os.sep, "/"))


@app.get("/api/mpt/media")
def mpt_media():
    """Serve a file this bridge generated, same-origin.

    Same reason /api/media proxies Replicate rather than linking it: a
    cross-origin clip drawn onto the board taints the canvas, and a tainted
    canvas cannot be captureStream()'d — the Record button would die the
    moment a lesson used stock footage. Resolved and confined to
    mpt/storage/tutor so a crafted name cannot walk out of it."""
    base = os.path.realpath(_tutor_dir())
    path = os.path.realpath(os.path.join(base, request.args.get("f", "")))
    if not path.startswith(base + os.sep) or not os.path.isfile(path):
        return jsonify({"error": {"message": "not found"}}), 404
    return send_from_directory(os.path.dirname(path), os.path.basename(path),
                               conditional=True)


@app.post("/api/tts")
def tts_gen():
    """Neural narration with word timings.

    {"text": "...", "lang": "en"|"id", "voice": "...", "rate": 1.0}
    -> {"audio": "/api/mpt/media?f=...", "seconds": 3.02,
        "cues": [{"t": 0.1, "d": 0.32, "w": "Tighten"}, ...]}

    The cues are why this is worth a round trip rather than just an <audio>
    src. The avatar's lipsync and the caption highlighting are both driven by
    word boundaries, which speechSynthesis emits and a plain audio element
    does not — without them the mouth would flap on a timer while a real
    voice said something else. Edge gives them for free.

    Cached on a hash of (voice, rate, text), so a replayed chapter and a
    re-run lesson cost one file read.
    """
    mods, why = _mpt()
    if not mods:
        return jsonify({"disabled": True, "why": why})
    body = request.get_json(silent=True) or {}
    text = str(body.get("text") or "").strip()[:MAX_TTS_CHARS]
    if not text:
        return jsonify({"error": {"message": "text required"}}), 400
    lang = str(body.get("lang") or "en")
    voice = str(body.get("voice") or MPT_VOICES.get(lang) or MPT_VOICES["en"])
    try:
        rate = max(0.5, min(2.0, float(body.get("rate", 1.0))))
    except (TypeError, ValueError):
        rate = 1.0

    key = hashlib.sha1(f"{voice}|{rate}|{text}".encode("utf-8")).hexdigest()[:16]
    mp3 = os.path.join(_tutor_dir(), f"tts-{key}.mp3")
    side = mp3 + ".json"
    if os.path.isfile(mp3) and os.path.getsize(mp3) and os.path.isfile(side):
        try:
            with open(side, encoding="utf-8") as fh:
                return jsonify(json.load(fh))
        except (OSError, ValueError):
            pass                                    # fall through and rewrite
    try:
        sm = mods["voice"].tts(text, voice, rate, mp3)
        if sm is None or not os.path.isfile(mp3) or not os.path.getsize(mp3):
            return jsonify({"error": {"message": "tts produced no audio"}}), 502
        cues = []
        for c in (getattr(sm, "cues", None) or []):
            t0 = c.start.total_seconds()
            cues.append({"t": round(t0, 3),
                         "d": round(max(0.0, c.end.total_seconds() - t0), 3),
                         "w": str(c.content)})
        seconds = float(mods["voice"].get_audio_duration(sm))
    except Exception as e:
        return jsonify({"error": {"message": f"tts failed: {e}"}}), 502
    out = {"audio": _tutor_url(mp3), "seconds": round(seconds, 3),
           "cues": cues, "voice": voice}
    try:
        with open(side, "w", encoding="utf-8") as fh:
            json.dump(out, fh)
    except OSError:
        pass                     # the audio is what matters; the cache is not
    return jsonify(out)


# ------------------------------------------------ is this a how-to at all?
#
# The action reel is film of a MOVEMENT — hands doing a thing, cut action by
# action. That is the strongest explanation this board has for a procedure and
# the WORST one for an idea: "how photosynthesis works" has no gesture to
# film, and a stock search for it returns pretty greenery that illustrates
# nothing. So the reel is gated on the lesson actually being technical
# how-to — a wiki-style procedure someone performs — rather than offered for
# every topic and quietly wasting a search on the ones it cannot serve.

# Procedural verbs: what a person DOES with their hands. Deliberately verbs,
# not topics — "install a water pump" is a procedure, "the water cycle" is not.
_HOWTO_VERBS = {
    "fix", "repair", "replace", "install", "remove", "change", "clean",
    "build", "make", "assemble", "attach", "mount", "adjust", "tighten",
    "loosen", "connect", "wire", "solder", "paint", "cut", "drill", "sand",
    "weld", "patch", "seal", "unclog", "sharpen", "lubricate", "grease",
    "bleed", "flush", "swap", "fit", "wrap", "tie", "sew", "glue", "screw",
    "bolt", "calibrate", "reset", "restore", "service", "test", "measure",
}
# Explanatory openers: the question is about a mechanism, not a task.
_CONCEPT_MARKERS = ("how does", "how do", "why does", "why do", "why is",
                    "what is", "what are", "explain", "works", "history of",
                    "difference between", "theory", "causes of")


@app.post("/api/howto")
def howto_gate():
    """Is this topic a technical how-to, or a concept? -> {"technical": bool}

    {"topic": "how to replace a bicycle chain"}
      -> {"technical": true, "why": "procedural verb 'replace'",
          "task": "Replace a bicycle chain", "steps": 7}

    Two independent signals, because either alone is wrong often enough:

    - THE GRAPH. If the hivemind already holds a task whose steps match this
      topic, it is a procedure by demonstration — somebody wrote the steps
      down. This is the strong signal and it needs no keyword list.
    - THE PHRASING. A procedural verb with no explanatory opener in front of
      it. "Replace a chain" passes; "how does a chain work" does not, and
      neither does "the history of chains", which contains no verb at all.

    Reported with the reason so the page can say WHY a lesson is getting film
    of hands or a diagram, instead of the choice looking arbitrary."""
    topic = str((request.get_json(silent=True) or {}).get("topic", "")).strip()
    if not topic:
        return jsonify({"error": {"message": "topic required"}}), 400
    low = topic.lower()[:300]

    # the graph first: a matching task WITH steps settles it
    task, nsteps = "", 0
    try:
        hits = HM.find_steps(topic, k=5)
        if hits and hits[0]["score"] > 0:
            task = hits[0].get("task", "")
            nsteps = sum(1 for h in hits if h.get("task") == task)
    except Exception:                                   # graph is advisory
        hits = []
    # Step-level retrieval scores any step sharing a token, so a topic can
    # land on a procedure it has nothing to do with — "replace a bicycle
    # chain" matching "Build a Large Modern Minecraft House" on the strength
    # of common words. Believe the graph only when the TASK ITSELF is about
    # the same thing, or the reason given is nonsense and, worse, a concept
    # could inherit a procedure's verdict by accident.
    if task and nsteps >= 2:
        want, have = set(toks(topic)), set(toks(task))
        if want & have:
            return jsonify({"technical": True, "task": task, "steps": nsteps,
                            "why": f"the hivemind already has {nsteps} steps "
                                   f"for {task!r}"})
        task, nsteps = "", 0        # unrelated match: judge on the words alone

    concept = any(m in low for m in _CONCEPT_MARKERS)
    # toks() STEMS, so the verb list has to be stemmed by the same function or
    # "replace" arrives as "replac" and matches nothing. Stemming both sides
    # also buys the inflections for free: replacing, tightened, unclogged.
    words = set(toks(low))
    verb = next((v for v in sorted(_HOWTO_VERBS)
                 if words & set(toks(v))), "")
    # "how to fix X" is procedural even though it starts with "how"
    imperative = low.startswith("how to ") or low.startswith("how do i ")
    technical = bool(verb) and (imperative or not concept)
    return jsonify({"technical": technical, "task": task, "steps": nsteps,
                    "why": (f"procedural verb {verb!r}" if technical
                            else "reads as a concept, not a procedure"
                                 if concept else "no procedural verb")})


@app.post("/api/stock")
def stock_gen():
    """Real footage per step, from a stock library instead of a diffusion model.

    {"terms": ["seat the chain on the teeth", ...], "seconds": 5}
    -> {"videos": [url|null, ...], "errors": [...], "source": "pexels"}

    Deliberately the SAME response shape as /api/videos, so a world chapter's
    step_videos machinery does not care which provider filled the screen — a
    failed term comes back null, that step's screen goes quiet, and the lesson
    carries on, exactly as it does with diffusion.

    One search per step, first unused result wins: the clips stay in step
    order and two steps never show the same footage.
    """
    mods, why = _mpt()
    if not mods:
        return jsonify({"disabled": True, "why": why})
    body = request.get_json(silent=True) or {}
    terms = body.get("terms") or body.get("prompts")
    if not isinstance(terms, list) or not terms:
        return jsonify({"error": {"message": "terms required"}}), 400
    terms = [str(t).strip()[:120] for t in terms[:MAX_STOCK_CLIPS]]
    if not all(terms):
        return jsonify({"error": {"message": "empty term in list"}}), 400
    try:
        least = max(1, min(20, int(body.get("seconds", 4))))
    except (TypeError, ValueError):
        least = 4
    source = str(body.get("source") or "pexels")

    material, schema = mods["material"], mods["schema"]
    search = {"pexels": material.search_videos_pexels,
              "pixabay": material.search_videos_pixabay,
              "coverr": material.search_videos_coverr}.get(source)
    if search is None:
        return jsonify({"error": {"message": f"unknown source {source}"}}), 400
    try:
        material.get_api_key(f"{source}_api_keys")
    except Exception as e:
        # no key is not something the lesson should die on — it is the same
        # "try it with no token at all" path the diffusion routes take
        return jsonify({"disabled": True,
                        "why": next((ln for ln in str(e).splitlines()
                                     if ln.strip()), "no api key").strip()})

    # the board is a wide surface: portrait stock would letterbox badly
    aspect = schema.VideoAspect.landscape
    urls, errs, used = [], [], set()
    for term in terms:
        try:
            found = search(term, least, aspect) or []
        except Exception as e:
            urls.append(None); errs.append(f"{term}: {e}"); continue
        pick = next((m for m in found if m.url and m.url not in used), None)
        if pick is None:
            urls.append(None); errs.append(f"{term}: nothing found"); continue
        used.add(pick.url)
        try:
            local = material.save_video(pick.url, save_dir=_tutor_dir())
        except Exception as e:
            urls.append(None); errs.append(f"{term}: {e}"); continue
        ok = bool(local) and os.path.isfile(local) and os.path.getsize(local) > 0
        urls.append(_tutor_url(local) if ok else None)
        errs.append(None if ok else f"{term}: download failed")
    return jsonify({"videos": urls, "errors": errs, "source": source})


# -------------------------------------------- the edited action reel (MPT)
#
# /api/reel is MoneyPrinterTurbo's actual EDIT, run per chapter and played
# inside the lesson. The browser-side reel cuts live between raw stock clips;
# this composes them properly, the way its renderer composes a short:
#
#   each ACTION is spoken by the neural voice, and its clip is trimmed to
#   EXACTLY the length of that sentence — the shot lasts as long as the
#   description does, by construction, because the audio's measured duration
#   IS the shot length. The description is burned in as the subtitle of its
#   own shot (MPT's font, outline and position), the narration is muxed into
#   the file, and the shots are joined with its fade transition.
#
# The result is one mp4 per chapter that the board plays as the chapter —
# not an export, part of the lesson — with the sketch annotations still drawn
# on top by the canvas. Minutes of ffmpeg, so it is a job the page polls,
# with file-based state because gunicorn runs two workers and the poll lands
# on whichever one is free.

MAX_REEL_ACTIONS = 5
# generate_video sizes and positions its subtitle strip for the resolution
# params.video_aspect names — 1920x1080 for landscape — so the cut has to be
# encoded at exactly that, or the strip lands partly outside a smaller frame
# and dies inside the compositor with a zero-height broadcast error.
REEL_SIZE = (1920, 1080)


def _reel_state(job, **kw):
    d = os.path.join(_tutor_dir(), f"reel-{job}")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "state.json")
    try:
        with open(path, encoding="utf-8") as fh:
            state = json.load(fh)
    except (OSError, ValueError):
        state = {}
    state.update(kw)
    if kw:                       # every write stamps its time, so a job
        state["at"] = time.time()   # orphaned by a killed worker looks old
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(state, fh)
    except OSError:
        pass
    return state


def _srt_time(sec):
    ms = int(round(sec * 1000))
    return "%02d:%02d:%02d,%03d" % (ms // 3600000, ms // 60000 % 60,
                                    ms // 1000 % 60, ms % 1000)


def _fit_clip(clip, w, h):
    """Cover-fit a stock clip to the frame: scale to fill, crop the overflow
    about the centre — the same letterbox-free treatment combine_videos gives
    its materials, without stretching anything."""
    cw, ch = clip.size
    scale = max(w / cw, h / ch)
    clip = clip.resized((int(round(cw * scale)), int(round(ch * scale))))
    return clip.cropped(x_center=clip.size[0] / 2, y_center=clip.size[1] / 2,
                        width=w, height=h)


def _hold_to(clip, seconds):
    """A shot lasts as long as its sentence. Footage longer than the sentence
    is trimmed; footage shorter is looped — a 3s clip under a 7s instruction
    plays twice and a bit rather than freezing or going dark."""
    from moviepy import concatenate_videoclips
    if clip.duration >= seconds:
        return clip.subclipped(0, seconds)
    reps = [clip] * (int(seconds / clip.duration) + 1)
    return concatenate_videoclips(reps).subclipped(0, seconds)


def _t2v_prompt(text):
    """An action's instruction, rewritten as a text-to-video prompt.

    Make-A-Video's premise (arXiv:2209.14792) is exactly this app's need:
    the model learned what the world looks like from text-image pairs and how
    it MOVES from unlabeled video, so a sentence describing an action becomes
    a clip OF that action — not stock that happens to be near it. The framing
    language mirrors the paper's prompt style: subject, motion, one shot."""
    return (f"Tight macro shot of a pair of real hands: {text}. Photoreal, "
            f"natural workshop light, shallow depth of field, one continuous "
            f"take, camera almost still, no text or captions in frame.")


def _t2v_fetch(prompt, dest):
    """Generate one clip and pull it local. -> path or None, never raises."""
    try:
        url, err = _replicate(VIDEO_MODEL, {"prompt": prompt}, 420)
        if not url:
            return None
        r = requests.get(url, stream=True, timeout=120)
        if r.status_code != 200:
            return None
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(256 * 1024):
                fh.write(chunk)
        return dest if os.path.getsize(dest) else None
    except Exception:
        return None


# ---------------------------------------- perceive: look before you cut
#
# The editor used to trust search ranking blindly: first Pexels hit, cut from
# t=0, hope it shows the action. AVI's Retrieve-Perceive-Review loop
# (arXiv:2511.14446) names exactly what was missing — captions and rankings
# are "unreliable hints for location only", and an answer needs VISUAL
# confirmation. So the reel editor now perceives its footage before cutting:
# a handful of frames from each candidate clip go to a vision model, which
# scores whether the clip actually shows the action, WHERE in the clip it
# shows best (its boundary_detect), and where the subject sits in frame (its
# grounding step) — which is what lets the sketch annotations land on the
# real pixels instead of an assumed centre.

PERCEIVE_MODEL = "claude-haiku-4-5-20251001"   # cheap, fast, has eyes
PERCEIVE_MIN_SCORE = 4        # below this the clip does not show the action
PERCEIVE_CANDIDATES = 2       # clips judged per action before settling


def _clip_frames_b64(video_mod, path, n=3, width=480):
    """n JPEG frames, evenly spaced, small. -> [(t_seconds, b64), ...]"""
    import base64
    import io as _io
    from PIL import Image
    out = []
    clip = None
    try:
        clip = video_mod._open_video_clip_quietly(path, audio=False)
        d = max(clip.duration or 1.0, 0.5)
        for k in range(n):
            t = d * (k + 1) / (n + 1)
            im = Image.fromarray(clip.get_frame(t))
            im.thumbnail((width, width * 3))
            buf = _io.BytesIO()
            im.save(buf, format="JPEG", quality=70)
            out.append((round(t, 1),
                        base64.b64encode(buf.getvalue()).decode("ascii")))
    except Exception:
        return []
    finally:
        if clip is not None:
            try:
                video_mod.close_clip(clip)
            except Exception:
                pass
    return out


def _perceive_clip(video_mod, path, action_text):
    """Does this footage show this action? -> dict or None.

    {"score": 0-10, "start": best offset seconds, "subject": "the thing",
     "x": 1-50, "y": 1-50}   (grid matches the board's, x1y1 bottom-left)

    None means the judgement could not be made (no key, no frames, bad
    reply) — the caller then falls back to trusting the search, which is
    exactly what it did before this existed."""
    if not API_KEY:
        return None
    frames = _clip_frames_b64(video_mod, path)
    if not frames:
        return None
    content = []
    for t, b64 in frames:
        content.append({"type": "text", "text": f"frame at {t}s:"})
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/jpeg", "data": b64}})
    content.append({"type": "text", "text":
        f'These frames are from one stock/generated clip. The action to '
        f'illustrate is: "{action_text}".\n'
        f'Answer ONLY strict JSON, no fences:\n'
        f'{{"score": 0-10 how well this clip shows that action being '
        f'performed (0=unrelated, 10=exactly this), "start": the frame time '
        f'in seconds where it shows best, "subject": 2-4 words naming the '
        f'main visible thing, "x": 1-50, "y": 1-50 grid position of that '
        f'subject (x1y1 is bottom-left, x50y50 top-right)}}'})
    try:
        r = requests.post(ANTHROPIC_URL, timeout=60, headers={
            "x-api-key": API_KEY, "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"},
            json={"model": PERCEIVE_MODEL, "max_tokens": 200,
                  "messages": [{"role": "user", "content": content}]})
        if r.status_code != 200:
            return None
        txt = "".join(b.get("text", "") for b in r.json().get("content", [])
                      if b.get("type") == "text")
        m = re.search(r"\{[^{}]*\}", txt)
        if not m:
            return None
        d = json.loads(m.group(0))
        return {"score": max(0, min(10, int(d.get("score", 0)))),
                "start": max(0.0, float(d.get("start", 0) or 0)),
                "subject": str(d.get("subject", ""))[:60],
                "x": max(1, min(50, int(d.get("x", 25)))),
                "y": max(1, min(50, int(d.get("y", 25))))}
    except Exception:
        return None


def _reel_run(job, actions, voice, source):
    mods, _why = _mpt()
    voice_mod, material = mods["voice"], mods["material"]
    video_mod, schema = mods["video"], mods["schema"]
    out = os.path.join(_tutor_dir(), f"reel-{job}")
    opened = []
    try:
        from moviepy import (AudioFileClip, concatenate_videoclips,
                             concatenate_audioclips)
        from app.services.utils import video_effects

        # 1. speak every action FIRST: the audio durations are the edit's
        #    timeline, so nothing can be cut until the sentences are measured
        _reel_state(job, state="narrating the actions", pct=10)
        durs, audios = [], []
        for i, a in enumerate(actions):
            mp3 = os.path.join(out, f"act-{i}.mp3")
            sm = voice_mod.tts(a["text"], voice, 1.0, mp3)
            if sm is None or not os.path.isfile(mp3):
                raise RuntimeError(f"narration failed on action {i + 1}")
            durs.append(max(1.5, float(voice_mod.get_audio_duration(sm)) + 0.35))
            audios.append(mp3)

        # 2. film per action, from one of two sources:
        #    GENERATED — text-to-video diffusion, the Make-A-Video lineage:
        #    the action's own sentence becomes a clip of that action being
        #    performed. All prompts go out concurrently, because serialising
        #    30-90s generations would take the reel from one wait to five.
        #    STOCK — the Pexels search. Also the per-action fallback when a
        #    generation fails, so a dead prompt costs one shot, not the reel.
        gen = source in ("generate", "auto") and bool(REPLICATE_TOKEN)
        paths = [None] * len(actions)
        if gen:
            _reel_state(job, state="generating video of each action", pct=25)
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=len(actions)) as pool:
                paths = list(pool.map(
                    lambda ia: _t2v_fetch(_t2v_prompt(ia[1]["text"]),
                                          os.path.join(out, f"gen-{ia[0]}.mp4")),
                    enumerate(actions)))
        anchors = [None] * len(actions)
        starts = [0.0] * len(actions)
        if not all(paths):
            _reel_state(job, state="finding footage per action", pct=32)
            aspect = schema.VideoAspect.landscape
            used = set()
            for i, a in enumerate(actions):
                if paths[i]:
                    # a generated clip shows the action by construction, but
                    # perceiving it still finds the subject for annotation
                    anchors[i] = _perceive_clip(video_mod, paths[i], a["text"])
                    continue
                term = (a.get("footage") or a["text"])[:120]
                hits = []
                try:
                    hits = material.search_videos_pexels(term, 3, aspect) or []
                except Exception:
                    pass
                # RETRIEVE gives candidates; PERCEIVE looks at their frames;
                # REVIEW keeps the best that actually shows the action. The
                # search ranking is a hint, never the verdict.
                _reel_state(job, state="perceiving the footage — checking "
                                       f"clips for action {i + 1}", pct=38)
                best, best_path = None, None
                fallback = None
                seen = 0
                for m in hits:
                    if not m.url or m.url in used:
                        continue
                    try:
                        p = material.save_video(m.url, save_dir=_tutor_dir())
                    except Exception:
                        continue
                    if not (p and os.path.isfile(p)):
                        continue
                    if fallback is None:
                        fallback = (m.url, p)
                    verdict = _perceive_clip(video_mod, p, a["text"])
                    seen += 1
                    if verdict and (best is None
                                    or verdict["score"] > best["score"]):
                        best, best_path = verdict, (m.url, p)
                    if best and best["score"] >= 8:
                        break                      # visually confirmed; stop
                    if seen >= PERCEIVE_CANDIDATES:
                        break
                if best and best["score"] >= PERCEIVE_MIN_SCORE:
                    used.add(best_path[0]); paths[i] = best_path[1]
                    anchors[i] = best; starts[i] = best["start"]
                elif fallback:
                    # no candidate was confirmed: keep the search's first
                    # pick rather than a dark screen, but say so
                    used.add(fallback[0]); paths[i] = fallback[1]
        if not any(paths):
            raise RuntimeError("no footage — generation and stock both empty")

        # 3. the edit: shot i = clip i held for exactly durs[i], fading in —
        #    a missing clip inherits the previous shot's footage so the reel
        #    never goes dark mid-sentence
        _reel_state(job, state="cutting the shots to the narration", pct=55)
        w, h = REEL_SIZE
        shots, last = [], next(p for p in paths if p)
        for i, p in enumerate(paths):
            last = p or last
            src = video_mod._open_video_clip_quietly(last, audio=False)
            opened.append(src)
            seg = src
            if p and starts[i] > 0 and starts[i] < (src.duration or 0) - 1.0:
                # start where the vision pass said the action shows best —
                # its boundary_detect, aimed at our one need
                seg = src.subclipped(starts[i], src.duration)
            shot = _hold_to(_fit_clip(seg, w, h), durs[i])
            if i:                       # its fade between shots, not a hard cut
                shot = video_effects.fadein_transition(shot, 0.3)
            shots.append(shot)
        combined = concatenate_videoclips(shots)
        narr = concatenate_audioclips([AudioFileClip(p) for p in audios])
        combined_path = os.path.join(out, "combined.mp4")
        _reel_state(job, state="encoding the cut", pct=65)
        combined.write_videofile(combined_path, codec="libx264", fps=24,
                                 audio=False, preset="veryfast", threads=2,
                                 logger=None)
        narr_path = os.path.join(out, "narration.mp3")
        narr.write_audiofile(narr_path, logger=None)

        # 4. each action's DESCRIPTION is the subtitle of its own shot
        srt_path = os.path.join(out, "reel.srt")
        t = 0.0
        with open(srt_path, "w", encoding="utf-8") as fh:
            for i, a in enumerate(actions):
                fh.write("%d\n%s --> %s\n%s\n\n" % (
                    i + 1, _srt_time(t), _srt_time(t + durs[i]), a["text"]))
                t += durs[i]

        # 5. MoneyPrinterTurbo's own finishing pass: burns the subtitles in
        #    its font and outline and muxes the narration
        _reel_state(job, state="burning subtitles, muxing the voice", pct=80)
        params = schema.VideoParams(
            video_subject="action reel", video_script=" ".join(
                a["text"] for a in actions),
            video_aspect=schema.VideoAspect.landscape, voice_name=voice,
            subtitle_enabled=True, font_name=MPT_SUBTITLE["font"],
            font_size=MPT_SUBTITLE["size"],
            text_fore_color=MPT_SUBTITLE["fore"],
            stroke_color=MPT_SUBTITLE["stroke"],
            stroke_width=MPT_SUBTITLE["stroke_width"],
            subtitle_position=MPT_SUBTITLE["position"], bgm_type="")
        final = os.path.join(out, "reel.mp4")
        video_mod.generate_video(video_path=combined_path,
                                 audio_path=narr_path,
                                 subtitle_path=srt_path,
                                 output_file=final, params=params)
        if not os.path.isfile(final):
            raise RuntimeError("the encoder produced no file")
        cues, t = [], 0.0
        for i, a in enumerate(actions):
            cue = {"start": round(t, 2), "end": round(t + durs[i], 2),
                   "text": a["text"]}
            v = anchors[i]
            if v and v.get("subject"):
                cue["subject"] = v["subject"]
                cue["grid"] = [v["x"], v["y"]]
                cue["confirmed"] = v["score"] >= PERCEIVE_MIN_SCORE
            cues.append(cue)
            t += durs[i]
        _reel_state(job, state="done", pct=100,
                    video=_tutor_url(final), seconds=round(t, 2), cues=cues)
    except Exception as e:
        _reel_state(job, state="error", pct=100,
                    error=f"{type(e).__name__}: {e}")
    finally:
        for c in opened:
            try:
                video_mod.close_clip(c)
            except Exception:
                pass


@app.post("/api/reel")
def reel_start():
    """One chapter's actions -> one MPT-edited clip, cut to the descriptions.

    {"actions": [{"text": "Undo the quick link", "footage": "..."}, ...],
     "lang": "en"} -> {"job": id, "poll": "/api/reel/status?job=id"}

    Ends {"state": "done", "video": same-origin url, "seconds": ...,
    "cues": [{start, end, text}, ...]} — the cue list is the edit decision
    list, so the page knows exactly when each action is on screen."""
    mods, why = _mpt()
    if not mods:
        return jsonify({"disabled": True, "why": why})
    body = request.get_json(silent=True) or {}
    raw = body.get("actions")
    if not isinstance(raw, list) or not raw:
        return jsonify({"error": {"message": "actions required"}}), 400
    actions = []
    for a in raw[:MAX_REEL_ACTIONS]:
        if isinstance(a, str):
            a = {"text": a}
        if isinstance(a, dict) and str(a.get("text") or "").strip():
            actions.append({"text": str(a["text"]).strip()[:300],
                            "footage": str(a.get("footage") or "")[:120]})
    if not actions:
        return jsonify({"error": {"message": "no action had text"}}), 400
    # "generate" = text-to-video diffusion of each action (Make-A-Video's
    # premise, one paid generation per shot); "stock" = the Pexels search;
    # "auto" = generate when a token is configured, stock otherwise, and
    # stock per-shot whenever a generation fails.
    source = str(body.get("source") or "auto")
    if source not in ("auto", "generate", "stock"):
        return jsonify({"error": {"message": f"unknown source {source}"}}), 400
    have_stock = True
    try:
        mods["material"].get_api_key("pexels_api_keys")
    except Exception as e:
        have_stock = False
        stock_why = next((ln for ln in str(e).splitlines()
                          if ln.strip()), "no api key").strip()
    if source == "generate" and not REPLICATE_TOKEN:
        return jsonify({"disabled": True, "why": "no REPLICATE_API_TOKEN"})
    if not have_stock and not (REPLICATE_TOKEN and source in ("auto",
                                                             "generate")):
        return jsonify({"disabled": True, "why": stock_why})
    lang = str(body.get("lang") or "en")
    voice = str(body.get("voice") or MPT_VOICES.get(lang) or MPT_VOICES["en"])

    job = hashlib.sha1(json.dumps([voice, source, actions], sort_keys=True)
                       .encode("utf-8")).hexdigest()[:16]
    final = os.path.join(_tutor_dir(), f"reel-{job}", "reel.mp4")
    state_now = _reel_state(job)
    if os.path.isfile(final) and state_now.get("state") == "done":
        return jsonify({"job": job, "cached": True,
                        "poll": f"/api/reel/status?job={job}"})
    if state_now.get("state") in ("narrating the actions",
                                  "generating video of each action",
                                  "finding footage per action",
                                  "cutting the shots to the narration",
                                  "encoding the cut",
                                  "burning subtitles, muxing the voice"):
        # the same chapter prefetched twice must not spawn a second ffmpeg —
        # but an in-progress state that has not been touched for 15 minutes
        # is a worker that died mid-render (deploy, hard kill): the job id
        # is deterministic, so without this check that chapter's reel would
        # be wedged forever. Fall through and restart it instead.
        if time.time() - float(state_now.get("at") or 0) < 900:
            return jsonify({"job": job, "poll": f"/api/reel/status?job={job}"})
    _reel_state(job, state="queued", pct=0, error=None, video=None)
    threading.Thread(target=_reel_run, daemon=True,
                     args=(job, actions, voice, source)).start()
    return jsonify({"job": job, "poll": f"/api/reel/status?job={job}"})


@app.get("/api/reel/status")
def reel_status():
    job = re.sub(r"[^0-9a-f]", "", request.args.get("job", ""))[:16]
    if not job:
        return jsonify({"error": {"message": "job required"}}), 400
    path = os.path.join(_tutor_dir(), f"reel-{job}", "state.json")
    if not os.path.isfile(path):
        return jsonify({"error": {"message": "unknown job"}}), 404
    try:
        with open(path, encoding="utf-8") as fh:
            return jsonify(json.load(fh))
    except (OSError, ValueError) as e:
        return jsonify({"error": {"message": str(e)}}), 500


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
    # PRECISE verification: the page can send back the EXACT solved clip a
    # chapter just demonstrated, and the reference is reduced from THAT
    # motion — the student is graded against the very frames they watched,
    # not a fresh synthesis that might phrase the movement differently.
    sent = p.get("clip")
    if isinstance(sent, dict) and isinstance(sent.get("frames"), list) \
            and len(sent.get("frames") or []) >= 2:
        try:
            ref = coach.reference(sent)
            ref["precise"] = True
            return jsonify({"clip": sent, "reference": ref})
        except Exception:
            pass                 # malformed clip: fall back to synthesis
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


# ------------------------------------------------------------- quiz
#
# The knowledge loop's mirror of camera verification: after a chapter is
# WATCHED, the student is asked whether it was LEARNED. Questions are
# grounded in the hivemind — the book's, the video's, the stream's own
# steps — so the quiz tests what was actually taught, not model trivia.

QUIZ_SYS = """You write ONE check question for a student who just watched a chapter of an interactive video lesson. Respond with ONLY strict JSON, no markdown fences:
{"question":"one short spoken-friendly question","choices":["...","...","...","..."],"answer":0,"why":"one sentence explaining the correct answer","topic":"2-4 words naming the knowledge point"}
Rules:
- Test the TAUGHT content. When source material is provided, the question and its correct answer must come from IT, nearly verbatim — never from general knowledge that contradicts it.
- Practical, applied phrasing ("what do you do when...", "which muscles drive..."), not trivia.
- Exactly 4 choices, one clearly correct, the distractors plausible but wrong.
- Everything short enough to be read aloud."""


@app.post("/api/quiz")
def quiz_gen():
    """{"title","narration","topic"} -> one grounded multiple-choice check.

    -> {"question","choices":[...],"answer":idx,"why","topic","grounded"}"""
    if not API_KEY:
        return jsonify({"disabled": True})
    p = request.get_json(silent=True) or {}
    title = str(p.get("title") or "")[:160]
    narration = str(p.get("narration") or "")[:600]
    topic = str(p.get("topic") or "")[:160]
    ctx = ""
    try:
        ctx = HM.retrieve(f"{title} {narration}"[:300], k=2) or ""
    except Exception:
        pass
    try:
        r = requests.post(ANTHROPIC_URL, timeout=60, headers={
            "x-api-key": API_KEY, "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"},
            json={"model": "claude-sonnet-4-6", "max_tokens": 700,
                  "system": QUIZ_SYS,
                  "messages": [{"role": "user", "content":
                      f"Lesson topic: {topic}\nChapter: {title}\n"
                      f"What was said: {narration}"
                      + (f"\n\nSource material:\n{ctx[:2000]}" if ctx else "")}]})
        raw = "".join(b.get("text", "") for b in r.json().get("content", [])
                      if b.get("type") == "text")
        q = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
        ch = [str(c)[:160] for c in (q.get("choices") or [])][:4]
        ai = int(q.get("answer") or 0)
        if len(ch) < 2 or not (0 <= ai < len(ch)) or not q.get("question"):
            raise ValueError("malformed question")
        return jsonify({"question": str(q["question"])[:300], "choices": ch,
                        "answer": ai, "why": str(q.get("why") or "")[:300],
                        "topic": str(q.get("topic") or title)[:80],
                        "grounded": bool(ctx)})
    except Exception as e:
        return jsonify({"error": {"message": f"no question: {e}"}}), 502


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


# ------------------------------------------------------- teach from a book
#
# A how-to book is a finished curriculum: someone already sequenced the
# skills, wrote the exercises as imperatives and photographed the hand
# positions. /api/book turns that directly into the lesson — the book
# replaces the director, not the player. Its exercises become action-reel
# and hand-rig chapters, its diagrams the rare "anim" chapter, and its
# procedures are ingested into the hivemind so an interruption is answered
# from what the BOOK says rather than from the model's memory.
#
# The same conversion the Mine button does for a YouTube transcript, done
# for the older medium.

MAX_BOOK_PAGES = 100
MAX_BOOK_CHARS = 55_000        # what one structuring call can actually hold

BOOK_SYS = """You convert instructional books into interactive ANIMATED video lessons — the kind a student watches, interrupts mid-play and gets answered from the book itself. You receive the extracted text of a how-to book. Respond with ONLY strict JSON, no markdown fences, shaped:
{"title": "short course title", "topic": "what is being taught, one phrase",
 "chapters": [ 8-12 chapters, each one of:
  {"narration":"2-3 conversational spoken sentences teaching the point in the book's own terms","title":"short noun phrase","mode":"actions","seconds":14,"actions":[{"text":"One imperative action STARTING WITH THE VERB, from the book","footage":"2-5 words naming the filmable thing"} , 2-4 actions]}
  {"narration":"...","title":"...","mode":"hand","seconds":10,"hand_steps":["Imperative finger/hand action from the book", 2-4 of them]}
  {"narration":"...","title":"...","mode":"skeleton","seconds":12,"motion_steps":["Whole-body imperative from the book", 2-4],"muscles":["glutes","quads"]}
  {"narration":"...","title":"...","mode":"hybrid","seconds":12,"video_prompt":"STYLE sentence + one sentence naming this exercise","motion_steps":["...", 2-4],"muscles":["..."]}
  {"narration":"...","title":"...","mode":"compose","seconds":12,"parts":[{"kind":"skeleton","title":"posture","steps":["..."],"muscles":["..."]},{"kind":"hand","title":"grip","steps":["..."]}]}
  {"narration":"...","title":"...","mode":"world","seconds":15,"actors":[{"kind":"skeleton","title":"the body","steps":["..."],"muscles":["..."]},{"kind":"hand","title":"grip","steps":["..."]}]}
  {"narration":"...","title":"...","mode":"anim","seconds":10,"steps":[{"beat":"...","els":[{"type":"box","x":20,"y":30,"w":12,"h":8,"label":"...","color":"blue","in":"pop"}]}]}
 ],
 "howtos": [ 4-8 procedures worth remembering, each {"title":"...","steps":["imperative step", ...]} ]}
Rules:
- FOLLOW THE BOOK's own teaching order and use ITS instructions, not your general knowledge. Quote its imperatives nearly verbatim in actions/hand_steps/motion_steps, keep its names for things, and stay terminologically consistent from chapter to chapter — the student can interrupt at any moment and the answer must match what the screen said.
- USE THE WHOLE PALETTE — a good lesson MIXES modes, never one mode throughout:
  "actions" for procedures best shown as real film, with concrete filmable footage terms.
  "hand" for finger technique (grips, fretting, picking, knots).
  "hybrid" is the flagship for EXERCISE — a diffusion-generated 3D animation, the anatomy-video look. For a fitness/yoga/dance/sports book make one hybrid chapter PER exercise. Its "video_prompt" MUST begin with this EXACT style sentence, identical in every chapter so the whole lesson looks like one production: "Smooth 3D-rendered fitness anatomy animation, matte white 3D figure with sculpted muscle definition, the working muscles glowing red-orange, modern gym with polished wooden floor and a large ocean-view window, soft studio lighting, slow orbiting camera, seamless loop, no text." — followed by ONE sentence naming THIS exercise, its equipment and the movement (e.g. "The figure performs barbell hip thrusts, shoulders on a flat bench, driving the hips upward."). ALWAYS give that same chapter "motion_steps" (its reps and form cues from the book) and "muscles" (lowercase names: glutes, quads, hamstrings, calves, core, back, chest, shoulders, biceps, triceps, neck) — when video generation is off or fails, the player performs the exercise itself as an anatomy figure with those muscle groups burning red-orange, so the chapter works either way.
  "skeleton" for posture and whole-body movement that is not a gym exercise; it too may carry "muscles".
  A NON-fitness book may use hybrid chapters the same way: FIRST read what kind of book this is, then invent ONE style sentence that fits ITS world — a cookbook gets overhead kitchen cinematography on a wooden counter with warm daylight; a woodworking manual gets a workbench close-up in warm shop light; a guitar primer gets a close-up of hands on the fretboard in soft window light; a gardening guide gets bright outdoor macro footage — and reuse that sentence VERBATIM as the opening of every video_prompt in the lesson, then one sentence for the chapter's specific action. One book, one visual production; the format of the video always follows the content's own context.
  "compose" when body and fingers matter at the same time — the panes play side by side.
  "world" for at least one chapter whenever a body, a tool and an object must be in the right places relative to each other — one 3D room at real scale.
  "anim" ONLY for genuinely abstract structure (a tuning diagram, a rep scheme chart) — at most 1.
- Every chapter may also carry "footage": 2-5 words naming real stock film to play behind the rig; use it on most chapters.
- Coordinates in anim els: 50x50 grid, x1y1 bottom-left, keep 4-46.
- "howtos" are the book's procedures as numbered-step recipes so a knowledge graph can answer questions about them later — this is what makes the lesson conversational, so be generous and detailed: every exercise, procedure, form cue and safety warning the book gives, steps imperative, one action each.
- Everything spoken ("narration") is conversational, as a tutor would say it aloud."""


MAX_BOOK_CHUNKS = 4            # paired extraction runs per chunk — bound it
BOOK_CHUNK_CHARS = 12_000      # roughly what one MINE_SYS call reads well


def _book_chunks(text):
    """Split extracted book text into procedure-sized chunks on page marks,
    so the detailed step extraction reads the WHOLE book a piece at a time
    instead of skimming one oversized prompt."""
    pages = re.split(r"(?=\[page \d+\])", text)
    chunks, cur = [], ""
    for pg in pages:
        if cur and len(cur) + len(pg) > BOOK_CHUNK_CHARS:
            chunks.append(cur)
            cur = pg
        else:
            cur += pg
    if cur.strip():
        chunks.append(cur)
    return chunks[:MAX_BOOK_CHUNKS]


def _book_detailed_steps(name, text, book_title):
    """The Mine pipeline, run over the book — with self-experimentation.

    Each chunk goes through pairing.extract_pair: a fast reader and an
    adversarial evidence-first reader on a stronger model extract the steps
    INDEPENDENTLY, reconcile() marks every step agreed/contested, and
    ingest_paired writes the result into the hivemind with confirmations on
    the corroborated steps and review-queue corrections on the contested
    ones. The corroborated detailed steps come back as playable chapters —
    the same hand/skeleton/compose walkthrough a mined YouTube video gets.

    -> (chapters, summary_dict). Never raises; a failed chunk is skipped.
    """
    chapters, agg = [], {"agreed": 0, "only_a": 0, "only_b": 0}
    steps_total, reader, failed = 0, "", 0
    for i, chunk in enumerate(_book_chunks(text)):
        header = (f"Book: {book_title or name} (part {i + 1})\n"
                  "This is a WRITTEN BOOK, not a video transcript. "
                  "Omit the \"at\" field entirely. Extract EVERY procedure "
                  "this part actually states, step by step, in its order.")
        try:
            a, b, errs = pairing.extract_pair(
                header, f"Book text:\n{chunk}", API_KEY)
            doc, verdicts = pairing.reconcile(a, b)
            doc = openclaw._finish(doc, book_title or name,
                                   {"url": "", "source": f"book: {name}"})
            for s in doc["steps"]:
                s["at"] = ""               # a book has nothing to seek to
            reader = ("deepseek" if errs.get("provider") == "deepseek"
                      else pairing.PAIR_B_MODEL)
            res = pairing.ingest_paired(HM, doc, verdicts, None,
                                        author=reader)
            for k in agg:
                agg[k] += res["verdicts"].get(k, 0)
            steps_total += len(doc["steps"])
            chapters += openclaw.doc_to_lesson(doc)
        except Exception:
            failed += 1
            continue
    if steps_total:
        HM.save(GRAPH_PATH)
    total = max(sum(agg.values()), 1)
    return chapters, {"steps": steps_total,
                      "agreed": agg["agreed"],
                      "contested": agg["only_a"] + agg["only_b"],
                      "agreement": round(agg["agreed"] / total, 3),
                      "second_reader": reader, "failed_chunks": failed}


def _book_text(data):
    """PDF bytes -> (text, pages_read). Text-layer only — a scanned book with
    no text layer comes back empty and is reported as such, not OCR'd."""
    import io as _io
    from pypdf import PdfReader
    rd = PdfReader(_io.BytesIO(data))
    out, n = [], 0
    for page in rd.pages[:MAX_BOOK_PAGES]:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t.strip():
            n += 1
            out.append(f"[page {n}]\n{t.strip()}")
        if sum(len(x) for x in out) > MAX_BOOK_CHARS:
            break
    return "\n\n".join(out)[:MAX_BOOK_CHARS], n


@app.post("/api/book")
def book_lesson():
    """Upload a PDF book -> a playable lesson grounded in that book.

    multipart/form-data with "file", or a raw application/pdf body.
    -> {"title", "topic", "chapters": [director-format chapters],
        "ingested": how many of its procedures now live in the hivemind,
        "pages": pages read}

    The chapters drop straight into the player; the howtos go into the graph,
    so pressing Ask mid-lesson retrieves the book's own steps."""
    if not API_KEY:
        return jsonify({"disabled": True, "why": "ANTHROPIC_API_KEY not set"})
    f = request.files.get("file")
    data = f.read() if f else request.get_data()
    name = (f.filename if f else "book")[:120]
    if not data or len(data) < 800:
        return jsonify({"error": {"message": "no PDF received"}}), 400
    if len(data) > 40 * 1024 * 1024:
        return jsonify({"error": {"message": "PDF over 40MB"}}), 400
    try:
        text, pages = _book_text(data)
    except Exception as e:
        return jsonify({"error": {"message": f"could not read the PDF: {e}"}}), 400
    if len(text) < 400:
        return jsonify({"error": {"message":
            "no readable text — this PDF looks scanned (images only)"}}), 400

    try:
        r = requests.post(ANTHROPIC_URL, timeout=240, headers={
            "x-api-key": API_KEY, "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"},
            json={"model": "claude-sonnet-4-6", "max_tokens": 9000,
                  "system": BOOK_SYS,
                  "messages": [{"role": "user", "content":
                      f"Book file: {name}\n\n{text}"}]})
        if r.status_code != 200:
            return jsonify({"error": {"message":
                f"structuring failed: HTTP {r.status_code}"}}), 502
        raw = "".join(b.get("text", "") for b in r.json().get("content", [])
                      if b.get("type") == "text")
        plan = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
    except Exception as e:
        return jsonify({"error": {"message": f"structuring failed: {e}"}}), 502

    chapters = [c for c in (plan.get("chapters") or [])
                if isinstance(c, dict) and c.get("narration")][:12]
    if not chapters:
        return jsonify({"error": {"message":
            "the book yielded no teachable chapters"}}), 502

    # the book's procedures become graph knowledge, so interrupts answer
    # from the book — with its title as the cited source
    ingested = 0
    for h in (plan.get("howtos") or [])[:12]:
        try:
            steps = [str(s).strip() for s in (h.get("steps") or []) if str(s).strip()]
            if len(steps) < 2:
                continue
            body = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
            doc = parse_plain_howto(str(h.get("title") or "untitled")[:120],
                                    body, source=f"book: {name}")
            if doc["steps"]:
                HM.ingest(doc)
                ingested += 1
        except Exception:
            continue
    if ingested:
        HM.save(GRAPH_PATH)

    # the Mine treatment, book edition: paired readers extract EVERY stated
    # procedure step by step, and the corroborated ones become a detailed
    # walkthrough appended after the course chapters — same conversation
    # video, finer grain, with the disagreements queued for review
    detail_ch, detailed = _book_detailed_steps(
        name, text, str(plan.get("title") or ""))
    if detail_ch:
        chapters.append({
            "mode": "anim", "seconds": 8, "title": "step by step",
            "narration": "Now the fine grain — every step the book actually "
                         "states, checked by two independent readers.",
            "steps": [{"beat": "the walkthrough begins", "els": [
                {"type": "text", "x": 25, "y": 30, "text": "step by step",
                 "color": "blue", "in": "fade"},
                {"type": "box", "x": 25, "y": 20, "w": 30, "h": 8,
                 "label": f"{detailed['steps']} steps, "
                          f"{detailed['agreed']} corroborated",
                 "color": "green", "in": "pop"}]}]})
        chapters += detail_ch[:16]

    return jsonify({"title": str(plan.get("title") or name)[:160],
                    "topic": str(plan.get("topic") or "")[:200],
                    "chapters": chapters, "ingested": ingested,
                    "detailed": detailed, "pages": pages})


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


# ------------------------------------------------- livestream -> tutorial
#
# A livestream is a how-to being written in real time. Following one polls
# its transcript, mines ONLY the tail that arrived since the last poll, and
# hands back fresh chapters — so the stream turns into instruction while it
# is still streaming, and its procedures join the hivemind as they are said.
# One cursor per video id, in memory: a restart just re-mines from the top.

LIVE_CURSORS = {}
MIN_LIVE_NEW_SECS = 45          # do not wake Claude for ten seconds of talk
MIN_LIVE_NEW_CHARS = 250


@app.post("/api/live")
def live_follow():
    """Poll a livestream (or any still-growing video) as a lesson.

    {"url": "https://youtube.com/watch?v=...", "reset": true?}
    -> {"waiting": true, "buffered": secs}   not enough new material yet
    -> {"title", "chapters": [...], "cursor": secs, "steps": n}  the new tail
    """
    if not API_KEY:
        return jsonify({"error": {"message":
            "Server has no ANTHROPIC_API_KEY set — export it and restart"}}), 500
    p = request.get_json(silent=True) or {}
    url = str(p.get("url") or "")[:300]
    if not url:
        return jsonify({"error": {"message": "url required"}}), 400
    try:
        vid, title, rows = openclaw.fetch_transcript(url)
    except (openclaw.MineError, requests.RequestException) as e:
        # a stream whose captions have not started yet — or a transient
        # network hiccup on a poll loop — is "not yet", never a 500
        return jsonify({"waiting": True, "buffered": 0,
                        "note": f"no transcript yet ({e})"}), 200
    key = vid or url
    if p.get("reset"):
        LIVE_CURSORS.pop(key, None)
    # The client echoes back the cursor it last received, which makes the
    # endpoint effectively stateless: under gunicorn -w 2 each worker has
    # its own LIVE_CURSORS, and without the echo a poll landing on the
    # other worker would re-mine (and re-bill) the same tail. The higher
    # of the two wins; "reset" trusts the client's explicit restart.
    client_cur = p.get("cursor")
    cur = float(LIVE_CURSORS.get(key, 0.0))
    if isinstance(client_cur, (int, float)) and not p.get("reset"):
        cur = max(cur, float(client_cur))
    if len(LIVE_CURSORS) > 200:              # never grows without bound
        LIVE_CURSORS.pop(next(iter(LIVE_CURSORS)))
    # rows are (start_seconds, text) tuples — see openclaw.fetch_transcript
    fresh = [(float(s), str(t)) for s, t in rows if float(s) >= cur]
    end = max((s for s, _ in fresh), default=cur)
    text = " ".join(t for _, t in fresh)
    if end - cur < MIN_LIVE_NEW_SECS or len(text) < MIN_LIVE_NEW_CHARS:
        return jsonify({"waiting": True, "cursor": cur,
                        "buffered": round(max(0.0, end - cur))})
    try:
        doc = openclaw._extract(
            API_KEY,
            f"Video title: {title or '(live stream)'}\n"
            "This is the LATEST SLICE of a livestream still in progress; "
            "earlier parts were already taught as their own chapters. "
            "Extract only what THIS slice teaches, in its own order.",
            "Transcript slice:\n" + openclaw.transcript_text(fresh),
            openclaw.EXTRACT_MODEL)
        doc = openclaw._finish(doc, title, {
            "video_id": vid, "video_title": title,
            "url": f"https://www.youtube.com/watch?v={vid}" if vid else url,
            "source": f"live:{vid or url}"})
    except openclaw.MineError as e:
        return jsonify({"waiting": True, "cursor": cur,
                        "buffered": round(end - cur),
                        "note": f"slice had no teachable steps ({e})"}), 200
    except Exception as e:
        return jsonify({"error": {"message": f"live mining failed: {e}"}}), 502
    chapters = openclaw.doc_to_lesson(doc)
    if doc.get("steps"):
        try:
            HM.ingest(openclaw.hivemind_doc(doc))
            HM.save(GRAPH_PATH)
        except Exception:
            pass
    LIVE_CURSORS[key] = end + 0.5      # the last mined row must not re-mine
    return jsonify({"title": title or "live lesson", "chapters": chapters,
                    "cursor": LIVE_CURSORS[key],
                    "steps": len(doc.get("steps") or [])})


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
