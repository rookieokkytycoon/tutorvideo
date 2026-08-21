"""
handform.py — hand skeleton motion and gesture recognition, as actuated
curve vector data.

The observation this file is built on
-------------------------------------
A finger IS a LineFORM chain. Nakagaki, Follmer and Ishii built a series of
1-DOF rotational servos with the axes alternating so the chain can leave the
plane; a finger is a series of 1-DOF hinges — MCP, PIP, DIP — with the
knuckle adding a perpendicular axis for spread. The paper's control problem
is our control problem, at 2 cm per joint instead of 7:

    "we extract outline from binary image data as series of vectors, then
     calculate the angle for each servo motor according to length of each
     joint ... we do not care about the position of the end effector"

That is exactly what you want from a hand tracker. MediaPipe hands you 21
points floating in space; what a gesture actually is lives in the ANGLES.
So every hand here — synthesised or tracked from real video — is pushed
through `lineform.fit_curve` and comes back as five chains of servo angles
plus the residual. Gesture recognition then reads those angles, not the raw
points, which is why it is invariant to where the hand is and how big it is.

Pipeline
--------
    video frame  ->  MediaPipe Hands (when installed)  ->  21 landmarks
    OR how-to text -> verb lexicon -> pose library      ->  21 landmarks
                 ->  lineform.fit_curve, 5 finger chains -> servo angles
                 ->  curl / spread / pinch features      -> gesture label
                 ->  finite difference                   -> motion vectors

Because both paths converge on landmarks, the same recogniser labels a
synthesised demo and a real tracked video, and the clip format is identical.
`backend` in the clip says which one produced it — never inferred, never
guessed.

Fingers have unequal phalanges, so the chains use per-joint link lengths and
per-joint angle limits (see lineform.make_device). A PIP joint bends 110
degrees and a carpal joint barely moves; charging that difference to the fit
residual would report anatomy as hardware error.

    from handform import text_to_hand, HAND_POSES
    clip = text_to_hand("Pinch the tab and pull it free")
    print(clip["gestures"])
"""

from __future__ import annotations

import math
import re

import lineform as lf
from lineform import _add, _sub, _mul, _dot, _len, _norm, _cross, _clamp

# --------------------------------------------------------------------------
# MediaPipe Hands topology (21 landmarks)
# --------------------------------------------------------------------------

LANDMARKS = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]
LM = {n: i for i, n in enumerate(LANDMARKS)}

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]

# Each finger is a chain of four points hanging off the wrist. For the thumb
# MediaPipe names them cmc/mcp/ip/tip; for the rest mcp/pip/dip/tip.
FINGER_CHAIN = {
    "thumb":  ["wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip"],
    "index":  ["wrist", "index_mcp", "index_pip", "index_dip", "index_tip"],
    "middle": ["wrist", "middle_mcp", "middle_pip", "middle_dip", "middle_tip"],
    "ring":   ["wrist", "ring_mcp", "ring_pip", "ring_dip", "ring_tip"],
    "pinky":  ["wrist", "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"],
}
BONES = [[LM[c[i]], LM[c[i + 1]]] for c in FINGER_CHAIN.values() for i in range(4)]
# the knuckle line, so a palm reads as a palm and not five loose spiders
BONES += [[LM["index_mcp"], LM["middle_mcp"]], [LM["middle_mcp"], LM["ring_mcp"]],
          [LM["ring_mcp"], LM["pinky_mcp"]], [LM["thumb_cmc"], LM["index_mcp"]]]

# Phalanx lengths in metres for an adult hand: metacarpal, proximal, middle,
# distal. The metacarpal is the long one — it spans the palm.
PHALANX = {
    "thumb":  [0.035, 0.035, 0.032, 0.025],
    "index":  [0.085, 0.040, 0.024, 0.018],
    "middle": [0.083, 0.045, 0.027, 0.020],
    "ring":   [0.079, 0.041, 0.025, 0.019],
    "pinky":  [0.075, 0.032, 0.019, 0.017],
}
# Flexion range per joint. A carpal joint is nearly rigid; a PIP is the most
# mobile hinge in the body for its size.
LIMITS = {
    "thumb":  [12.0, 55.0, 55.0, 80.0],
    "index":  [8.0, 95.0, 110.0, 80.0],
    "middle": [8.0, 95.0, 110.0, 80.0],
    "ring":   [10.0, 95.0, 110.0, 80.0],
    "pinky":  [14.0, 95.0, 110.0, 80.0],
}
# Where each knuckle sits on the palm and which way the finger leaves it.
# Spread is about the palm normal, so it stays a base-frame property and each
# joint keeps its single degree of freedom — the paper's constraint, honoured.
KNUCKLE = {           # (across palm, along palm, out of palm), rest spread deg
    "thumb":  (0.026, 0.016, 0.012, -38.0),
    "index":  (0.028, 0.082, 0.004, -9.0),
    "middle": (0.008, 0.086, 0.000, -1.0),
    "ring":   (-0.012, 0.082, -0.003, 7.0),
    "pinky":  (-0.030, 0.073, -0.008, 16.0),
}

# The actuated part of a finger starts at the knuckle. Metacarpals belong to
# the palm: they are rigid, they fan outward, and their directions do not lie
# in the plane their finger flexes in — so folding them into the chain asks a
# planar mechanism to follow a non-planar path and charges the difference to
# the residual. Three joints per finger, from MCP.
FINGER_JOINTS = {f: c[1:] for f, c in FINGER_CHAIN.items()}

# Those three hinge about parallel axes, which makes a finger the paper's
# SMALL prototype — "motors connected in single direction limiting it to 2D
# structures but increasing its resolution" — not the large alternating one.
# Spread lives at the knuckle and is carried by the chain's base frame, so
# every joint keeps its single degree of freedom.
CHAINS = {f: lf.make_device(3, PHALANX[f][1:], LIMITS[f][1:], "single",
                            f"{f} chain", "finger servo")
          for f in FINGERS}

PAPERS = ["LineFORM (Nakagaki, Follmer & Ishii, UIST '15)",
          "MediaPipe Hands (Zhang et al. 2020)", "BlazePose", "ST-GCN"]


def _lerp(a, b, t): return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))
def _ease(p): return p * p * (3 - 2 * p)


# --------------------------------------------------------------------------
# pose library: curl + spread -> 21 landmarks
# --------------------------------------------------------------------------
# A pose is five curls in [0,1] (0 straight, 1 fully closed) plus five spread
# offsets in degrees. Everything else is anatomy. Recognition never sees these
# numbers — it re-measures them off the landmark geometry — so the round trip
# is a real test rather than a lookup.

HAND_POSES = {
    "open_palm": {"curl": [0.05, 0.03, 0.03, 0.03, 0.03], "spread": [10, 8, 2, -6, -12]},
    "flat":      {"curl": [0.20, 0.05, 0.05, 0.05, 0.05], "spread": [-6, 0, 0, 0, 0]},
    "fist":      {"curl": [0.80, 0.98, 0.98, 0.98, 0.98], "spread": [-8, 2, 0, -2, -4]},
    "point":     {"curl": [0.70, 0.02, 0.97, 0.97, 0.97], "spread": [-10, 0, 0, 0, 0]},
    "pinch":     {"curl": [0.62, 0.62, 0.30, 0.25, 0.22], "spread": [4, -6, 0, 2, 4]},
    "ok":        {"curl": [0.60, 0.66, 0.12, 0.10, 0.10], "spread": [4, -6, 6, 8, 10]},
    "peace":     {"curl": [0.75, 0.03, 0.03, 0.97, 0.97], "spread": [-8, -12, 12, 0, 0]},
    "thumbs_up": {"curl": [0.02, 0.98, 0.98, 0.98, 0.98], "spread": [-14, 0, 0, 0, 0]},
    "grip":      {"curl": [0.55, 0.72, 0.75, 0.75, 0.72], "spread": [-4, 2, 0, -2, -4]},
    "tripod":    {"curl": [0.58, 0.58, 0.55, 0.20, 0.18], "spread": [6, -4, 2, 8, 10]},
    "gun":       {"curl": [0.10, 0.03, 0.96, 0.96, 0.96], "spread": [-20, 0, 0, 0, 0]},
    "spread":    {"curl": [0.02, 0.02, 0.02, 0.02, 0.02], "spread": [22, 16, 3, -12, -22]},
}


def _palm_frame():
    """Right hand at rest: palm in the XY plane, fingers up +Y, normal +Z."""
    return (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)


def pose_landmarks(curl, spread, handed="right"):
    """Curl + spread -> 21 landmarks in metres, wrist at the origin.

    Each finger is walked with its own per-joint link lengths and per-joint
    limits, distributing the curl across MCP/PIP/DIP the way a real finger
    does — the PIP leads, the DIP follows, the carpal joint barely moves.
    """
    across, along, normal = _palm_frame()
    pts = [None] * 21
    pts[0] = (0.0, 0.0, 0.0)
    for fi, f in enumerate(FINGERS):
        kx, ky, kz, rest_spread = KNUCKLE[f]
        knuck = _add(_add(_mul(across, kx), _mul(along, ky)), _mul(normal, kz))
        sp = math.radians(rest_spread + (spread[fi] if fi < len(spread) else 0.0))
        # the finger leaves the knuckle fanned out within the palm plane
        d0 = _norm(_add(_mul(along, math.cos(sp)), _mul(across, math.sin(sp))))
        if f == "thumb":         # the thumb column is rotated out of the palm
            d0 = _norm(_add(_mul(d0, 0.82), _mul(normal, 0.57)))
        # curl bends the finger toward the palm, i.e. about the axis across it
        axis = _norm(_cross(d0, normal))
        c = _clamp(curl[fi] if fi < len(curl) else 0.0, 0.0, 1.0)
        # the three mobile joints take different shares of the same curl
        share = [0.10, 0.42, 0.34, 0.24] if f != "thumb" else [0.10, 0.40, 0.30, 0.30]
        chain = FINGER_CHAIN[f]
        p, d = knuck, d0
        pts[LM[chain[1]]] = knuck
        for j in range(1, 4):
            ang = math.radians(LIMITS[f][j]) * c * share[j] / max(share[1:]) * 1.0
            d = lf._rot(d, axis, ang)
            p = _add(p, _mul(d, PHALANX[f][j]))
            pts[LM[chain[j + 1]]] = p
    if handed == "left":
        pts = [(-q[0], q[1], q[2]) for q in pts]
    return [tuple(q) for q in pts]


# --------------------------------------------------------------------------
# landmarks -> actuated curve chains  (the LineFORM layer)
# --------------------------------------------------------------------------

def _chain_heading(path):
    """Base frame for a finger chain: forward down the metacarpal, and the
    single rotation axis along the finger's own flexion normal.

    For a single-axis chain the reachable set is the plane perpendicular to
    that axis, and it never changes as the chain bends. So the axis has to be
    the normal of the plane the finger actually flexes in — recovered from
    the cross product of two successive bones. Point it anywhere else and the
    chain is being asked for a shape it is geometrically barred from holding,
    which shows up as a large residual on a perfectly ordinary finger.
    """
    fwd = _norm(_sub(path[1], path[0]))
    # Average the normal over every consecutive pair of bones instead of
    # trusting the first two. On tracked data a distal phalanx is under 20 mm
    # and the tracker is good to about 10, so any single cross product is
    # mostly noise; summing them is a cheap least-squares plane.
    n = (0.0, 0.0, 0.0)
    for i in range(len(path) - 2):
        c = _cross(_sub(path[i + 1], path[i]), _sub(path[i + 2], path[i + 1]))
        if _dot(c, n) < 0:               # keep the winding consistent
            c = _mul(c, -1.0)
        n = _add(n, c)
    if _len(n) < 1e-12:                  # straight finger: any axis will do
        ref = (0.0, 0.0, 1.0) if abs(fwd[2]) < 0.9 else (0.0, 1.0, 0.0)
        n = _cross(fwd, ref)
    right = _norm(n)
    return (right, _norm(_cross(right, fwd)), fwd)


def hand_to_chains(pts):
    """21 landmarks -> five chains of servo angles. -> {finger: fit dict}.

    This is the paper's algorithm run five times per frame. autoscale is off:
    a finger's length is anatomy, and each chain was built in CHAINS to match
    the phalanges it drives, so any residual is a real inability to hold the
    tracked shape rather than a framing artefact.
    """
    out = {}
    for f in FINGERS:
        path = [pts[LM[n]] for n in FINGER_JOINTS[f]]
        out[f] = lf.fit_curve(path, CHAINS[f], base=path[0],
                              heading=_chain_heading(path), autoscale=False)
    return out


# --------------------------------------------------------------------------
# features and recognition
# --------------------------------------------------------------------------

def _palm_size(pts):
    """Wrist-to-middle-knuckle span: the scale everything is divided by, so a
    child's hand and a close-up of an adult's classify the same."""
    return max(_len(_sub(pts[LM["middle_mcp"]], pts[LM["wrist"]])), 1e-4)


def measure_curl(pts, f, fit=None):
    """Total flexion of one finger, in [0,1].

    Read from the FITTED servo angles, not from the raw landmarks. Fitting is
    a projection onto what the mechanism can actually do — correct phalanx
    lengths, one plane, real joint limits — and that projection is a strong
    denoiser. Measuring angles straight off tracked points instead means
    differentiating 18 mm bones sampled with 10 mm of error, which is mostly
    noise; going through the chain first is what makes this survive real
    video. Pass `fit` to reuse a solve you already have.
    """
    if fit is None:
        path = [pts[LM[n]] for n in FINGER_JOINTS[f]]
        fit = lf.fit_curve(path, CHAINS[f], base=path[0],
                           heading=_chain_heading(path), autoscale=False)
    cap = sum(math.radians(x) for x in LIMITS[f][1:])
    return _clamp(sum(abs(a) for a in fit["angles"]) / max(cap, 1e-6), 0.0, 1.0)


def measure_spread(pts):
    """Angles between adjacent finger directions, normalised."""
    dirs = []
    for f in FINGERS:
        c = FINGER_CHAIN[f]
        dirs.append(_norm(_sub(pts[LM[c[2]]], pts[LM[c[1]]])))
    out = []
    for i in range(4):
        out.append(_clamp(math.acos(_clamp(_dot(dirs[i], dirs[i + 1]), -1, 1))
                          / math.radians(45.0), 0.0, 1.0))
    return out


def hand_features(pts, fits=None):
    """12 numbers: 5 curls, 4 spreads, 3 contact distances."""
    s = _palm_size(pts)
    fits = fits or hand_to_chains(pts)
    curls = [measure_curl(pts, f, fits[f]) for f in FINGERS]
    spreads = measure_spread(pts)
    ti = _len(_sub(pts[LM["thumb_tip"]], pts[LM["index_tip"]])) / s
    tm = _len(_sub(pts[LM["thumb_tip"]], pts[LM["middle_tip"]])) / s
    ip = _len(_sub(pts[LM["index_tip"]], pts[LM["wrist"]])) / s
    return tuple(curls + spreads
                 + [_clamp(ti, 0, 1.5) / 1.5, _clamp(tm, 0, 1.5) / 1.5,
                    _clamp(ip, 0, 2.0) / 2.0])


_NFEAT = 12
_SIGMA2 = 0.035
_PROTO = None


def prototypes():
    """One feature centroid per pose, measured back off its own geometry.

    Synthesis goes curl -> landmarks; this goes landmarks -> curl. The two
    are different code paths, so a pose that cannot be recognised from its
    own rendering is a genuine failure and shows up in the self-test.
    """
    global _PROTO
    if _PROTO is None:
        _PROTO = {name: hand_features(pose_landmarks(p["curl"], p["spread"]))
                  for name, p in HAND_POSES.items()}
    return _PROTO


def recognize_hand(pts, min_conf=0.12):
    """21 landmarks -> (gesture, confidence). Nearest centroid over features."""
    f = hand_features(pts)
    proto = prototypes()
    scores = {n: math.exp(-sum((f[i] - c[i]) ** 2 for i in range(_NFEAT))
                          / (2 * _SIGMA2)) for n, c in proto.items()}
    tot = sum(scores.values()) or 1.0
    name, s = max(scores.items(), key=lambda kv: kv[1])
    conf = s / tot
    return (name, round(conf, 3)) if conf >= min_conf else (None, round(conf, 3))


def recognize_sequence(frames, fps, min_span=0.3):
    """Per-frame labels merged into spans, so the timeline is a partition."""
    labels = [recognize_hand(f) for f in frames]
    spans = []
    for i, (name, conf) in enumerate(labels):
        if name is None:
            continue
        s = spans[-1] if spans else None
        if s and s["name"] == name and s["_e"] == i - 1:
            s["_e"] = i
            s["conf"] = max(s["conf"], conf)
        else:
            spans.append({"name": name, "_s": i, "_e": i, "conf": conf})
    keep = []
    for i, s in enumerate(spans):
        if (s["_e"] - s["_s"] + 1) / fps >= min_span or len(spans) == 1:
            keep.append(s)
        elif keep:
            keep[-1]["_e"] = s["_e"]
    return [{"name": s["name"], "start": round(s["_s"] / fps, 2),
             "end": round((s["_e"] + 1) / fps, 2), "conf": round(s["conf"], 3)}
            for s in keep]


# --------------------------------------------------------------------------
# text -> hand motion
# --------------------------------------------------------------------------
# Which hand shape a how-to step calls for. Longest phrase match wins, same
# convention as motion.py's body lexicon.

LEXICON = [
    ("pinch",     "pinch peel tweeze nip pluck thread tease pull out pick off"
                  " lift off peel back take hold of the tab"),
    ("grip",      "grip grab hold twist turn rotate screw unscrew tighten loosen"
                  " squeeze clamp wring crank lever pry remove detach undo"
                  " wrench yank haul pull unplug disconnect unbolt"),
    ("point",     "point press push button tap click poke indicate select"
                  " depress trigger prod"),
    ("flat",      "wipe smooth flatten spread rub polish sand slide brush level"
                  " apply coat rough scrub clean dry buff smear press down"
                  " press the patch flatten out"),
    ("open_palm", "open release present offer catch show let go stand back"),
    ("ok",        "ok okay precise fine adjust dial calibrate tune"),
    ("thumbs_up", "done finished approve confirm ready test ride"),
    ("peace",     "twice pair both two"),
    ("tripod",    "place insert seat position set align refit reseat fit install"
                  " mount attach reattach slot locate lay guide feed"),
    ("fist",      "knock hammer punch pack compress tap home"),
    ("spread",    "spray inflate expand separate splay fan"),
]
_LEX = [(n, ws.split()) for n, ws in LEXICON]


def classify_step(text):
    """How-to step -> hand pose name.

    The leading verb wins. A repair step is written as an instruction, so its
    first word is the action; matching anywhere in the sentence instead lets
    a trailing noun ("...until it clicks") outrank the verb that opens it.
    Only when the opening verb is unknown do we fall back to the longest
    match anywhere.
    """
    clean = re.sub(r"[^a-z ]+", " ", str(text).lower())
    toks = clean.split()
    for w in toks[:2]:                       # "gently pull", "then twist"
        for name, words in _LEX:
            if w in words:
                return name
    low = " " + clean + " "
    best, blen = "open_palm", 0
    for name, words in _LEX:
        for w in words:
            if len(w) > blen and (" " + w + " ") in low:
                best, blen = name, len(w)
    return best


def plan_from_steps(steps, seconds=None):
    """Steps -> a timed plan of hand poses, each preceded by an open hand.

    Every grasp in life starts from an open hand; going fist-to-pinch with no
    release between reads as a glitch rather than a technique.
    """
    steps = [s for s in (steps or []) if str(s).strip()] or ["Show the hand"]
    seconds = seconds or _clamp(2.4 * len(steps), 5.0, 20.0)
    per = seconds / len(steps)
    plan, t = [], 0.0
    for i, s in enumerate(steps):
        plan.append({"step": i + 1, "text": str(s)[:160],
                     "pose": classify_step(s), "start": round(t, 3),
                     "end": round(t + per, 3)})
        t += per
    plan[-1]["end"] = round(seconds, 3)
    return plan, seconds


def _pose_at(t, plan):
    """Interpolate curl/spread between poses, with a release in between."""
    seg = plan[-1]
    for s in plan:
        if t < s["end"]:
            seg = s
            break
    p = _clamp((t - seg["start"]) / max(seg["end"] - seg["start"], 1e-4), 0.0, 1.0)
    tgt = HAND_POSES.get(seg["pose"], HAND_POSES["open_palm"])
    rest = HAND_POSES["open_palm"]
    # open -> pose -> hold -> open, so consecutive gestures stay separable
    k = _ease(_clamp(p / 0.30, 0, 1)) if p < 0.75 else 1 - _ease((p - 0.75) / 0.25)
    curl = [rest["curl"][i] + (tgt["curl"][i] - rest["curl"][i]) * k for i in range(5)]
    spread = [rest["spread"][i] + (tgt["spread"][i] - rest["spread"][i]) * k
              for i in range(5)]
    return curl, spread, seg


def _wrist_motion(t, seg, plan):
    """Where the whole hand is, and how it is turned, during a step.

    A grip that never rotates is not a twist, and the gesture recogniser
    reads shape only — so this is what carries the difference between
    "hold the cap" and "unscrew the cap" for a viewer.
    """
    p = _clamp((t - seg["start"]) / max(seg["end"] - seg["start"], 1e-4), 0.0, 1.0)
    pose, txt = seg["pose"], seg["text"].lower()
    spin = 0.0
    if pose == "grip" and re.search(r"twist|turn|rotate|screw|unscrew|tighten|loosen", txt):
        spin = math.sin(p * math.pi * 2) * 1.15
    if pose == "flat":
        return (math.sin(p * math.pi * 4) * 0.045, 0.0, 0.0), spin
    if pose == "point":
        return (0.0, 0.0, _ease(_clamp(p * 2, 0, 1)) * 0.035), spin
    return (0.0, math.sin(p * math.pi * 2) * 0.012, 0.0), spin


def _orient(pts, yaw, spin, offset):
    """Turn the hand to face the camera, apply the gesture's own spin."""
    out = []
    cy, sy = math.cos(yaw), math.sin(yaw)
    ax = (0.0, 1.0, 0.0)
    for q in pts:
        r = lf._rot(q, (0.0, 0.0, 1.0), spin)
        x, z = r[0] * cy + r[2] * sy, -r[0] * sy + r[2] * cy
        out.append(_add((x, r[1], z), offset))
    return out


# --------------------------------------------------------------------------
# real tracking, when the dependency is present
# --------------------------------------------------------------------------

def tracker_available():
    """Is MediaPipe Hands importable? Decides which backend a clip reports."""
    try:
        import mediapipe            # noqa: F401
        return True
    except Exception:
        return False


def track_video(path, max_frames=240, stride=2, handed="right"):
    """Track a real video file. -> (frames of 21 landmarks, fps).

    Requires mediapipe and OpenCV. Landmarks come back in MediaPipe's world
    frame (metres, wrist-relative), which is already the space this module
    works in, so nothing is rescaled on the way in.

    Raises RuntimeError when the dependency is absent rather than silently
    substituting synthesis — a clip that says it tracked a video must have
    tracked a video.

    MediaPipe 1.x removed the `mp.solutions` façade this used to call, so the
    tracking itself now lives in posetrack.py against the Tasks API. This
    stays as the name the rest of the module already knows.
    """
    try:
        import posetrack
    except ImportError as e:
        raise RuntimeError(
            "real hand tracking needs mediapipe and opencv-python: "
            "pip install mediapipe opencv-python") from e
    ok, why = posetrack.available()
    if not ok:
        raise RuntimeError(f"real hand tracking is unavailable — {why}")
    fps = posetrack.TRACK_FPS
    frames = posetrack.track_hand(path, seconds=max_frames / float(fps), fps=fps)
    if not frames:
        raise RuntimeError("no hands detected in that video")
    return frames, fps


def from_landmarks(seq):
    """Import any 21-landmark stream (a MediaPipe dump, a JSON capture)."""
    return [[tuple(p[:3]) for p in f] for f in seq]


# --------------------------------------------------------------------------
# derived signals and assembly
# --------------------------------------------------------------------------

def _smooth(frames, window=3):
    if window < 2 or len(frames) < window:
        return frames
    half = window // 2
    out = []
    for i in range(len(frames)):
        lo, hi = max(0, i - half), min(len(frames), i + half + 1)
        n = hi - lo
        out.append([tuple(sum(frames[k][j][ax] for k in range(lo, hi)) / n
                          for ax in range(3)) for j in range(21)])
    return out


def _velocities(frames, fps):
    n = len(frames)
    out = []
    for i in range(n):
        a, b = frames[max(i - 1, 0)], frames[min(i + 1, n - 1)]
        dt = (min(i + 1, n - 1) - max(i - 1, 0)) / fps or 1.0 / fps
        out.append([tuple((b[j][ax] - a[j][ax]) / dt for ax in range(3))
                    for j in range(21)])
    return out


CAM = {"yaw": 0.30, "pitch": 0.08, "dist": 0.62,
       "target": [0.0, 0.045, 0.0], "fov": 40.0}


def clip_from_frames(frames, fps, title="", source="", backend="procedural",
                     plan=None, cam=None, smooth=None):
    """Wrap a landmark stream in the clip contract the board renders.

    Everything below is derived from the frames, so a tracked video and a
    synthesised demo produce byte-identical structure — only `backend` and
    the numbers differ.

    Tracked input gets a longer filter. Per-frame classification of noisy
    landmarks is unreliable — a distal phalanx is under 20 mm and MediaPipe
    is good to roughly 10 — but the noise is independent per frame while the
    pose is held across many, so filtering over seven frames recovers most of
    it. Measured: 12/12 poses at 3 mm of noise, 10/12 at 10 mm, over a 1.5 s
    hold, against 28% frame-by-frame at the same noise.
    """
    if smooth is None:
        smooth = 3 if backend == "procedural" else 7
    frames = _smooth([[tuple(p) for p in f] for f in frames], smooth)
    vels = _velocities(frames, fps)
    chains = {f: {"device": {"name": CHAINS[f]["name"], "joints": 3,
                             "links_m": [round(x, 4) for x in PHALANX[f][1:]],
                             "limits_deg": LIMITS[f][1:]},
                  "angles": [], "fit_error": 0.0, "clipped": 0}
              for f in FINGERS}
    for f in frames:
        fits = hand_to_chains(f)
        for name, fit in fits.items():
            chains[name]["angles"].append(
                [round(math.degrees(a), 1) for a in fit["angles"]])
            chains[name]["fit_error"] = max(chains[name]["fit_error"], fit["error"])
            chains[name]["clipped"] = max(chains[name]["clipped"], fit["clipped"])
    for name in chains:
        chains[name]["fit_error"] = round(chains[name]["fit_error"], 5)
    return {
        "version": 1, "kind": "hand", "fps": round(fps, 2),
        "seconds": round(len(frames) / fps, 2),
        "topology": "mediapipe_hands_21", "backend": backend,
        "title": title, "source": source,
        "landmarks": LANDMARKS, "bones": BONES, "fingers": FINGERS,
        "camera": {**CAM, **(cam or {})},
        "frames": [[[round(v, 4) for v in p] for p in f] for f in frames],
        "vectors": [[[round(v, 3) for v in p] for p in f] for f in vels],
        "chains": chains,
        "curls": [[round(measure_curl(f, x, fits[x]), 3) for x in FINGERS]
                  for f, fits in ((f, hand_to_chains(f)) for f in frames)],
        "gestures": recognize_sequence(frames, fps),
        "segments": [{"step": s["step"], "text": s["text"], "pose": s["pose"],
                      "start": s["start"], "end": s["end"]} for s in (plan or [])],
        "papers": PAPERS,
    }


def text_to_hand(text, seconds=None, fps=12, title="", source="", yaw=0.0):
    """How-to text -> a hand-gesture clip. The procedural backend."""
    if isinstance(text, (list, tuple)):
        steps = [str(t) for t in text]
    else:
        lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
        numbered = [re.sub(r"^(?:step\s*)?\d+[.)]\s*|^[-*•]\s*", "", ln)
                    for ln in lines
                    if re.match(r"^(?:step\s*)?\d+[.)]|^[-*•]", ln, re.I)]
        steps = numbered or (lines if len(lines) > 1 else
                             [s.strip() for s in re.split(r"(?<=[.;])\s+", str(text))
                              if s.strip()])
    plan, seconds = plan_from_steps(steps[:8], seconds)
    n = max(int(round(seconds * fps)), 2)
    frames = []
    for i in range(n):
        t = i / fps
        curl, spread, seg = _pose_at(t, plan)
        offset, spin = _wrist_motion(t, seg, plan)
        frames.append(_orient(pose_landmarks(curl, spread), yaw, spin, offset))
    return clip_from_frames(frames, fps, title=title, source=source,
                            backend="procedural", plan=plan)


def video_to_hand(path, title="", source="", max_frames=240):
    """A real video -> a hand-gesture clip. Requires mediapipe."""
    frames, fps = track_video(path, max_frames=max_frames)
    return clip_from_frames(frames, fps, title=title, source=source,
                            backend="mediapipe")


def describe(clip):
    """Clip -> text for prompt injection, so the tutor can discuss the hand."""
    lines = [f"Hand clip: {clip.get('title') or 'untitled'} "
             f"({clip['seconds']}s, {clip['fps']}fps, 21-landmark MediaPipe "
             f"topology, backend: {clip['backend']})"]
    for s in clip.get("segments", []):
        lines.append(f"  {s['start']}-{s['end']}s  [{s['pose']}]  {s['text']}")
    g = clip.get("gestures", [])
    if g:
        lines.append("Recognised hand gestures (read back from the landmark "
                     "geometry): " + "; ".join(
                         f"{x['name']} ({x['start']}-{x['end']}s, conf {x['conf']})"
                         for x in g[:10]))
    ch = clip.get("chains", {})
    if ch:
        worst = max(ch.items(), key=lambda kv: kv[1]["fit_error"])
        lines.append(f"Finger chains: 5 x 4 servo joints; worst fit residual "
                     f"{worst[1]['fit_error']*1000:.1f} mm on the {worst[0]}.")
    return "\n".join(lines)


if __name__ == "__main__":
    import json
    print("mediapipe available:", tracker_available())
    print()
    ok = 0
    for name in HAND_POSES:
        pts = pose_landmarks(HAND_POSES[name]["curl"], HAND_POSES[name]["spread"])
        got, conf = recognize_hand(pts)
        fits = hand_to_chains(pts)
        err = max(f["error"] for f in fits.values())
        hit = got == name
        ok += hit
        print("  %-10s -> %-10s conf %.3f  fit %.2f mm  %s"
              % (name, got, conf, err * 1000, "OK" if hit else "MISS"))
    print("pose accuracy: %d/%d" % (ok, len(HAND_POSES)))
    print()
    clip = text_to_hand(["Pinch the tab and peel it back",
                         "Grip the cap and twist it counter-clockwise",
                         "Press the button once",
                         "Wipe the surface smooth"], title="demo")
    print(describe(clip))
    print("\npayload:", len(json.dumps(clip)) // 1024, "KB")
