"""
motion.py — 3D skeleton animation and gesture tracking, built out of
LineFORM actuated curve chains.

The idea, taken from the paper
------------------------------
LineFORM turns vector data into physical shape: resample a curve at the
joint pitch, solve one angle per servo, and what you get back is the shape
the hardware can actually hold. This file applies that to a human body.

Every limb here IS an actuated curve — a chain of 1-DOF servos with real
angle limits and a mass-spring settle. A how-to step becomes a wrist
trajectory, two-link IK gives the elbow, and the resulting shoulder-elbow-
wrist path is handed to `lineform.fit_curve` exactly like a CAD outline.
The skeleton the board draws is therefore not a rig with bones; it is a set
of chains obeying the paper's kinematics, which is why the arms lag and bow
during a fast gesture instead of snapping between poses. The paper calls
this out directly:

    "it can also record motion and replay back on your body ... to learn
     kinesthetic motion such as sports and dances as an external motor
     memory or to provide physical feedforward and guidance for gestural
     interaction"                                        (LineFORM, Fig. 9)

That is this whole file's job: record a how-to gesture, replay it on a body.

Pipeline
--------
    step text ->  verb lexicon      ->  gesture primitive
              ->  wrist trajectory  ->  2-link IK  ->  limb control path
              ->  lineform.fit_curve + mass-spring ->  physical chain frames
              ->  finite difference ->  per-joint motion vectors
              ->  Gaussian splat    ->  dense flow field
              ->  graph features    ->  recognised gestures

Recognition runs on the frames, not on the plan that produced them, so the
same code labels an imported MediaPipe/BlazePose track. Synthesis and
tracking are a closed loop.

Interfaces kept swappable (see PAPERS.md): BlazePose / MediaPipe Hands
landmark maps, a VideoPose3D-shaped 2D->3D lift, SMPL joint aliases, an
MDM-shaped text->motion entry point, ST-GCN-style recognition, RAFT-style
dense flow. The default backends are procedural: no weights, no GPU, no
network.

    from motion import text_to_motion
    clip = text_to_motion("Twist the valve cap counter-clockwise", seconds=6)
    print(clip["gestures"])
"""

from __future__ import annotations

import math
import re

import lineform as lf
from lineform import _add, _sub, _mul, _dot, _len, _norm, _cross, _clamp

# --------------------------------------------------------------------------
# topology  (BlazePose / COCO-17 body, MediaPipe hand decimated to 6)
# --------------------------------------------------------------------------

JOINTS = ["nose", "eye_l", "eye_r", "ear_l", "ear_r",
          "shoulder_l", "shoulder_r", "elbow_l", "elbow_r", "wrist_l", "wrist_r",
          "hip_l", "hip_r", "knee_l", "knee_r", "ankle_l", "ankle_r"]
J = {n: i for i, n in enumerate(JOINTS)}

BONES = [("shoulder_l", "shoulder_r"), ("hip_l", "hip_r"),
         ("shoulder_l", "hip_l"), ("shoulder_r", "hip_r"),
         ("shoulder_l", "elbow_l"), ("elbow_l", "wrist_l"),
         ("shoulder_r", "elbow_r"), ("elbow_r", "wrist_r"),
         ("hip_l", "knee_l"), ("knee_l", "ankle_l"),
         ("hip_r", "knee_r"), ("knee_r", "ankle_r")]
BONE_IDX = [[J[a], J[b]] for a, b in BONES]

# MediaPipe Hands ships 21 landmarks; at board scale only the wrist and the
# five tips survive legibly, so we carry the documented decimation.
HAND_JOINTS = ["wrist", "thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
HAND_BONES = [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5]]
MEDIAPIPE_HAND_MAP = {"wrist": 0, "thumb_tip": 4, "index_tip": 8,
                      "middle_tip": 12, "ring_tip": 16, "pinky_tip": 20}

# BlazePose's 33 landmarks -> our 17, so a real tracker's output loads
# without a bespoke adapter.
BLAZEPOSE_MAP = {"nose": 0, "eye_l": 2, "eye_r": 5, "ear_l": 7, "ear_r": 8,
                 "shoulder_l": 11, "shoulder_r": 12, "elbow_l": 13, "elbow_r": 14,
                 "wrist_l": 15, "wrist_r": 16, "hip_l": 23, "hip_r": 24,
                 "knee_l": 25, "knee_r": 26, "ankle_l": 27, "ankle_r": 28}

SMPL_ALIAS = {"shoulder_l": "left_shoulder", "shoulder_r": "right_shoulder",
              "elbow_l": "left_elbow", "elbow_r": "right_elbow",
              "wrist_l": "left_wrist", "wrist_r": "right_wrist",
              "hip_l": "left_hip", "hip_r": "right_hip",
              "knee_l": "left_knee", "knee_r": "right_knee",
              "ankle_l": "left_ankle", "ankle_r": "right_ankle", "nose": "head"}

PAPERS = ["LineFORM (Nakagaki, Follmer & Ishii, UIST '15)", "SketchAgent",
          "CogVideoX", "BlazePose", "MediaPipe Hands", "VideoPose3D",
          "SMPL", "MDM", "ST-GCN", "RAFT"]

# --------------------------------------------------------------------------
# the body, as chains of actuated curve
# --------------------------------------------------------------------------
# Limb lengths are anatomical; joint counts are chosen so each chain's total
# length matches its limb, because a chain cannot be longer or shorter than
# the sum of its links. Angle limits are looser than the published +/-103 deg
# because an elbow is a tighter hinge than a servo bracket.

UPPER_ARM, FOREARM = 0.28, 0.26
THIGH, SHIN = 0.43, 0.44

CHAINS = {
    "spine":  lf.make_device(6, 0.0780, 60.0, "alternating", "spine chain"),
    "neck":   lf.make_device(3, 0.0870, 45.0, "alternating", "neck chain"),
    "arm_l":  lf.make_device(8, 0.0675, 150.0, "alternating", "left arm chain"),
    "arm_r":  lf.make_device(8, 0.0675, 150.0, "alternating", "right arm chain"),
    "leg_l":  lf.make_device(10, 0.0870, 150.0, "alternating", "left leg chain"),
    "leg_r":  lf.make_device(10, 0.0870, 150.0, "alternating", "right leg chain"),
}

REST = {
    "nose": (0.00, 1.62, 0.09), "eye_l": (0.035, 1.65, 0.07), "eye_r": (-0.035, 1.65, 0.07),
    "ear_l": (0.075, 1.63, 0.00), "ear_r": (-0.075, 1.63, 0.00),
    "shoulder_l": (0.19, 1.42, 0.00), "shoulder_r": (-0.19, 1.42, 0.00),
    "elbow_l": (0.23, 1.16, 0.02), "elbow_r": (-0.23, 1.16, 0.02),
    "wrist_l": (0.25, 0.92, 0.05), "wrist_r": (-0.25, 0.92, 0.05),
    "hip_l": (0.10, 0.95, 0.00), "hip_r": (-0.10, 0.95, 0.00),
    "knee_l": (0.11, 0.52, 0.01), "knee_r": (-0.11, 0.52, 0.01),
    "ankle_l": (0.11, 0.08, 0.00), "ankle_r": (-0.11, 0.08, 0.00),
}

# where the hands do their work: in front, chest-high, slightly right
WORK = (0.16, 1.14, 0.34)


def _lerp(a, b, t): return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))
def _ease(p): return p * p * (3 - 2 * p)                    # smoothstep


def two_link_ik(root, target, l1, l2, pole):
    """Analytic 2-bone IK. -> (mid_joint, reachable_target).

    Out-of-reach targets are pulled onto the reach sphere rather than
    snapping the limb straight at full stretch, which is what keeps a
    procedural arm reading as an arm instead of a compass needle.
    """
    d = _sub(target, root)
    dist = _len(d)
    reach = l1 + l2 - 1e-3
    if dist > reach:
        target = _add(root, _mul(_norm(d), reach))
        d, dist = _sub(target, root), reach
    dist = max(dist, 1e-4)
    u = _mul(d, 1.0 / dist)
    a = (l1 * l1 - l2 * l2 + dist * dist) / (2 * dist)
    h = math.sqrt(max(l1 * l1 - a * a, 0.0))
    p = _sub(pole, root)
    perp = _sub(p, _mul(u, _dot(p, u)))
    if _len(perp) < 1e-6:
        perp = _cross(u, (0.0, 1.0, 0.0))
        if _len(perp) < 1e-6:
            perp = (1.0, 0.0, 0.0)
    return _add(_add(root, _mul(u, a)), _mul(_norm(perp), h)), target


# --------------------------------------------------------------------------
# gesture primitives: how-to verb -> wrist trajectory
# --------------------------------------------------------------------------
# f(p, side) -> pose intent, p is 0..1 through the gesture, side is +1 left
# / -1 right. Trajectories rather than keyframes, so any duration resamples.

def _work(side, dx=0.0, dy=0.0, dz=0.0):
    return (side * WORK[0] + side * dx, WORK[1] + dy, WORK[2] + dz)


def _g_point(p, side):
    reach = _ease(min(p * 1.8, 1.0))
    jab = 0.03 * math.sin(p * math.pi * 4) if p > 0.45 else 0.0
    return {"wrist": _lerp(REST["wrist_r"], _work(side, 0.10, 0.10, 0.16 + jab), reach),
            "open": 0.15, "index": 1.0, "twist": 0.0,
            "look": _work(side, 0.10, 0.10, 0.30), "lean": 0.02 * reach}


def _g_grasp(p, side):
    approach = _ease(min(p * 2.2, 1.0))
    close = _clamp((p - 0.42) * 4.0, 0.0, 1.0)
    lift = _ease(_clamp((p - 0.66) * 3.0, 0.0, 1.0)) * 0.10
    return {"wrist": _lerp(REST["wrist_r"], _work(side, 0.02, lift, 0.02), approach),
            "open": 1.0 - 0.92 * close, "index": 0.0, "twist": -0.25 * close,
            "look": _work(side), "lean": 0.03}


def _g_twist(p, side):
    settle = _ease(min(p * 2.5, 1.0))
    ang = p * math.pi * 2.4
    tgt = _add(_work(side), (0.03 * math.cos(ang), 0.03 * math.sin(ang), 0.0))
    return {"wrist": _lerp(REST["wrist_r"], tgt, settle), "open": 0.10,
            "index": 0.0, "twist": ang * side, "look": _work(side), "lean": 0.02}


def _g_pull(p, side):
    grab = _ease(min(p * 2.6, 1.0))
    back = _ease(_clamp((p - 0.40) * 1.9, 0.0, 1.0))
    tgt = _lerp(_work(side), _work(side, -0.06, 0.06, -0.34), back)
    return {"wrist": _lerp(REST["wrist_r"], tgt, grab), "open": 0.12,
            "index": 0.0, "twist": 0.15, "look": _work(side), "lean": -0.06 * back}


def _g_push(p, side):
    ready = _ease(min(p * 2.4, 1.0))
    out = _ease(_clamp((p - 0.35) * 1.9, 0.0, 1.0))
    tgt = _lerp(_work(side, -0.04, 0.02, -0.26), _work(side, 0.02, 0, 0.12), out)
    return {"wrist": _lerp(REST["wrist_r"], tgt, ready), "open": 0.30,
            "index": 0.0, "twist": -0.1, "look": _work(side), "lean": 0.07 * out}


def _g_lift(p, side):
    grab = _ease(min(p * 2.0, 1.0))
    rise = _ease(_clamp((p - 0.42) * 2.0, 0.0, 1.0))
    tgt = _lerp(_work(side, 0.0, -0.26, 0.0), _work(side, 0.0, 0.20, -0.06), rise)
    return {"wrist": _lerp(REST["wrist_r"], tgt, grab), "open": 0.15, "index": 0.0,
            "twist": 0.0, "look": tgt, "lean": -0.03 * rise, "both": True}


def _g_wipe(p, side):
    settle = _ease(min(p * 2.6, 1.0))
    sweep = math.sin(p * math.pi * 5) * 0.20
    tgt = _add(_work(side, 0, 0.02, 0.02), (sweep, 0, 0))
    return {"wrist": _lerp(REST["wrist_r"], tgt, settle), "open": 0.45,
            "index": 0.0, "twist": 0.6 * side, "look": _work(side), "lean": 0.02}


def _g_cut(p, side):
    settle = _ease(min(p * 2.4, 1.0))
    snip = abs(math.sin(p * math.pi * 4))
    # The scissor action is mostly in the fingers, but a real cut rocks the
    # whole hand with each snip. Without that the wrist track is a slow creep
    # indistinguishable from a lift, and both the animation and the gesture
    # classifier lose the thing that makes cutting look like cutting.
    tgt = _work(side, p * 0.10, 0.02 + 0.05 * snip, 0.04 - 0.02 * snip)
    return {"wrist": _lerp(REST["wrist_r"], tgt, settle),
            "open": 0.55 * snip, "index": 0.55, "twist": 0.35 * side,
            "look": _work(side), "lean": 0.03, "both": True}


def _g_pour(p, side):
    raise_ = _ease(min(p * 2.2, 1.0))
    tilt = _ease(_clamp((p - 0.35) * 1.8, 0.0, 1.0))
    return {"wrist": _lerp(REST["wrist_r"], _work(side, 0.02, 0.20 - 0.05 * tilt, 0.02), raise_),
            "open": 0.12, "index": 0.0, "twist": -1.5 * tilt * side,
            "look": _work(side, 0, -0.10, 0), "lean": 0.02}


def _g_tap(p, side):
    settle = _ease(min(p * 3.0, 1.0))
    jab = abs(math.sin(p * math.pi * 6)) * 0.11
    return {"wrist": _lerp(REST["wrist_r"], _work(side, 0.04, jab, 0.06), settle),
            "open": 0.20, "index": 0.9, "twist": 0.0, "look": _work(side), "lean": 0.01}


def _g_crank(p, side):
    settle = _ease(min(p * 2.4, 1.0))
    ang = p * math.pi * 4
    tgt = _add(_work(side, 0.02, -0.06, 0.0),
               (0.0, 0.15 * math.sin(ang), 0.15 * math.cos(ang)))
    return {"wrist": _lerp(REST["wrist_r"], tgt, settle), "open": 0.10,
            "index": 0.0, "twist": ang * 0.3 * side, "look": _work(side), "lean": 0.02}


def _g_inspect(p, side):
    up = _ease(min(p * 2.0, 1.0))
    tgt = (side * 0.16, 1.48, 0.26)
    return {"wrist": _lerp(REST["wrist_r"], tgt, up), "open": 0.18, "index": 0.0,
            "twist": math.sin(p * math.pi * 2) * 0.5 * side, "look": tgt,
            "lean": 0.0, "head_tilt": 0.10 * math.sin(p * math.pi * 2)}


def _g_explain(p, side):
    open_ = _ease(min(p * 2.0, 1.0))
    sway = math.sin(p * math.pi * 2.2)
    tgt = (side * (0.30 + 0.05 * sway), 1.12 + 0.05 * sway, 0.24)
    return {"wrist": _lerp(REST["wrist_r"], tgt, open_), "open": 1.0, "index": 0.0,
            "twist": 1.1 * side, "look": (0.0, 1.45, 0.6), "lean": 0.0, "both": True}


PRIMITIVES = {"point": _g_point, "grasp": _g_grasp, "twist": _g_twist,
              "pull": _g_pull, "push": _g_push, "lift": _g_lift, "wipe": _g_wipe,
              "cut": _g_cut, "pour": _g_pour, "tap": _g_tap, "crank": _g_crank,
              "inspect": _g_inspect, "explain": _g_explain}

# verb -> primitive; the longest phrase found in the step wins.
LEXICON = [
    ("twist", "twist turn rotate screw unscrew tighten loosen thread wind"
              " counterclockwise clockwise"),
    ("pull", "pull remove pry detach extract yank unplug peel disconnect"
             " take off pull out lift off"),
    ("push", "push press insert seat fit slide plug snap mount reattach refit"
             " install press down"),
    ("lift", "lift raise hold up carry pick up elevate"),
    ("wipe", "wipe clean rub scrub polish sand buff dry brush"),
    ("cut", "cut trim snip slice score strip shear prune"),
    ("pour", "pour apply spray lubricate water drop dispense oil squeeze coat"),
    ("tap", "tap knock bump nudge click"),
    ("crank", "pedal backpedal spin crank cycle roll rotate the pedals"),
    ("inspect", "inspect examine look at check for find locate identify observe"
                " test check"),
    ("grasp", "grasp grab hold take grip pick clamp secure"),
    ("point", "point show note notice see this here"),
]
_LEX = [(name, [w.strip() for w in words.split()]) for name, words in LEXICON]


def classify_step(text: str) -> str:
    """How-to step -> gesture primitive name. Longest phrase match wins."""
    low = " " + re.sub(r"[^a-z ]+", " ", str(text).lower()) + " "
    best, best_len = "explain", 0
    for name, words in _LEX:
        for w in words:
            if len(w) > best_len and (" " + w + " ") in low:
                best, best_len = name, len(w)
    return best


# --------------------------------------------------------------------------
# posing: intent -> anatomical landmarks
# --------------------------------------------------------------------------

def _head_pose(look, tilt):
    """Rotate the five face landmarks to face `look` about the neck."""
    neck = (0.0, 1.50, 0.0)
    d = _norm(_sub(look, neck))
    yaw = math.atan2(d[0], max(d[2], 1e-4)) * 0.55
    pitch = _clamp(math.asin(_clamp(d[1] - 0.05, -1, 1)) * 0.45, -0.5, 0.5)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch + tilt), math.sin(pitch + tilt)
    out = {}
    for name in ("nose", "eye_l", "eye_r", "ear_l", "ear_r"):
        x, y, z = _sub(REST[name], neck)
        y, z = y * cp - z * sp, y * sp + z * cp
        x, z = x * cy + z * sy, -x * sy + z * cy
        out[name] = _add(neck, (x, y, z))
    return out


def _hand_points(wrist, elbow, state, side):
    """MediaPipe-style hand landmarks in world space.

    Fingers fan out along the forearm in a frame twisted by the gesture, so
    an open palm, a fist and a pointing index all read from whichever side
    the camera sees.
    """
    fwd = _norm(_sub(wrist, elbow))
    up = (0.0, 1.0, 0.0)
    rt = _norm(_cross(fwd, up))
    up = _norm(_cross(rt, fwd))
    tw = state.get("twist", 0.0)
    c, s = math.cos(tw), math.sin(tw)
    rt2 = _add(_mul(rt, c), _mul(up, s))
    up2 = _add(_mul(rt, -s), _mul(up, c))
    open_, idx = state.get("open", 0.5), state.get("index", 0.0)
    pts = [wrist]
    #          spread  length  lift
    fingers = [(-0.55, 0.052, -0.020),      # thumb
               (0.30, 0.082, 0.010),        # index
               (0.10, 0.088, 0.004),        # middle
               (-0.10, 0.082, -0.002),      # ring
               (-0.30, 0.072, -0.010)]      # pinky
    for i, (spread, length, lift) in enumerate(fingers):
        curl = open_ if i != 1 else max(open_, idx)
        L = length * (0.42 + 0.58 * curl)
        drop = (1.0 - curl) * 0.45
        v = _add(_add(_mul(fwd, L * (1 - drop * 0.5)), _mul(rt2, spread * L * 0.9)),
                 _mul(up2, lift - drop * L * 0.8))
        pts.append(_add(wrist, v))
    return pts


def _landmarks(t, plan, seconds):
    """One frame of anatomical landmarks + hand state. -> (dict, hands, seg)."""
    seg = plan[-1]
    for s in plan:
        if t < s["end"]:
            seg = s
            break
    p = _clamp((t - seg["start"]) / max(seg["end"] - seg["start"], 1e-4), 0.0, 1.0)
    st = PRIMITIVES.get(seg["primitive"], _g_explain)(p, seg.get("side", -1))
    side, both = seg.get("side", -1), st.get("both", False)

    breath, shift = 0.006 * math.sin(t * 1.9), 0.012 * math.sin(t * 0.7)
    lean = st.get("lean", 0.0)

    pos = dict(REST)
    for n in ("hip_l", "hip_r"):
        pos[n] = _add(REST[n], (shift, 0.0, 0.0))
    for n in ("shoulder_l", "shoulder_r"):
        pos[n] = _add(REST[n], (shift * 0.5, breath, lean * 0.5))
    for n in ("knee_l", "knee_r"):
        pos[n] = _add(REST[n], (shift * 0.6, 0.0, 0.0))

    hands = {}
    for hand_side, sfx in ((1, "l"), (-1, "r")):
        acting = both or (hand_side == side)
        sh = pos["shoulder_" + sfx]
        if acting:
            w = st["wrist"]
            if hand_side != side:                       # mirror the other hand
                w = (-w[0] + (0.06 if both else 0.0), w[1], w[2])
            state = st
        else:
            w = _add(REST["wrist_" + sfx], (shift * 0.4, breath * 0.6, 0.0))
            state = {"open": 0.5, "index": 0.0, "twist": 0.0}
        pole = _add(sh, (hand_side * 0.30, -0.34, -0.16))
        elbow, w = two_link_ik(sh, w, UPPER_ARM, FOREARM, pole)
        pos["elbow_" + sfx], pos["wrist_" + sfx] = elbow, w
        if acting:
            hands[sfx] = _hand_points(w, elbow, state, hand_side)

    pos.update(_head_pose(st.get("look", (0.0, 1.5, 0.8)), st.get("head_tilt", 0.0)))
    return pos, hands, seg


# --------------------------------------------------------------------------
# the LineFORM layer: landmarks -> actuated curve chains
# --------------------------------------------------------------------------

def _control_paths(pos):
    """Landmarks -> one control path per chain, ready for `fit_curve`.

    These are the "series of vectors" the paper extracts from an outline;
    here the outline is a body.
    """
    mid = lambda a, b: _mul(_add(pos[a], pos[b]), 0.5)
    hips, shou = mid("hip_l", "hip_r"), mid("shoulder_l", "shoulder_r")
    return {
        "spine": [hips, _lerp(hips, shou, 0.5), shou],
        "neck": [shou, _lerp(shou, pos["nose"], 0.6), pos["nose"]],
        "arm_l": [pos["shoulder_l"], pos["elbow_l"], pos["wrist_l"]],
        "arm_r": [pos["shoulder_r"], pos["elbow_r"], pos["wrist_r"]],
        "leg_l": [pos["hip_l"], pos["knee_l"], pos["ankle_l"]],
        "leg_r": [pos["hip_r"], pos["knee_r"], pos["ankle_r"]],
    }


def _chain_heading(path):
    """Start each chain pointing down its own first segment.

    A limb chain is mounted on the body, not on the world, so it does not
    inherit lineform's default base frame. Its two joint axes are then the
    directions perpendicular to that heading.
    """
    fwd = _norm(_sub(path[1], path[0]))
    ref = (0.0, 0.0, 1.0) if abs(fwd[2]) < 0.9 else (0.0, 1.0, 0.0)
    right = _norm(_cross(fwd, ref))
    up = _norm(_cross(right, fwd))
    return (right, up, fwd)


def _fit_chains(paths):
    """Fit every chain to its control path. -> {name: fit dict}.

    autoscale is off: a limb's length is anatomy, not framing, and each
    chain was sized in CHAINS to match the limb it drives.
    """
    out = {}
    for name, path in paths.items():
        dev = CHAINS[name]
        out[name] = lf.fit_curve(path, dev, base=path[0],
                                 heading=_chain_heading(path), autoscale=False)
    return out


def _chain_stiffness():
    """Per-chain compliance, the paper's variable-stiffness control.

    The base of a limb is held rigid and the far end is left soft, so the
    wrist trails the shoulder through a fast gesture — the physical read
    that a positional lerp cannot give you.
    """
    return {name: lf.constraint_stiffness(dev["n"], 0.45, 0.30)
            for name, dev in CHAINS.items()}


# --------------------------------------------------------------------------
# derived signals
# --------------------------------------------------------------------------

def _smooth(frames, window=3):
    """Temporal filter over a joint track.

    VideoPose3D's contribution is a temporal receptive field over noisy
    per-frame estimates; this is the cheap moving-average stand-in.
    """
    if window < 2 or len(frames) < window:
        return frames
    half = window // 2
    out = []
    for i in range(len(frames)):
        lo, hi = max(0, i - half), min(len(frames), i + half + 1)
        n = hi - lo
        out.append([tuple(sum(frames[k][j][ax] for k in range(lo, hi)) / n
                          for ax in range(3)) for j in range(len(frames[i]))])
    return out


def _velocities(frames, fps):
    """Central-difference velocity in metres/second, per joint per frame."""
    n = len(frames)
    out = []
    for i in range(n):
        a, b = frames[max(i - 1, 0)], frames[min(i + 1, n - 1)]
        dt = (min(i + 1, n - 1) - max(i - 1, 0)) / fps or 1.0 / fps
        out.append([tuple((b[j][ax] - a[j][ax]) / dt for ax in range(3))
                    for j in range(len(frames[i]))])
    return out


DEFAULT_CAM = {"yaw": 0.38, "pitch": 0.10, "dist": 3.2,
               "target": [0.0, 1.05, 0.0], "fov": 40.0}


def project(p, cam):
    """World metres -> normalised device coords in [-1,1]. -> (x, y, depth).

    index.html implements this identically; the flow field below is computed
    in this exact space, so the overlay lands on the skeleton rather than
    near it.
    """
    t = cam["target"]
    x, y, z = p[0] - t[0], p[1] - t[1], p[2] - t[2]
    cy, sy = math.cos(cam["yaw"]), math.sin(cam["yaw"])
    x, z = x * cy + z * sy, -x * sy + z * cy
    cp, sp = math.cos(cam["pitch"]), math.sin(cam["pitch"])
    y, z = y * cp - z * sp, y * sp + z * cp
    depth = max(cam["dist"] - z, 0.05)
    f = 1.0 / math.tan(math.radians(cam["fov"]) / 2)
    return x * f / depth, y * f / depth, depth


FLOW_W, FLOW_H = 14, 12


def flow_field(frames, vels, cam, fps):
    """Dense 2D motion field — RAFT's output format on a coarse grid.

    Joint velocities are projected to image space and Gaussian-splatted into
    a FLOW_W x FLOW_H grid, then quantised to signed bytes. This is the
    motion-vector data the board draws as a field of little arrows: the same
    representation a codec's macroblock vectors or an optical-flow net give
    you, just sparse-sourced from the skeleton.
    """
    raw, peak = [], 1e-6
    for fi in range(len(frames)):
        grid = [[0.0, 0.0] for _ in range(FLOW_W * FLOW_H)]
        wsum = [0.0] * (FLOW_W * FLOW_H)
        for j in range(len(frames[fi])):
            px, py, _ = project(frames[fi][j], cam)
            v = vels[fi][j]
            qx, qy, _ = project(_add(frames[fi][j], _mul(v, 1.0 / fps)), cam)
            dx, dy = (qx - px) * fps, (qy - py) * fps
            if abs(dx) + abs(dy) < 1e-4:
                continue
            gx = (px + 1) / 2 * (FLOW_W - 1)
            gy = (1 - (py + 1) / 2) * (FLOW_H - 1)
            for cyi in range(FLOW_H):
                for cxi in range(FLOW_W):
                    d2 = (cxi - gx) ** 2 + (cyi - gy) ** 2
                    if d2 > 9.0:
                        continue
                    w = math.exp(-d2 / 2.2)
                    k = cyi * FLOW_W + cxi
                    grid[k][0] += dx * w
                    grid[k][1] += dy * w
                    wsum[k] += w
        for k in range(len(grid)):
            if wsum[k] > 1e-6:
                grid[k][0] /= wsum[k]
                grid[k][1] /= wsum[k]
            peak = max(peak, abs(grid[k][0]), abs(grid[k][1]))
        raw.append(grid)
    # Quantise once the peak is known, then base64 the bytes. As JSON number
    # arrays this block dwarfs everything else on the wire; as base64 it is a
    # third the size, and a flow field is a bitmap anyway. Values are stored
    # biased by +128 so they survive as unsigned bytes.
    import base64
    step = 2 if len(raw) > 60 else 1               # 6 Hz is plenty for a field
    out = []
    for i in range(0, len(raw), step):
        buf = bytearray()
        for gx, gy in raw[i]:
            buf.append(int(_clamp(round(gx / peak * 127), -127, 127)) + 128)
            buf.append(int(_clamp(round(gy / peak * 127), -127, 127)) + 128)
        out.append(base64.b64encode(bytes(buf)).decode("ascii"))
    return {"w": FLOW_W, "h": FLOW_H, "scale": round(peak, 4),
            "encoding": "base64-int8+128", "step": step, "frames": out}


# --------------------------------------------------------------------------
# recognition: frames -> gestures  (ST-GCN-style graph features)
# --------------------------------------------------------------------------

# speed, vert, lat, depth, periodicity, symmetry, reach,
# net-depth, net-vert, net-lat, straightness, elbow-flex
_FEATS = 12
_SIGMA2 = 0.06      # nearest-centroid bandwidth over those twelve features

_PROTO_CACHE = {}


def prototypes(fps=12, win=None, with_hands=True):
    """One feature centroid per primitive, measured from the generator.

    Hand-authored prototype vectors are guesswork, and guessing wrong means
    the recogniser confidently mislabels its own output. Instead each
    primitive is run in isolation and the mean of its own feature vectors
    becomes its centroid — nearest-centroid classification calibrated on the
    thing it has to classify.

    The centroid must be measured through the same window the recogniser
    will use. Net displacement and straightness both depend on how much time
    a window spans, so a centroid built from a 2.4 s look at a gesture does
    not describe what a 1.4 s window sees of it; that mismatch is what makes
    a lift read as a cut. Cached per (fps, win).

    For imported tracks this stays meaningful: the centroids describe the
    kinematics of each gesture — how fast, how periodic, how symmetric, which
    way it travels — not anything specific to how we drew it.
    """
    win = win or max(int(fps * 1.4), 4)
    key = (fps, win, with_hands)
    if key in _PROTO_CACHE:
        return _PROTO_CACHE[key]
    dur = max(2.6, win / fps * 1.8)
    n = int(dur * fps)
    stride = max(win // 4, 1)
    nfeat = _FEATS + (2 if with_hands else 0)
    out = {}
    for name in PRIMITIVES:
        plan = [{"step": 1, "text": "", "primitive": name, "side": -1,
                 "start": 0.0, "end": dur}]
        raw = [_landmarks(i / fps, plan, dur) for i in range(n)]
        fr = _smooth([[r[0][j] for j in JOINTS] for r in raw], 3)
        hd = [r[1] for r in raw] if with_hands else None
        vl = _velocities(fr, fps)
        acc, cnt = [0.0] * nfeat, 0
        for lo in range(0, max(n - win + 1, 1), stride):
            f = _window_features(fr, vl, lo, min(lo + win, n), hd)
            if "r" not in f or len(f["r"][0]) != nfeat:
                continue
            for i in range(nfeat):
                acc[i] += f["r"][0][i]
            cnt += 1
        if cnt:
            out[name] = tuple(a / cnt for a in acc)
    _PROTO_CACHE[key] = out
    return out


def _aperture(hand_pts):
    """Mean wrist-to-fingertip distance, normalised. Fist 0 -> open palm 1."""
    w = hand_pts[0]
    d = sum(_len(_sub(p, w)) for p in hand_pts[1:]) / max(len(hand_pts) - 1, 1)
    return _clamp((d - 0.030) / 0.070, 0.0, 1.0)


def _window_features(frames, vels, lo, hi, hands=None):
    """Spatio-temporal features over one window, per hand.

    ST-GCN convolves over (joint graph x time); we keep both axes but reduce
    them to twelve interpretable numbers, because this classifier has to ship
    without weights and stay explainable in the UI.

    Magnitudes alone are not enough. Push and pull have identical speed,
    span and periodicity and differ only in which way the hand goes, so the
    signed net displacements (7-9) are what separate them. Straightness (10)
    separates a trajectory that goes somewhere — point, push — from one that
    returns to where it started, like crank or wipe. Elbow flex (11) catches
    the difference between working close in and reaching out.

    Every feature is scaled into [0,1] so one Gaussian bandwidth fits all.
    """
    out = {}
    for sfx, wj, sj, ej in (("r", J["wrist_r"], J["shoulder_r"], J["elbow_r"]),
                            ("l", J["wrist_l"], J["shoulder_l"], J["elbow_l"])):
        if hi - lo < 2:
            continue
        speeds = [_len(vels[i][wj]) for i in range(lo, hi)]
        disp = [_sub(frames[i][wj], frames[i][sj]) for i in range(lo, hi)]
        mean_speed = sum(speeds) / len(speeds)
        spans = [max(d[a] for d in disp) - min(d[a] for d in disp) for a in range(3)]
        total = sum(spans) or 1e-6

        ax = max(range(3), key=lambda a: spans[a])
        flips, prev = 0, 0.0
        for i in range(lo, hi):                    # sign changes = oscillation
            s = vels[i][wj][ax]
            if prev * s < 0:
                flips += 1
            prev = s
        period = _clamp(flips / max(hi - lo, 1) * 3.0, 0.0, 1.0)

        other = J["wrist_l"] if sfx == "r" else J["wrist_r"]
        osp = sum(_len(vels[i][other]) for i in range(lo, hi)) / max(hi - lo, 1)
        sym = _clamp(min(mean_speed, osp) / max(mean_speed, osp, 1e-6), 0.0, 1.0)
        reach = _clamp(sum(_len(d) for d in disp) / max(hi - lo, 1) / 0.52, 0.0, 1.0)

        net = _sub(frames[hi - 1][wj], frames[lo][wj])
        path = sum(_len(_sub(frames[i + 1][wj], frames[i][wj]))
                   for i in range(lo, hi - 1)) or 1e-6
        straight = _clamp(_len(net) / path, 0.0, 1.0)
        sgn = lambda v: _clamp(v / 0.30, -1.0, 1.0) * 0.5 + 0.5   # -> [0,1]

        flex = 0.0
        for i in range(lo, hi):
            a = _norm(_sub(frames[i][sj], frames[i][ej]))
            b = _norm(_sub(frames[i][wj], frames[i][ej]))
            flex += math.acos(_clamp(_dot(a, b), -1.0, 1.0))
        flex /= (hi - lo)

        feat = [_clamp(mean_speed / 1.4, 0, 1),
                spans[1] / total, spans[0] / total, spans[2] / total,
                period, sym, reach,
                sgn(net[2]), sgn(net[1]), sgn(net[0]),
                straight, _clamp(flex / math.pi, 0.0, 1.0)]

        # Grip, when the tracker gives us fingers. Some gestures live almost
        # entirely in the hand: cutting is a scissor snip whose wrist track is
        # a slow creep, indistinguishable from a lift on body joints alone.
        # MediaPipe Hands supplies exactly this, so when hand landmarks are
        # present we read mean aperture and how much it opens and closes.
        if hands is not None:
            ap = [_aperture(hands[i][sfx]) for i in range(lo, hi)
                  if hands[i].get(sfx)]
            if ap:
                feat.append(sum(ap) / len(ap))
                feat.append(_clamp((max(ap) - min(ap)) / 0.5, 0.0, 1.0))
            else:
                feat += [0.5, 0.0]
        out[sfx] = (tuple(feat), mean_speed)
    return out


def recognize_gestures(frames, vels, fps, hands=None, min_conf=0.2, min_span=0.8):
    """Sliding-window skeleton action recognition. -> [{name, hand, ...}].

    Runs on frames alone, so it behaves identically on a synthesised clip
    and an imported track. Adjacent windows sharing a label merge into one
    labelled span.

    The window has to be long enough to contain a whole gesture. At 0.6 s a
    twist looks like a point on the way out and a pull on the way back, and
    the output is a stutter of half-gestures; 1.4 s spans the primitives
    this lexicon generates. Spans shorter than `min_span` survive only if
    nothing else claims their time, which drops the boundary flicker without
    hiding a genuinely brief beat.
    """
    nf = len(frames)
    win = max(int(fps * 1.4), 4)
    stride = max(win // 4, 1)
    # A track without finger landmarks is compared against centroids built
    # the same way, so the grip features are absent from both sides rather
    # than defaulted on one — otherwise every imported clip reads as a fist.
    proto = prototypes(fps, win, hands is not None)

    # Overlapping windows disagree, so every window votes into the frames it
    # covers and each frame is decided afterwards. Labelling per window and
    # merging instead produces spans that overlap in time, which is not a
    # segmentation at all.
    votes = [{} for _ in range(nf)]
    hand_of = [{} for _ in range(nf)]
    speed_of = [0.0] * nf
    for lo in range(0, max(nf - win + 1, 1), stride):
        hi = min(lo + win, nf)
        feats = _window_features(frames, vels, lo, hi, hands)
        if not feats:
            continue
        hand, (f, speed) = max(feats.items(), key=lambda kv: kv[1][1])
        scores = {name: math.exp(-sum((f[i] - c[i]) ** 2 for i in range(min(len(f), len(c))))
                                 / (2 * _SIGMA2)) for name, c in proto.items()}
        tot = sum(scores.values()) or 1.0
        for fi in range(lo, hi):
            for name, s in scores.items():
                votes[fi][name] = votes[fi].get(name, 0.0) + s / tot
            hand_of[fi][hand] = hand_of[fi].get(hand, 0) + 1
            speed_of[fi] = max(speed_of[fi], speed)

    labels = []
    for fi in range(nf):
        if not votes[fi]:
            labels.append((None, 0.0, "r"))
            continue
        name, s = max(votes[fi].items(), key=lambda kv: kv[1])
        conf = s / (sum(votes[fi].values()) or 1.0)
        hand = max(hand_of[fi].items(), key=lambda kv: kv[1])[0] if hand_of[fi] else "r"
        labels.append((name if conf >= min_conf else None, conf, hand))

    spans = []
    for fi, (name, conf, hand) in enumerate(labels):
        if name is None:
            continue
        s = spans[-1] if spans else None
        if s and s["name"] == name and s["hand_k"] == hand and s["_end_i"] == fi - 1:
            s["_end_i"] = fi
            s["conf"] = max(s["conf"], conf)
            s["speed"] = max(s["speed"], speed_of[fi])
        else:
            spans.append({"name": name, "hand_k": hand, "_start_i": fi,
                          "_end_i": fi, "conf": conf, "speed": speed_of[fi]})

    # Drop flicker: a sub-min_span run is absorbed by whichever neighbour is
    # more confident, so the timeline stays a partition.
    keep = []
    for i, s in enumerate(spans):
        dur = (s["_end_i"] - s["_start_i"] + 1) / fps
        if dur >= min_span or len(spans) == 1:
            keep.append(s)
            continue
        prev = keep[-1] if keep else None
        nxt = spans[i + 1] if i + 1 < len(spans) else None
        if prev and (not nxt or prev["conf"] >= nxt["conf"]):
            prev["_end_i"] = s["_end_i"]
        elif nxt:
            nxt["_start_i"] = s["_start_i"]
        else:
            keep.append(s)

    out = []
    for s in keep:
        if out and out[-1]["name"] == s["name"] and out[-1]["hand_k"] == s["hand_k"]:
            out[-1]["_end_i"] = s["_end_i"]
            out[-1]["conf"] = max(out[-1]["conf"], s["conf"])
            out[-1]["speed"] = max(out[-1]["speed"], s["speed"])
        else:
            out.append(s)
    return [{"name": s["name"], "hand": "left" if s["hand_k"] == "l" else "right",
             "start": round(s["_start_i"] / fps, 2),
             "end": round((s["_end_i"] + 1) / fps, 2),
             "conf": round(s["conf"], 3), "speed": round(s["speed"], 3)}
            for s in out]


# --------------------------------------------------------------------------
# importing real tracked pose (VideoPose3D / BlazePose interfaces)
# --------------------------------------------------------------------------

def lift_2d_to_3d(seq2d, height=1.7):
    """Lift a 2D keypoint track to 3D. -> frames of (x, y, z) in metres.

    seq2d: [[ (x, y) x 17 joints ], ...] in normalised image coords, y up.

    The real thing is a temporal dilated conv net over a 243-frame receptive
    field. This is the geometric baseline: scale from stature, recover |z|
    per limb from the bone-length prior, disambiguate sign by continuity,
    then run the same temporal filter. Enough to drive the board, and the
    exact signature to swap a checkpoint into.
    """
    prior = {("shoulder_l", "elbow_l"): UPPER_ARM, ("elbow_l", "wrist_l"): FOREARM,
             ("shoulder_r", "elbow_r"): UPPER_ARM, ("elbow_r", "wrist_r"): FOREARM,
             ("hip_l", "knee_l"): THIGH, ("knee_l", "ankle_l"): SHIN,
             ("hip_r", "knee_r"): THIGH, ("knee_r", "ankle_r"): SHIN}
    out, prev = [], None
    for f2 in seq2d:
        span = max(max(p[1] for p in f2) - min(p[1] for p in f2), 1e-4)
        s = height / span
        pos = [[p[0] * s, p[1] * s, 0.0] for p in f2]
        for (a, b), L in prior.items():
            ia, ib = J[a], J[b]
            flat = math.hypot(pos[ib][0] - pos[ia][0], pos[ib][1] - pos[ia][1])
            dz = math.sqrt(max(L * L - flat * flat, 0.0))
            if prev is not None and prev[ib][2] < prev[ia][2]:
                dz = -dz
            pos[ib][2] = pos[ia][2] + dz
        base = min(p[1] for p in pos)
        for p in pos:
            p[1] -= base
        prev = pos
        out.append([tuple(p) for p in pos])
    return _smooth(out, 5)


def from_landmarks(seq, mapping=None):
    """Import a tracker's landmark stream. -> frames in our 17-joint order.

    `mapping` defaults to BlazePose's 33 landmarks. Landmarks may be
    (x,y,z) or (x,y,z,visibility); the fourth element is ignored.
    """
    mapping = mapping or BLAZEPOSE_MAP
    return [[tuple(f[mapping[n]][:3]) for n in JOINTS] for f in seq]


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

def plan_from_steps(steps, seconds=None):
    """How-to steps -> a timed plan of gesture primitives.

    Steps are weighted by word count so a fiddly step gets more screen time
    than "wipe your hands", then normalised to the requested duration.
    """
    steps = [s for s in (steps or []) if str(s).strip()] or ["Explain the idea"]
    weights = [max(1.0, min(3.0, len(str(s).split()) / 7.0)) for s in steps]
    total_w = sum(weights)
    seconds = seconds or _clamp(2.6 * len(steps), 6.0, 20.0)
    plan, t = [], 0.0
    for i, (s, w) in enumerate(zip(steps, weights)):
        d = seconds * w / total_w
        plan.append({"step": i + 1, "text": str(s)[:160],
                     "primitive": classify_step(s), "side": -1,
                     "start": round(t, 3), "end": round(t + d, 3)})
        t += d
    plan[-1]["end"] = round(seconds, 3)
    return plan, seconds


def build_clip(steps, seconds=None, fps=12, cam=None, source="", title="",
               constraint=False):
    """The whole pipeline. -> the clip dict the board renders.

    Everything downstream of the landmark frames is derived rather than
    authored, so an imported track produces an identically-shaped clip.

    `constraint=True` adds the Figure 9 wrap: a separate actuated curve
    bandaged around the acting forearm, following it frame by frame — the
    paper's on-body constraint replaying the recorded motion.
    """
    cam = {**DEFAULT_CAM, **(cam or {})}
    plan, seconds = plan_from_steps(steps, seconds)
    n = max(int(round(seconds * fps)), 2)

    lm_frames, hands = [], []
    for i in range(n):
        pos, hnd, _seg = _landmarks(i / fps, plan, seconds)
        lm_frames.append([pos[j] for j in JOINTS])
        hands.append(hnd)
    lm_frames = _smooth(lm_frames, 3)
    return _assemble(lm_frames, hands, plan, seconds, fps, cam,
                     source, title, constraint, "lineform-procedural")


def _assemble(lm_frames, hands, plan, seconds, fps, cam, source, title,
              constraint, backend):
    """Landmark frames -> clip. Everything here is derived, never authored,
    which is what lets a MediaPipe track and the procedural synthesiser come
    out the same shape (see clip_from_landmarks)."""
    stiff = _chain_stiffness()
    n = len(lm_frames)

    # --- the LineFORM layer: every limb becomes an actuated curve ----------
    # What crosses the wire is servo angles, not points. The board runs the
    # same forward kinematics to get the shape back, which is the paper's own
    # architecture: the model lives on the computer, angles go down the serial
    # link, and the chain resolves them into a shape. It is also 3x smaller.
    state = None
    chain_ang = {k: [] for k in CHAINS}
    chain_err = {k: 0.0 for k in CHAINS}
    chain_clip = {k: 0 for k in CHAINS}
    dt = 1.0 / fps
    for i in range(n):
        pos = {name: lm_frames[i][J[name]] for name in JOINTS}
        fits = _fit_chains(_control_paths(pos))
        if state is None:
            state = {k: [(a, 0.0) for a in fits[k]["angles"]] for k in CHAINS}
        for k, dev in CHAINS.items():
            for _ in range(max(int(lf.CONTROL_HZ * dt), 1)):
                state[k] = lf.relax(state[k], fits[k]["angles"], stiff[k],
                                    1.0 / lf.CONTROL_HZ, dev)
            chain_ang[k].append([round(math.degrees(a), 1) for a, _ in state[k]])
            chain_err[k] = max(chain_err[k], fits[k]["error"])
            chain_clip[k] = max(chain_clip[k], fits[k]["clipped"])

    vels = _velocities(lm_frames, fps)
    clip = {
        "version": 1, "kind": "skeleton", "fps": fps, "seconds": round(seconds, 2),
        "topology": "coco17", "backend": backend,
        "title": title, "source": source, "camera": cam,
        "joints": JOINTS, "bones": BONE_IDX,
        "hand_joints": HAND_JOINTS, "hand_bones": HAND_BONES,
        "frames": [[[round(v, 3) for v in j] for j in f] for f in lm_frames],
        "vectors": [[[round(v, 2) for v in j] for j in f] for f in vels],
        "hands": [{k: [[round(v, 3) for v in p] for p in pts]
                   for k, pts in h.items()} for h in hands],
        "chains": {k: {"device": {"name": CHAINS[k]["name"], "joints": CHAINS[k]["n"],
                                  "link_m": round(CHAINS[k]["link"], 4),
                                  "limit_deg": round(math.degrees(CHAINS[k]["limit"]), 1),
                                  "axes": CHAINS[k]["axes"]},
                       "stiffness": [round(s, 2) for s in stiff[k]],
                       "fit_error": round(chain_err[k], 4),
                       "clipped": chain_clip[k],
                       "angles": chain_ang[k]} for k in CHAINS},
        "flow": flow_field(lm_frames, vels, cam, fps),
        "segments": [{"step": s["step"], "text": s["text"],
                      "primitive": s["primitive"],
                      "start": s["start"], "end": s["end"]} for s in plan],
        "gestures": recognize_gestures(lm_frames, vels, fps, hands),
        "papers": PAPERS,
    }
    if constraint:
        clip["constraint"] = _constraint_track(lm_frames)
    return clip


def normalize_track(frames, height=1.7):
    """Put a tracker's output into the space the board draws in.

    BlazePose world landmarks are hip-centred metres with y pointing DOWN;
    this module wants y up and the feet on the floor at y=0. Scaling to a
    nominal stature as well means a clip looks the same whether the subject
    filled the frame or stood at the back of the room.
    """
    out = []
    for f in frames:
        pts = [[p[0], -p[1], p[2]] for p in f]           # y down -> y up
        span = max(max(p[1] for p in pts) - min(p[1] for p in pts), 1e-4)
        s = height / span
        pts = [[p[0] * s, p[1] * s, p[2] * s] for p in pts]
        cx = sum(p[0] for p in pts) / len(pts)
        base = min(p[1] for p in pts)
        out.append([(p[0] - cx, p[1] - base, p[2]) for p in pts])
    return _smooth(out, 5)


def clip_from_landmarks(frames, fps=12, cam=None, steps=None, seconds=None,
                        title="", source="", constraint=False,
                        backend="blazepose", normalize=True):
    """A real tracked body -> the same clip the synthesiser produces.

    `frames` is already in this module's 17-joint order — run it through
    from_landmarks() first if it came from BlazePose's 33. Gestures are then
    recognised from the tracked motion rather than from the text, so what the
    HUD names is what the person in the video actually did.
    """
    if len(frames) < 2:
        raise ValueError("need at least 2 tracked frames")
    lm = normalize_track(frames) if normalize else _smooth([
        [tuple(p) for p in f] for f in frames], 3)
    seconds = seconds or len(lm) / float(fps)

    if steps:
        plan, _ = plan_from_steps(steps, seconds)
    else:
        plan = [{"step": 1, "text": title or "tracked motion",
                 "primitive": "explain", "side": -1,
                 "start": 0.0, "end": round(seconds, 3)}]
    hands = [{} for _ in lm]           # body track carries no finger data
    return _assemble(lm, hands, plan, seconds, fps,
                     {**DEFAULT_CAM, **(cam or {})},
                     source, title, constraint, backend)


def _constraint_track(lm_frames):
    """The Figure 9 on-body constraint, per frame.

    An actuated curve wrapped around the acting forearm like a bandage. The
    paper uses it to record a motion and replay it onto the learner's body;
    on the board it is the visible proof that the gesture is a physical
    trajectory, not a drawing.
    """
    dev = lf.make_device(14, 0.038, 120.0, "alternating", "forearm constraint")
    out = []
    for f in lm_frames:
        elbow, wrist = f[J["elbow_r"]], f[J["wrist_r"]]
        path = lf.wrap_limb(elbow, wrist, radius=0.05, turns=2.5, samples=40, lead=0.0)
        fit = lf.fit_curve(path, dev, base=path[0],
                           heading=_chain_heading(path[:2] + [path[2]]),
                           autoscale=False)
        out.append([[round(c, 4) for c in p] for p in fit["points"]])
    return {"device": {"name": dev["name"], "joints": dev["n"],
                       "link_m": dev["link"],
                       "limit_deg": round(math.degrees(dev["limit"]), 1)},
            "attached_to": "forearm_r", "points": out,
            "paper": "LineFORM Fig. 9 — on-body constraint, record and replay"}


def text_to_motion(text, seconds=None, fps=12, cam=None, source="", title="",
                   constraint=False):
    """Text -> 3D motion, MDM's interface.

    Accepts a paragraph, a numbered how-to, or a list of steps. Set
    MOTION_BACKEND=mdm and implement `_mdm_sample` to route this to a real
    motion diffusion checkpoint; the clip contract does not change.
    """
    if isinstance(text, (list, tuple)):
        steps = [str(t) for t in text]
    else:
        lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
        numbered = [re.sub(r"^(?:step\s*)?\d+[.)]\s*|^[-*•]\s*", "", ln)
                    for ln in lines
                    if re.match(r"^(?:step\s*)?\d+[.)]|^[-*•]", ln, re.I)]
        if numbered:
            steps = numbered
        elif len(lines) > 1:
            steps = lines
        else:
            steps = [s.strip() for s in re.split(r"(?<=[.;])\s+", str(text)) if s.strip()]
    return build_clip(steps[:8], seconds=seconds, fps=fps, cam=cam,
                      source=source, title=title, constraint=constraint)


def describe(clip) -> str:
    """Clip -> a text block for prompt injection.

    This is what lets the tutor talk about the motion it is showing: which
    gesture is on screen, which hand, how fast, and how well the chains can
    physically hold the pose. Without it the skeleton is decoration the
    model cannot see.
    """
    lines = []
    if clip.get("title"):
        lines.append(f"Motion clip: {clip['title']} ({clip['seconds']}s, "
                     f"{clip['fps']}fps, {clip['topology']} skeleton driven by "
                     f"{len(clip.get('chains', {}))} actuated curve chains)")
    for s in clip.get("segments", []):
        lines.append(f"  {s['start']}-{s['end']}s  [{s['primitive']}]  {s['text']}")
    g = clip.get("gestures", [])
    if g:
        lines.append("Recognised gestures (ST-GCN-style, read back from the frames): "
                     + "; ".join(f"{x['name']} ({x['hand']} hand, {x['start']}-"
                                 f"{x['end']}s, conf {x['conf']}, peak "
                                 f"{x['speed']} m/s)" for x in g[:8]))
    ch = clip.get("chains", {})
    if ch:
        worst = max(ch.items(), key=lambda kv: kv[1]["fit_error"])
        lines.append(f"Chain fit: worst residual {worst[1]['fit_error']*100:.1f} cm on "
                     f"the {worst[0]} chain "
                     f"({worst[1]['device']['joints']} servos, "
                     f"{worst[1]['device']['link_m']*100:.1f} cm links).")
    fl = clip.get("flow", {})
    if fl:
        lines.append(f"Dense motion field: {fl['w']}x{fl['h']} vectors per frame, "
                     f"peak {fl['scale']} ndc/s")
    if clip.get("constraint"):
        lines.append("On-body constraint active: an actuated curve is wrapped around "
                     "the right forearm, replaying the recorded motion (LineFORM Fig. 9).")
    return "\n".join(lines)


if __name__ == "__main__":
    import json
    demo = ["Shift to the smallest rear cog and stop pedaling",
            "Push the rear derailleur forward to slacken the chain",
            "Seat the chain onto the bottom of the chainring teeth",
            "Rotate the pedals slowly forward until the chain seats",
            "Wipe your hands and test-ride at low speed"]
    clip = text_to_motion(demo, title="Fix a slipped bicycle chain", constraint=True)
    print(describe(clip))
    print()
    planned = [s["primitive"] for s in clip["segments"]]
    seen = [g["name"] for g in clip["gestures"]]
    print("planned :", planned)
    print("read back:", seen)
    print("frames:", len(clip["frames"]), " payload:", len(json.dumps(clip)) // 1024, "KB")
