"""
coach.py — the rig stops performing and starts watching.

Everything else in this project points one way: text becomes a motion, the
motion becomes a clip, the clip plays. That is a demonstration, and a
demonstration is only half of what the paper is after —

    "it can also record motion and replay back on your body ... to learn
     kinesthetic motion such as sports and dances as an external motor
     memory or to provide physical feedforward and guidance for gestural
     interaction"                                        (LineFORM, Fig. 9)

Guidance needs the other direction. This module closes it: a synthesised or
tracked clip is reduced to a small set of measured joint angles per step, a
live body's landmarks are reduced the same way, and the difference between
the two is a correction in words.

    reference clip  ->  reference(clip)  ->  per-step target angles + tolerance
    live landmarks  ->  angles_of(frame) ->  the same nine numbers
    the two together ->  score()         ->  0-100 and "straighten your right elbow"

Why the angles and not the points
---------------------------------
Landmark positions depend on where the student stands, how tall they are and
which way the camera points. Joint angles do not. Nine angles compare a 1.9 m
adult filmed from the side against a 1.5 m one filmed head-on, with no
calibration step and no asking anybody to stand on a mark. It is also the
representation motion.py already produces, so the reference for a coaching
session is any clip the rest of the codebase can make: synthesised from a
how-to, retrieved from the hivemind, or tracked off a real video with
posetrack.py — the coach does not know or care which.

The browser scores every frame itself (30 fps over HTTP would be absurd), so
`client_spec()` ships the joint triples and the phrasing table rather than
letting the page invent its own. The arithmetic — the angle at b between
b->a and b->c — is three lines on either side; the *definitions* live here,
once.

    from coach import reference, angles_of, score
    ref = reference(motion.text_to_motion("Squat down, keeping your back flat"))
    score(angles_of(live_frame), ref["steps"][0])
    -> {"score": 71, "cues": [{"text": "straighten your right knee ...
"""

from __future__ import annotations

import math

from motion import J, JOINTS, BONE_IDX, BLAZEPOSE_MAP

# --------------------------------------------------------------------------
# what gets measured
# --------------------------------------------------------------------------
# name -> (a, b, c): the angle at b, between b->a and b->c, in degrees.
# Eight of them, four per side, chosen because they are the joints a person
# can actually be told to change. There is no wrist angle here: the body
# track has no fingers (handform.py owns those) and "rotate your wrist 12
# degrees" is not a coachable instruction anyway.
ANGLES = {
    "elbow_r":    ("shoulder_r", "elbow_r", "wrist_r"),
    "elbow_l":    ("shoulder_l", "elbow_l", "wrist_l"),
    "shoulder_r": ("hip_r", "shoulder_r", "elbow_r"),
    "shoulder_l": ("hip_l", "shoulder_l", "elbow_l"),
    "hip_r":      ("shoulder_r", "hip_r", "knee_r"),
    "hip_l":      ("shoulder_l", "hip_l", "knee_l"),
    "knee_r":     ("hip_r", "knee_r", "ankle_r"),
    "knee_l":     ("hip_l", "knee_l", "ankle_l"),
}

# The ninth is not a joint triple: the trunk's angle away from vertical.
# It is the single most common fault in every physical how-to there is
# ("keep your back straight"), and no three landmarks encode it — it is
# hip-midpoint to shoulder-midpoint against world up.
TORSO = "torso_lean"

MEASURES = list(ANGLES) + [TORSO]

# Phrasing, per measure: what to say when the live angle is LARGER than the
# target, what to say when it is smaller, and a human name for the readout.
# A bigger elbow angle is a straighter arm, so "too big" means "bend it" —
# the direction flips per joint, which is exactly why this is a table and
# not a sign convention.
CUES = {
    "elbow_r":    ("bend your right elbow more", "straighten your right elbow",
                   "right elbow"),
    "elbow_l":    ("bend your left elbow more", "straighten your left elbow",
                   "left elbow"),
    "shoulder_r": ("bring your right arm in closer to your body",
                   "lift your right arm higher", "right shoulder"),
    "shoulder_l": ("bring your left arm in closer to your body",
                   "lift your left arm higher", "left shoulder"),
    "hip_r":      ("hinge further forward at the right hip",
                   "open your right hip, stand taller", "right hip"),
    "hip_l":      ("hinge further forward at the left hip",
                   "open your left hip, stand taller", "left hip"),
    "knee_r":     ("bend your right knee more", "straighten your right knee",
                   "right knee"),
    "knee_l":     ("bend your left knee more", "straighten your left knee",
                   "left knee"),
    TORSO:        ("stand taller — your back is leaning too far over",
                   "lean forward a little more from the hips", "torso"),
}

# A student facing the camera mirrors the demonstration by instinct: they
# copy the arm they SEE moving, which is the opposite one. Scoring can swap
# sides to accept that instead of scolding them for it.
MIRROR = {}
for _n in MEASURES:
    if _n.endswith("_r"):
        MIRROR[_n] = _n[:-2] + "_l"
    elif _n.endswith("_l"):
        MIRROR[_n] = _n[:-2] + "_r"
    else:
        MIRROR[_n] = _n

# Scoring shape. An error inside the tolerance band costs nothing; beyond it,
# the score falls off linearly and hits zero SPAN degrees out, so one badly
# wrong joint cannot be hidden by seven correct ones.
# A step is a MOVEMENT, so its target is a band, not a number: the interval
# the angle actually passes through while the reference performs that step,
# widened by a tolerance. Scoring against the step's mean pose instead would
# mark down a student copying the demonstration exactly, at every instant
# except the middle of the swing — the elbow is meant to travel 60 degrees,
# so 60 degrees of travel cannot be the error. Inside the band costs nothing;
# outside it, the error is the distance to the nearer edge.
SPAN = 45.0
KEY_WEIGHT = 2.0
# A plain weighted mean is too kind: eight correct angles drown out one
# badly wrong one, and "76/100, and by the way your arm is in completely the
# wrong place" is not coaching. The mean is therefore pulled down by the
# WORST of the step's key measures — perfect on that one costs nothing,
# hopeless on it caps the score at 55% of the mean.
WORST_PULL = 0.45
# The band already absorbs the step's own motion, so the tolerance on top of
# it only has to cover tracker noise and the difference between two bodies.
TOL_MIN, TOL_MAX = 8.0, 18.0
KEY_N = 4                      # how many measures a step is said to be "about"
PASS = 78                      # score that counts as the step performed
HOLD = 1.6                     # seconds it must be held before moving on


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _mid(a, b):
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5)


def _len(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def _angle_at(a, b, c):
    """Degrees at b, between b->a and b->c. Degenerate limbs read 180."""
    u, v = _sub(a, b), _sub(c, b)
    nu, nv = _len(u), _len(v)
    if nu < 1e-9 or nv < 1e-9:
        return 180.0
    cos = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (nu * nv)
    return math.degrees(math.acos(_clamp(cos, -1.0, 1.0)))


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs, m=None):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs) if m is None else m
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def angles_of(frame):
    """One pose -> the nine measured angles, in degrees.

    `frame` is 17 (x, y, z) points in motion.JOINTS order — whatever produced
    it. Run a BlazePose stream through motion.from_landmarks() first.
    """
    p = [tuple(q[:3]) for q in frame]
    out = {n: round(_angle_at(p[J[a]], p[J[b]], p[J[c]]), 1)
           for n, (a, b, c) in ANGLES.items()}
    trunk = _sub(_mid(p[J["shoulder_l"]], p[J["shoulder_r"]]),
                 _mid(p[J["hip_l"]], p[J["hip_r"]]))
    n = _len(trunk)
    out[TORSO] = round(math.degrees(math.acos(_clamp(trunk[1] / n, -1.0, 1.0)))
                       if n > 1e-9 else 0.0, 1)
    return out


def mirror_angles(a):
    """Left/right swapped, for a student copying the demonstration facing it."""
    return {MIRROR[n]: v for n, v in a.items() if n in MIRROR}


# --------------------------------------------------------------------------
# a clip becomes something to be judged against
# --------------------------------------------------------------------------

def reference(clip, key_n=KEY_N):
    """Clip -> per-step target bands. The thing to copy.

    Each measure gets the interval it travels through during that step, plus
    a tolerance for tracker noise and body differences. A step that sweeps
    the elbow through 60 degrees therefore accepts the whole sweep, while a
    step that holds a position accepts almost nothing — the band is derived
    from the reference's own motion, per step per angle, rather than being a
    constant somewhere.

    The `key` list is what the step is ABOUT — the measures that either move
    during it or sit furthest from the clip's own average posture. They are
    weighted double when scoring, and they are what the readout names, so a
    step about the knees does not get graded down for an idle elbow.
    """
    frames = clip.get("frames") or []
    if len(frames) < 2:
        raise ValueError("reference needs a clip with at least 2 frames")
    fps = float(clip.get("fps") or 12)
    seconds = float(clip.get("seconds") or len(frames) / fps)
    segs = clip.get("segments") or [{
        "step": 1, "text": clip.get("title") or "hold the position",
        "primitive": "explain", "start": 0.0, "end": seconds}]

    per_frame = [angles_of(f) for f in frames]
    overall = {n: _mean([a[n] for a in per_frame]) for n in MEASURES}

    steps = []
    for s in segs:
        i0 = int(_clamp(round(float(s["start"]) * fps), 0, len(frames) - 1))
        i1 = int(_clamp(round(float(s["end"]) * fps), i0 + 1, len(frames)))
        win = per_frame[i0:i1] or [per_frame[i0]]
        targets, spread = {}, {}
        for n in MEASURES:
            vals = [a[n] for a in win]
            m = _mean(vals)
            sd = _std(vals, m)
            spread[n] = sd
            targets[n] = {"lo": round(min(vals), 1), "hi": round(max(vals), 1),
                          "deg": round(m, 1),
                          "tol": round(_clamp(0.5 * sd + 6.0, TOL_MIN, TOL_MAX), 1)}
        rank = sorted(MEASURES,
                      key=lambda n: -(abs(targets[n]["deg"] - overall[n]) + spread[n]))
        steps.append({
            "step": int(s.get("step", len(steps) + 1)),
            "text": str(s.get("text", ""))[:160],
            "primitive": s.get("primitive", "explain"),
            "start": round(float(s["start"]), 3),
            "end": round(float(s["end"]), 3),
            "targets": targets,
            "key": rank[:max(1, key_n)],
        })

    # motion.classify_step falls back to "explain" — a talking gesture — for
    # any text with no action verb in it. A reference made entirely of those
    # is a body standing still waving its hands, and scoring somebody against
    # it would be a number with nothing behind it. Say so rather than grade it.
    physical = any(s["primitive"] != "explain" for s in steps)

    return {
        "version": 1,
        "kind": "coach",
        "physical": physical,
        "title": clip.get("title", ""),
        "source": clip.get("source", ""),
        "backend": clip.get("backend", ""),
        "seconds": round(seconds, 2),
        "pass": PASS,
        "hold": HOLD,
        "steps": steps,
        **client_spec(),
    }


def client_spec():
    """The definitions the browser needs to compute the same nine numbers.

    Shipped with every reference so the page never hard-codes a joint triple
    or a phrasing. Add a measure here and the live overlay picks it up with
    no change on the other side.
    """
    return {
        "joints": JOINTS,
        "bones": BONE_IDX,
        # the page's tracker speaks BlazePose's 33 landmarks; this is the same
        # decimation posetrack.py applies server-side, so the browser overlay
        # and an offline tracked clip are the identical seventeen points
        "blazepose": BLAZEPOSE_MAP,
        "angles": {n: list(t) for n, t in ANGLES.items()},
        "torso": TORSO,
        "measures": MEASURES,
        "mirror": MIRROR,
        "cues": {n: {"over": o, "under": u, "label": l}
                 for n, (o, u, l) in CUES.items()},
        "span": SPAN,
        "key_weight": KEY_WEIGHT,
        "worst_pull": WORST_PULL,
    }


# --------------------------------------------------------------------------
# judging
# --------------------------------------------------------------------------

def _off_band(v, t):
    """Signed distance from a target band. Zero anywhere inside it.

    Positive means the angle is too big — which joint-by-joint may mean too
    straight or too far from the body; CUES owns that translation.
    """
    tol = float(t.get("tol", TOL_MIN))
    lo = float(t.get("lo", t.get("deg", 0.0))) - tol
    hi = float(t.get("hi", t.get("deg", 0.0))) + tol
    if v < lo:
        return v - lo
    if v > hi:
        return v - hi
    return 0.0


def _cue(name, delta):
    over, under, label = CUES[name]
    return {"angle": name, "label": label,
            "text": over if delta > 0 else under,
            # floor(x + .5), not round(): Python rounds halves to even and
            # JavaScript rounds them up, so an error of exactly 16.5 degrees
            # would be reported as 16 in the tutor's answer and 17 on the
            # overlay drawn beside it. Same number, both sides.
            "off": int(math.floor(abs(delta) + 0.5))}


def score(live, step, mirror=False):
    """One live pose against one step's targets. -> score, deltas, corrections.

    `live` is the output of angles_of(); `step` is one entry of
    reference()["steps"]. Missing measures are skipped rather than counted
    as zero, so a partly-visible body still gets a usable score instead of
    a punishing one.
    """
    if mirror:
        live = mirror_angles(live)
    keys = set(step.get("key") or [])
    tot = wtot = 0.0
    worst_key = 1.0
    deltas, errs = {}, []
    for n, t in (step.get("targets") or {}).items():
        if n not in live:
            continue
        d = _off_band(float(live[n]), t)
        e = abs(d)
        w = KEY_WEIGHT if n in keys else 1.0
        s = max(0.0, 1.0 - e / SPAN)
        tot += w * s
        wtot += w
        if n in keys:
            worst_key = min(worst_key, s)
        deltas[n] = round(d, 1)
        if e > 0:
            errs.append((w * e, n, d))
    if not wtot:
        return {"score": 0, "deltas": {}, "cues": [], "worst": None,
                "ok": False, "seen": False}
    errs.sort(key=lambda x: -x[0])
    sc = int(round(100.0 * (tot / wtot) * (1.0 - WORST_PULL * (1.0 - worst_key))))
    return {"score": sc, "deltas": deltas,
            "cues": [_cue(n, d) for _, n, d in errs[:2]],
            "worst": errs[0][1] if errs else None,
            "ok": sc >= PASS, "seen": True}


def describe(ref, step_i=0, last=None):
    """Live coaching state -> a text block for prompt injection.

    Without this, a student who interrupts with "what am I doing wrong?"
    is asking a tutor that cannot see them; the board is a webcam feed the
    model has no access to. With it, the same nine numbers the overlay is
    drawing go into the prompt, and the answer is about the actual fault.
    """
    steps = ref.get("steps") or []
    if not steps:
        return ""
    s = steps[max(0, min(step_i, len(steps) - 1))]
    lines = [f"LIVE COACHING — the student's camera is on and I am scoring "
             f"their form against \"{ref.get('title') or 'the reference motion'}\"",
             f"  current step {s['step']}/{len(steps)}: {s['text']}",
             "  this step is judged on: "
             + ", ".join(CUES[n][2] for n in s.get("key", []))]
    if last and last.get("seen"):
        lines.append(f"  their score right now: {last.get('score', 0)}/100")
        for c in last.get("cues") or []:
            lines.append(f"  fault: {c['label']} is {c['off']} degrees off "
                         f"target — \"{c['text']}\"")
        if not (last.get("cues") or []):
            lines.append("  no fault right now — they are inside tolerance "
                         "on every measured angle")
    else:
        lines.append("  I cannot see them in frame at the moment")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# the session, after the fact
# --------------------------------------------------------------------------

def report(ref, samples, mirror=False):
    """A session's samples -> what actually happened, per step and overall.

    Samples are {"step": i, "t": seconds, "angles": {...}} — the browser
    records one every few frames, not every frame, because the summary wants
    a trend and not a 30 Hz log. Scoring them here rather than trusting the
    numbers the page computed means the verdict comes from the same code the
    reference did, whatever the page did on screen.
    """
    steps = ref.get("steps") or []
    if not steps:
        return {"error": "reference has no steps"}
    buckets = [[] for _ in steps]
    for s in samples or []:
        try:
            i = int(s.get("step", 0))
            a = s.get("angles") or {}
        except (TypeError, ValueError):
            continue
        if 0 <= i < len(steps) and a:
            buckets[i].append(score({k: float(v) for k, v in a.items()
                                     if isinstance(v, (int, float))},
                                    steps[i], mirror=mirror))

    faults, out, scored = {}, [], []
    for i, (st, got) in enumerate(zip(steps, buckets)):
        seen = [g for g in got if g["seen"]]
        if not seen:
            out.append({"step": st["step"], "text": st["text"], "samples": 0,
                        "best": None, "mean": None, "in_range": 0.0,
                        "fault": None,
                        "verdict": "not attempted — you were not in frame"})
            continue
        vals = [g["score"] for g in seen]
        best, mean = max(vals), _mean(vals)
        inr = sum(1 for g in seen if g["ok"]) / float(len(seen))
        f = {}
        for g in seen:
            for c in g["cues"]:
                f[c["angle"]] = f.get(c["angle"], 0) + 1
        worst = max(f, key=lambda k: f[k]) if f else None
        if worst:
            faults[worst] = faults.get(worst, 0) + f[worst]
        out.append({
            "step": st["step"], "text": st["text"], "samples": len(seen),
            "best": best, "mean": int(round(mean)),
            "in_range": round(inr, 2),
            "fault": CUES[worst][2] if worst else None,
            "verdict": ("clean — held it inside tolerance most of the way"
                        if inr >= 0.6 else
                        "got there — but only briefly" if best >= PASS else
                        f"not there yet — {CUES[worst][2]} was the sticking point"
                        if worst else "not there yet"),
        })
        scored.append(mean)

    done = [s for s in out if s["samples"]]
    overall = int(round(_mean(scored))) if scored else 0
    passed = sum(1 for s in out if s["best"] is not None and s["best"] >= PASS)
    top = max(faults, key=lambda k: faults[k]) if faults else None
    return {
        "kind": "coach-report",
        "title": ref.get("title", ""),
        "overall": overall,
        "steps_passed": passed,
        "steps_total": len(steps),
        "steps_attempted": len(done),
        "work_on": CUES[top][2] if top else None,
        "steps": out,
        "verdict": _verdict(overall, passed, len(steps), top),
    }


def _verdict(overall, passed, total, top):
    if not passed and not overall:
        return ("Nothing to grade — the camera never got a clear look at you. "
                "Stand back so your whole body is in frame and run it again.")
    head = (f"{passed} of {total} steps hit the mark, averaging {overall}/100")
    if overall >= 85:
        tail = "That is solid form. Speed it up and keep the same shape."
    elif overall >= PASS:
        tail = "Good — the shape is right, it just needs to be repeatable."
    elif overall >= 55:
        tail = "The movement is there but the positions drift."
    else:
        tail = "Slow it right down; accuracy first, then pace."
    if top:
        tail += f" One thing to work on: your {CUES[top][2]}."
    return head + ". " + tail


if __name__ == "__main__":                       # python coach.py "squat down"
    import json
    import sys

    import motion
    text = " ".join(sys.argv[1:]) or "Squat down keeping your back flat, then stand"
    ref = reference(motion.text_to_motion(text, seconds=8))
    print(json.dumps({"steps": [{"text": s["text"], "key": s["key"]}
                                for s in ref["steps"]]}, indent=2))
    # a body that is simply standing still, judged against step 1
    rest = [motion.REST[n] for n in JOINTS]
    print(json.dumps(score(angles_of(rest), ref["steps"][0]), indent=2))
