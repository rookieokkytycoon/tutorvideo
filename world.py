"""
world.py — one shared 3D space instead of a split screen.

What this file is for
---------------------
Every other clip in this project owns its own camera: a skeleton clip is
filmed from 3.2 m, a hand clip from 62 cm, a LineFORM clip from 3 m, and
`compose` shows them at the same time by cutting the board into panes. That
works, but the panes are three separate rooms. Nothing in one pane can be
BEHIND anything in another, the diffusion footage can only ever be a flat
backdrop pasted behind all of them, and "look at that from the side" is not
a request the board can answer.

A world clip puts all of it in one room:

    ground plane   y = 0, a metre grid
    screen         a quad hanging in the room, playing the generated clip
    actors         skeleton / hand / lineform clips, each PLACED at a
                   position, yaw and scale in that same room
    camera         one camera, moving on a shot list the director wrote,
                   which the student can also drag, and which the tutor can
                   move mid-answer ("show me that from the side")

Why the paper wants this
------------------------
LineFORM's three affordances are display, interaction and CONSTRAINT, and
the third one is the one a pane cannot show:

    "whole body motion can be constrained by wrapping the actuated curve
     interface around limbs or joints like bandages so that it acts as an
     exoskeleton"                                        (LineFORM, p.7)

A wrap is a spatial relationship. In `compose` the chain and the body are
in different rectangles, so the chain can only ever be a picture of a
constraint. Here an actor can carry `bind`, and its base is re-solved every
frame from another actor's joint — so the chain is ON the forearm, in front
of the body from one angle and behind it from another, and the generated
footage is a surface in the same room that the chain can be measured
against. That is the paper's constraint, not an illustration of it.

The transform, which the browser mirrors exactly
------------------------------------------------
    p_world = at + yaw_rot(scale * (p_local - origin), yaw)

`origin` exists because a LineFORM clip is solved around a base of
(0, 0.9, 0); subtracting it lets the chain be re-based onto a moving wrist
without re-solving the fit on the server.

Nothing here re-implements a rig. Actors are ordinary clips built by
motion.text_to_motion / handform.text_to_hand / lineform.clip_from_shapes,
sharing one `seconds` and one `fps`, which is what lets a single playhead
scrub the whole room.
"""

from __future__ import annotations

import math

import handform
import lineform
import motion

KINDS = ("skeleton", "hand", "lineform")
MAX_ACTORS = 4

# The opening camera. Metres; `dist` is measured back along +z, which is
# where the viewer sits — so the wall the screen hangs on is at negative z.
DEFAULT_CAM = {"yaw": 0.38, "pitch": 0.10, "dist": 3.4,
               "target": [0.0, 1.02, 0.0], "fov": 40.0}

GROUND = {"y": 0.0, "half": 2.6, "step": 0.4}

# 16:9 at 3.4 m wide, hung at eye height on the back wall.
SCREEN = {"center": [0.0, 1.52, -1.75], "w": 3.4, "h": 1.91, "yaw": 0.0,
          "label": "generated footage"}

# Where each kind of actor stands when the director does not say. The hand
# sits at motion.WORK — the point the synthesised body reaches toward — so
# an unbound hand still lands where the work is happening.
PLACEMENT = {
    "skeleton": {"at": [0.0, 0.0, 0.0], "yaw": 0.0, "scale": 1.0,
                 "origin": [0.0, 0.0, 0.0]},
    "hand":     {"at": list(motion.WORK), "yaw": 0.0, "scale": 1.0,
                 "origin": [0.0, 0.0, 0.0]},
    "lineform": {"at": [-1.02, 0.0, 0.16], "yaw": 0.42, "scale": 1.0,
                 "origin": [0.0, 0.0, 0.0]},
}
# Alone in the room, a 18 cm hand needs to be a prop rather than anatomy.
SOLO_HAND = {"at": [0.0, 1.15, 0.15], "scale": 2.6}

# Named points each kind of actor can be pointed at, aimed at, or bound to.
ANCHORS = {
    "skeleton": ["wrist_r", "wrist_l", "elbow_r", "nose"],
    "hand": ["index_tip", "thumb_tip", "wrist"],
    "lineform": ["base", "tip", "middle"],
}


# --------------------------------------------------------------------------
# actors
# --------------------------------------------------------------------------

def _place(kind, spec, solo=False):
    """Merge the director's placement over the default for this kind."""
    p = dict(PLACEMENT[kind])
    if kind == "hand" and solo:
        p.update(SOLO_HAND)
    for key in ("at", "origin"):
        v = spec.get(key)
        if isinstance(v, (list, tuple)) and len(v) >= 3:
            try:
                p[key] = [float(v[0]), float(v[1]), float(v[2])]
            except (TypeError, ValueError):
                pass
    for key, lo, hi in (("yaw", -math.pi, math.pi), ("scale", 0.2, 6.0)):
        try:
            if spec.get(key) is not None:
                p[key] = max(lo, min(hi, float(spec[key])))
        except (TypeError, ValueError):
            pass
    return p


def _bind(spec, by_id):
    """`bind` -> a validated reference to another actor's joint, or None.

    This is the LineFORM constraint: the bound actor's base is re-solved
    from the host joint every frame, so a chain wraps a forearm that moves.
    """
    b = spec.get("bind")
    if not isinstance(b, dict):
        return None
    host = str(b.get("actor", ""))
    if host not in by_id:
        return None
    kind = by_id[host]
    joint = str(b.get("point") or b.get("joint") or ANCHORS[kind][0])
    if joint not in ANCHORS[kind]:
        joint = ANCHORS[kind][0]
    off = b.get("offset")
    try:
        off = [float(off[0]), float(off[1]), float(off[2])]
    except (TypeError, ValueError, IndexError):
        off = [0.0, 0.0, 0.0]
    return {"actor": host, "point": joint, "offset": off}


def build_actor(spec, seconds, fps, question="", resolve=None, solo=False):
    """One actor -> (actor dict, error message).

    The clip inside is an ordinary clip of its kind. `resolve` is the
    hivemind lookup (question -> (steps, title)); passing it in keeps this
    module free of the graph.
    """
    kind = spec.get("kind")
    if kind not in KINDS:
        return None, f"unknown actor kind {kind!r}"
    title = str(spec.get("title", ""))[:120]
    source = ""

    if kind == "lineform":
        device = spec.get("device") if spec.get("device") in lineform.DEVICES \
            else "large"
        shapes = [s for s in (spec.get("shapes") or ["curve", "wristband"])
                  if s in lineform.SHAPES][:6] or ["curve"]
        clip = lineform.clip_from_shapes(
            shapes, seconds=seconds, fps=fps, device=device,
            snap=bool(spec.get("snap")), values=spec.get("values"), title=title)
    else:
        steps = spec.get("steps") or spec.get("text") or ""
        if not steps and (spec.get("question") or question) and resolve:
            steps, found = resolve(spec.get("question") or question)
            if not steps:
                return None, "the hivemind has no repair for that yet"
            title, source = title or found, "hivemind"
        if not steps:
            return None, f"{kind} actor needs steps, text or question"
        steps = [str(s)[:200] for s in steps[:10]] if isinstance(steps, list) \
            else str(steps)[:4000]
        if kind == "hand":
            clip = handform.text_to_hand(steps, seconds=seconds, fps=fps,
                                         title=title, source=source)
        else:
            clip = motion.text_to_motion(steps, seconds=seconds, fps=fps,
                                         title=title, source=source,
                                         constraint=bool(spec.get("constraint")))
            # The optical-flow field is quantised in the clip's OWN camera's
            # normalised device space. Under a camera that moves it would be
            # a field pinned to the wrong frame of reference, so it is
            # dropped rather than drawn wrong — and it is most of the bytes.
            clip.pop("flow", None)
    actor = {"id": str(spec.get("id") or kind)[:40], "kind": kind,
             "title": title, "clip": clip}
    actor.update(_place(kind, spec, solo=solo))
    return actor, None


# --------------------------------------------------------------------------
# the shot list
# --------------------------------------------------------------------------

# A step decides how close the camera gets: twisting a cap is a finger shot,
# lifting a wheel is a whole-body shot. Both tables are keyed by labels the
# clip already carries — motion.classify_step's primitive for a body, the
# hand pose for fingers — so the cutting comes out of the how-to text rather
# than a fixed rhythm.
PRIMITIVE_SHOT = {                      # skeleton segments
    "grasp": "close", "twist": "close", "tap": "close", "cut": "close",
    "pull": "medium", "push": "medium", "crank": "medium", "pour": "medium",
    "wipe": "medium", "lift": "wide",
    "point": "wide", "inspect": "wide", "explain": "wide",
}
POSE_SHOT = {                           # hand segments — fingers want closer
    "pinch": "close", "ok": "close", "tripod": "close", "grip": "close",
    "gun": "close", "point": "close",
    "fist": "medium", "thumbs_up": "medium", "peace": "medium",
    "flat": "medium", "open_palm": "medium", "spread": "medium",
}


def _shot(kind, actors, screen):
    """A named shot -> camera. Targets may name a live joint, in which case
    the browser resolves them per frame and the camera follows it."""
    hand = next((a for a in actors if a["kind"] == "hand"), None)
    curve = next((a for a in actors if a["kind"] == "lineform"), None)
    body = next((a for a in actors if a["kind"] == "skeleton"), None)

    if kind == "close":
        tgt = ({"actor": hand["id"], "point": "index_tip"} if hand else
               {"actor": body["id"], "point": "wrist_r"} if body else
               [0.16, 1.14, 0.34])
        return {"yaw": 0.30, "pitch": 0.05, "dist": 0.78, "target": tgt}
    if kind == "medium":
        tgt = ({"actor": body["id"], "point": "elbow_r"} if body else
               [0.0, 1.10, 0.10])
        return {"yaw": 0.72, "pitch": 0.06, "dist": 1.70, "target": tgt}
    if kind == "side":
        return {"yaw": 1.24, "pitch": 0.05, "dist": 2.60,
                "target": [0.0, 1.05, 0.10]}
    if kind == "curve" and curve:
        c = curve["at"]
        return {"yaw": 0.10, "pitch": 0.08, "dist": 1.90,
                "target": [c[0], c[1] + 0.92, c[2]]}
    if kind == "screen" and screen:
        c = screen["center"]
        return {"yaw": 0.06, "pitch": 0.02, "dist": 3.10,
                "target": [c[0], c[1] - 0.15, c[2] + 0.9]}
    return {"yaw": 0.38, "pitch": 0.10, "dist": 3.40, "target": [0.0, 1.02, 0.0]}


def plan_shots(actors, screen, seconds):
    """Actors -> a camera track keyed to the steps they are performing.

    One shot per step of the leading actor, chosen by that step's motion
    primitive, opening and closing wide, with a look at the generated
    footage inserted once if there is any. `t` is when the move COMPLETES;
    `move` is how long it takes, so the cut is a camera move, not a jump.
    """
    lead = next((a for a in actors if a["kind"] == "hand"),
                next((a for a in actors if a["kind"] == "skeleton"), None))
    segs = (lead["clip"].get("segments") if lead else None) or []

    table = POSE_SHOT if lead and lead["kind"] == "hand" else PRIMITIVE_SHOT
    plan = [("wide", 0.0)]
    for s in segs:
        kind = table.get(s.get("primitive") or s.get("pose") or "", "medium")
        start = float(s.get("start", 0.0))
        if start < 0.6:                     # the opening shot already covers it
            continue
        plan.append((kind, start))
    if screen and seconds > 6:
        plan.append(("screen", seconds * 0.62))
    if any(a["kind"] == "lineform" for a in actors) and seconds > 8:
        plan.append(("curve", seconds * 0.80))
    if seconds > 5:
        plan.append(("wide", max(seconds - 1.6, 1.0)))

    plan.sort(key=lambda x: x[1])
    shots, last_t, last_kind = [], -9.0, None
    for kind, t in plan:
        if t - last_t < 1.4 or kind == last_kind:      # no strobing cuts
            continue
        cam = _shot(kind, actors, screen)
        cam.update({"t": round(t, 2), "label": kind,
                    "move": 0.0 if not shots else 1.15, "fov": 40.0})
        shots.append(cam)
        last_t, last_kind = t, kind
    return shots


# --------------------------------------------------------------------------
# the world clip
# --------------------------------------------------------------------------

def build_world(specs, seconds=12.0, fps=12, question="", title="",
                screen=True, stage=None, shots=None, resolve=None):
    """Actor specs -> the world clip the board renders. Raises ValueError.

    A hand and a body in the same world are bound together by default: the
    hand IS the body's right hand, so the close shot pushes into the same
    anatomy the wide shot showed, instead of into a second, unrelated hand
    floating beside it.
    """
    specs = [s for s in specs if isinstance(s, dict)][:MAX_ACTORS]
    if not specs:
        raise ValueError("at least one actor is required")

    seen, by_id = set(), {}
    for s in specs:                          # ids must be unique to bind by
        k = s.get("kind")
        base = str(s.get("id") or k or "actor")[:40]
        i, name = 2, base
        while name in seen:
            name, i = f"{base}{i}", i + 1
        s["id"], _ = name, seen.add(name)
        if k in KINDS:
            by_id[name] = k

    solo_hand = not any(s.get("kind") == "skeleton" for s in specs)
    actors = []
    for s in specs:
        actor, err = build_actor(s, seconds, fps, question, resolve,
                                 solo=solo_hand)
        if err:
            raise ValueError(err)
        actors.append(actor)

    body = next((a for a in actors if a["kind"] == "skeleton"), None)
    for a, s in zip(actors, specs):
        b = _bind(s, {i: k for i, k in by_id.items() if i != a["id"]})
        if b is None and body and a["kind"] == "hand" and s.get("bind") != False:
            # the default union: fingers are the body's own right hand
            b = {"actor": body["id"], "point": "wrist_r", "offset": [0, 0, 0]}
        if b is None and body and a["kind"] == "lineform" and s.get("wrap"):
            # the paper's Fig. 9 constraint, re-based onto the live forearm
            b = {"actor": body["id"], "point": "elbow_r", "offset": [0, 0, 0]}
            a["origin"] = [0.0, 0.9, 0.0]
        if b:
            a["bind"] = b
            if a["kind"] == "lineform":
                a["origin"] = [0.0, 0.9, 0.0]
    if body and any(a.get("bind", {}).get("point") == "wrist_r" and
                    a["kind"] == "hand" for a in actors):
        body["hide_hands"] = True            # the detailed hand replaces the stub

    stage_out = {"ground": dict(GROUND)}
    scr = None
    if screen:
        scr = dict(SCREEN)
        if isinstance(stage, dict) and isinstance(stage.get("screen"), dict):
            for k in ("w", "h", "yaw", "label"):
                if stage["screen"].get(k) is not None:
                    scr[k] = stage["screen"][k]
            c = stage["screen"].get("center")
            if isinstance(c, (list, tuple)) and len(c) >= 3:
                scr["center"] = [float(v) for v in c[:3]]
        stage_out["screen"] = scr

    track = shots if isinstance(shots, list) and shots else \
        plan_shots(actors, scr, seconds)
    papers = sorted({p for a in actors
                     for p in (a["clip"].get("papers") or
                               ([a["clip"]["paper"]] if a["clip"].get("paper") else []))})
    return {
        "version": 1, "kind": "world", "fps": fps,
        "seconds": round(float(seconds), 2), "title": title,
        "source": next((a["clip"].get("source") for a in actors
                        if a["clip"].get("source")), ""),
        "camera": {**DEFAULT_CAM,
                   **{k: v for k, v in (track[0] if track else {}).items()
                      if k in ("yaw", "pitch", "dist", "fov")}},
        "stage": stage_out, "shots": track,
        "actors": [{k: v for k, v in a.items()} for a in actors],
        "anchors": {a["id"]: ANCHORS[a["kind"]] for a in actors},
        "papers": papers,
    }


def describe(clip) -> str:
    """World clip -> text for prompt injection, so the tutor can talk about
    the room it is standing in rather than narrating over it blindly."""
    out = [f"A single 3D room, {clip['seconds']}s, one camera."]
    for a in clip.get("actors", []):
        c = a["clip"]
        where = ", ".join(f"{v:+.2f}" for v in a["at"])
        line = (f"- {a['id']} ({a['kind']}) at ({where})"
                + (f" x{a['scale']:.1f}" if a["scale"] != 1.0 else ""))
        if a.get("bind"):
            line += f", bound to {a['bind']['actor']}/{a['bind']['point']}"
        if c.get("segments"):
            line += " — " + "; ".join(s.get("text", "")[:40]
                                      for s in c["segments"][:3])
        elif c.get("labels"):
            line += " — holding " + " -> ".join(x for x in c["labels"] if x)
        out.append(line)
    if clip.get("stage", {}).get("screen"):
        s = clip["stage"]["screen"]
        out.append(f"- a {s['w']:.1f}m screen on the back wall playing the "
                   "generated footage")
    if clip.get("shots"):
        out.append("camera: " + " -> ".join(
            f"{s['label']}@{s['t']}s" for s in clip["shots"]))
    return "\n".join(out)


if __name__ == "__main__":
    w = build_world([{"kind": "skeleton",
                      "steps": ["Grip the handle firmly",
                                "Twist it counter-clockwise"]},
                     {"kind": "hand",
                      "steps": ["Pinch the chain between thumb and finger",
                                "Place the chain onto the chainring teeth"]},
                     {"kind": "lineform", "shapes": ["curve", "wristband"],
                      "wrap": True}],
                    seconds=12.0)
    print(describe(w))
    print("shots:", [(s["label"], s["t"]) for s in w["shots"]])
