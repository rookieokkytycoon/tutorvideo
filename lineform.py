"""
lineform.py — actuated curve interface, after Nakagaki, Follmer & Ishii,
"LineFORM: Actuated Curve Interfaces for Display, Interaction, and
Constraint", UIST '15.

Why this file exists
--------------------
The tutor needed a physically honest way to animate 3D vector data. LineFORM
supplies it: a serial chain of 1-DOF servos whose axes alternate (X after Y)
so the chain can leave the plane and form 3D structure. Everything the board
draws — a CAD outline, a data curve, a bone chain, an iconic form — is the
same thing here: a 3D polyline that the chain tries to become.

The paper's control algorithm is the heart of it (p.4):

    "we extract outline from binary image data as series of vectors, then
     calculate the angle for each servo motor according to length of each
     joint ... this is a slightly different, and more simple, task than the
     inverse kinematic control needed to move a serpentine robot towards a
     goal, as we do not care about the position of the end effector"

So `fit_curve` resamples a target path at the joint pitch and solves one
angle per joint — no end-effector IK. Because each joint has ONE degree of
freedom, the chain generally cannot pass through every point of an arbitrary
3D curve; `fit_curve` returns the shape it can actually reach plus the
residual error. That limitation is the physics, not a bug, and animating the
reachable shape is what makes this read as a real device instead of a spline.

What is modelled
----------------
- forward kinematics of the alternating-axis chain           (paper, p.5)
- curve fitting from vector data, one angle per joint        (paper, p.4)
- per-joint stiffness and the mass-spring settle             (paper, p.4)
- snap-to-grid / right-angle constraint                      (paper, Fig. 11)
- display primitives: Curve, Surface, Solid                  (paper, p.3)
- Data Representations, Iconic Forms, UI Elements            (paper, p.3)
- on-body constraint: wrap around a limb and replay motion   (paper, Fig. 9)
- both published prototypes' real dimensions                 (paper, p.5)

Not modelled: torque, power, collision between servo housings — the paper
lists collision prediction as future work and so is it.

    from lineform import DEVICES, fit_curve, clip_from_shapes, iconic
    clip = clip_from_shapes(["curve", "phone", "wristband"], seconds=9)
"""

from __future__ import annotations

import math

# --------------------------------------------------------------------------
# the two published prototypes (paper, p.5)
# --------------------------------------------------------------------------

DEVICES = {
    # 28 Dynamixel AX-18A, 7 cm per joint incl. bracket, 186 cm total,
    # 206 deg range, axes alternately perpendicular -> 3D structure.
    "large": {"name": "LineFORM 3D (large)", "n": 28, "link": 0.066,
              "limit": math.radians(103.0), "axes": "alternating",
              "servo": "Dynamixel AX-18A", "torque_control": True,
              "deform_sensing": True, "total_m": 1.86},
    # 21 HS-5035HD nano servos, ~2.4 cm per joint, 47 cm total, 232 deg
    # range, single axis -> planar only, but higher resolution.
    "small": {"name": "LineFORM 2D (small)", "n": 21, "link": 0.0224,
              "limit": math.radians(116.0), "axes": "single",
              "servo": "Hitec HS-5035HD", "torque_control": False,
              "deform_sensing": False, "total_m": 0.47},
}
DEFAULT_DEVICE = "large"


def make_device(n, link, limit_deg=103.0, axes="alternating",
                name="actuated curve", servo="Dynamixel AX-12A"):
    """A chain spec that is not one of the two published prototypes.

    The paper's own future work asks for "different sections of actuated
    curve interfaces" connected together; a skeleton built from limb-sized
    chains is that, so every function here takes either a DEVICES key or a
    spec dict like this one.

    `link` may be a single length or a list of per-joint lengths, and
    `limit_deg` likewise. Both prototypes in the paper are uniform, but a
    finger is not: its metacarpal, proximal, middle and distal phalanges all
    differ, and its knuckles have different ranges. Forcing a finger onto a
    uniform chain would charge anatomy to the fit residual, where it would
    read as the hardware failing to hold a shape it was never asked to.
    """
    links = [float(x) for x in link] if isinstance(link, (list, tuple)) \
        else [float(link)] * int(n)
    lims = [math.radians(x) for x in limit_deg] \
        if isinstance(limit_deg, (list, tuple)) else \
        [math.radians(limit_deg)] * int(n)
    return {"name": name, "n": int(n), "link": sum(links) / len(links),
            "links": links, "limit": max(lims), "limits": lims,
            "axes": axes, "servo": servo,
            "torque_control": True, "deform_sensing": True,
            "total_m": round(sum(links), 4)}


def _dev(device):
    """Accept a DEVICES key or a spec dict, uniformly."""
    return device if isinstance(device, dict) else DEVICES[device]


def _links(d):
    """Per-joint link lengths, broadcasting the uniform case."""
    return d.get("links") or [d["link"]] * d["n"]


def _limits(d):
    """Per-joint angle limits in radians, broadcasting the uniform case."""
    return d.get("limits") or [d["limit"]] * d["n"]

# The Processing model in the paper runs at 60 fps; the board only needs
# enough frames to look continuous once they cross the wire.
CONTROL_HZ = 60

# Base frame as (right, up, forward). The chain leaves the base pointing +X
# and every joint's axis is one of `right`/`up`.
#
# This orientation is load-bearing for the planar prototype. Its motors all
# share one axis, and rotating about `right` never changes `right` — so the
# whole chain is confined to the plane perpendicular to it, forever. With
# right = +Z that plane is XY, which is where the shapes are authored. Point
# it any other way and the single-axis rig can only approximate a shape it
# is geometrically barred from reaching.
DEFAULT_HEADING = ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0))


# --------------------------------------------------------------------------
# vector helpers (no numpy: this must pip-install on anything)
# --------------------------------------------------------------------------

def _add(a, b): return (a[0] + b[0], a[1] + b[1], a[2] + b[2])
def _sub(a, b): return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
def _mul(a, s): return (a[0] * s, a[1] * s, a[2] * s)
def _dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def _len(a): return math.sqrt(max(_dot(a, a), 0.0))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _norm(a):
    n = _len(a)
    return (0.0, 0.0, 1.0) if n < 1e-12 else (a[0] / n, a[1] / n, a[2] / n)


def _rot(v, axis, ang):
    """Rodrigues rotation of v about a unit axis."""
    c, s = math.cos(ang), math.sin(ang)
    return _add(_add(_mul(v, c), _mul(_cross(axis, v), s)),
                _mul(axis, _dot(axis, v) * (1 - c)))


def _clamp(v, lo, hi): return lo if v < lo else hi if v > hi else v


# --------------------------------------------------------------------------
# kinematics of the alternating-axis chain
# --------------------------------------------------------------------------

def _axis_of(i, frame, mode):
    """Which of the frame's two lateral axes joint i rotates about.

    The large prototype mounts motors "with their axis of rotation alternately
    perpendicular to one another (X after Y) so that it can form 3D
    structures"; the small one mounts them all the same way and is therefore
    planar. Both cases fall out of this one line.
    """
    right, up, _ = frame
    if mode == "single":
        return right
    return right if i % 2 == 0 else up


def forward_kinematics(angles, device=DEFAULT_DEVICE,
                       base=(0.0, 0.0, 0.0), heading=None):
    """Servo angles (radians) -> the chain's 3D polyline of n+1 points.

    The frame is carried along the chain, so a joint's axis is defined in the
    *previous* link's coordinates — which is what makes an alternating chain
    able to leave the plane at all.
    """
    d = _dev(device)
    L, lim = _links(d), _limits(d)
    right, up, fwd = heading or DEFAULT_HEADING
    pts = [base]
    p = base
    for i, th in enumerate(angles[:d["n"]]):
        a = _axis_of(i, (right, up, fwd), d["axes"])
        th = _clamp(th, -lim[i], lim[i])
        right, up, fwd = _rot(right, a, th), _rot(up, a, th), _rot(fwd, a, th)
        p = _add(p, _mul(fwd, L[i]))
        pts.append(p)
    return pts


def resample(path, step, count):
    """Walk a polyline at fixed arc length — the paper's "according to length
    of each joint". -> exactly `count` points, extrapolating past the end
    along the last direction if the path is shorter than the chain.

    `step` is one length, or one length per segment so a chain whose joints
    differ in size samples its target where its own joints actually fall.
    """
    steps = list(step) if isinstance(step, (list, tuple)) \
        else [step] * max(count - 1, 1)
    if len(path) < 2:
        return [tuple(path[0]) if path else (0.0, 0.0, 0.0)] * count
    out = [tuple(path[0])]
    i, cur = 0, tuple(path[0])
    while len(out) < count:
        need = steps[min(len(out) - 1, len(steps) - 1)]
        while need > 0 and i < len(path) - 1:
            seg = _sub(path[i + 1], cur)
            L = _len(seg)
            if L < 1e-9:
                i += 1
                continue
            if L >= need:
                cur = _add(cur, _mul(_norm(seg), need))
                need = 0
            else:
                need -= L
                cur = tuple(path[i + 1])
                i += 1
        if need > 0:                       # ran off the end: keep going straight
            tail = _norm(_sub(path[-1], path[-2]))
            cur = _add(cur, _mul(tail, need))
        out.append(cur)
    return out


def arclength(path):
    return sum(_len(_sub(path[i + 1], path[i])) for i in range(len(path) - 1))


def fit_length(path, total):
    """Scale a path about its centroid until its arc length is `total`.

    A device renders a shape at the size that uses the whole chain — the
    handset in Figure 1d is as big as 186 cm of servos makes it. Without
    this the tail of the chain runs off the end of a short outline and the
    residual measures our framing, not the hardware.
    """
    L = arclength(path)
    if L < 1e-9:
        return list(path)
    s = total / L
    cx = sum(p[0] for p in path) / len(path)
    cy = sum(p[1] for p in path) / len(path)
    cz = sum(p[2] for p in path) / len(path)
    return [((p[0] - cx) * s, (p[1] - cy) * s, (p[2] - cz) * s) for p in path]


def fit_curve(path, device=DEFAULT_DEVICE, base=None, heading=None, autoscale=True):
    """Vector data -> per-joint angles. The paper's shape-from-outline method.

    -> {"angles": [...], "points": [...], "error": metres, "clipped": k}

    At each joint the achievable headings lie on a cone around that joint's
    single axis, so we take the angle that brings the link closest to the
    resampled target direction and carry the residual forward. `error` is the
    mean distance between the target vertices and the shape the hardware can
    actually hold — the honest gap between a drawing and a device.
    """
    d = _dev(device)
    L, lim = _links(d), _limits(d)
    path = [tuple(p) for p in path]
    if autoscale:
        path = fit_length(path, sum(L))
    tgt = resample(path, L, d["n"] + 1)
    if base is not None:
        # Placing the chain somewhere means moving the shape there, not
        # leaving the target behind at the origin — otherwise the residual
        # below is dominated by that translation and reports a perfect fit
        # as a quarter-metre miss.
        off = _sub(tuple(base), tgt[0])
        tgt = [_add(q, off) for q in tgt]
    right, up, fwd = heading or DEFAULT_HEADING
    p = tuple(base) if base is not None else tgt[0]
    pts, angles, clipped = [p], [], 0
    for i in range(d["n"]):
        want = _norm(_sub(tgt[i + 1], p))
        a = _axis_of(i, (right, up, fwd), d["axes"])
        # rotating fwd about a sweeps the plane spanned by (fwd, a x fwd)
        b = _cross(a, fwd)
        th = math.atan2(_dot(want, b), _dot(want, fwd))
        if abs(th) > lim[i]:
            th = _clamp(th, -lim[i], lim[i])
            clipped += 1
        right, up, fwd = _rot(right, a, th), _rot(up, a, th), _rot(fwd, a, th)
        p = _add(p, _mul(fwd, L[i]))
        angles.append(th)
        pts.append(p)
    err = sum(_len(_sub(pts[i], tgt[i])) for i in range(len(pts))) / len(pts)
    return {"angles": angles, "points": pts, "error": err, "clipped": clipped}


def snap_to_grid(angles, step_deg=90.0, device=DEFAULT_DEVICE):
    """Physical snap-to-grid: "the line automatically snap to certain angles
    after users manipulate it roughly" (paper, Fig. 11 — the CAD model is
    computationally limited to form right angles)."""
    lim = _limits(_dev(device))
    s = math.radians(step_deg)
    return [_clamp(round(t / s) * s, -lim[i], lim[i])
            for i, t in enumerate(angles)]


def relax(state, target, stiffness, dt=1.0 / CONTROL_HZ, device=DEFAULT_DEVICE):
    """One step of the paper's "mass spring optimization", per joint.

    `stiffness` is the per-joint compliance the paper controls through the
    servo's PID terms — 1.0 is a rigid joint that snaps to the commanded
    angle, 0.0 is a limp joint the user can backdrive freely. Running the
    morph through this instead of a lerp is what gives the transition weight:
    the loose end of the chain lags and overshoots the way a real one does.
    """
    lim = _limits(_dev(device))
    out_a, out_v = [], []
    for i, (a, v) in enumerate(state):
        k = 90.0 * (0.15 + 0.85 * (stiffness[i] if i < len(stiffness) else 1.0))
        c = 2.0 * math.sqrt(k)                      # critically damped
        acc = k * (target[i] - a) - c * v
        v += acc * dt
        a = _clamp(a + v * dt, -lim[i], lim[i])
        out_a.append(a)
        out_v.append(v)
    return list(zip(out_a, out_v))


# --------------------------------------------------------------------------
# display primitives (paper, p.3: Curves / Surfaces / Solids)
# --------------------------------------------------------------------------

def _scale_to(path, span):
    """Fit a path inside a box of the given diagonal, centred on origin."""
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    zs = [p[2] for p in path]
    ext = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 1e-6)
    s = span / ext
    cx, cy, cz = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2, (max(zs) + min(zs)) / 2
    return [((p[0] - cx) * s, (p[1] - cy) * s, (p[2] - cz) * s) for p in path]


# Every generator below is sized from the device, because the achievable
# feature size is bounded by the link length: a chain of 6.6 cm links cannot
# hold a 2.75 cm U-turn no matter what you command. The paper says the same
# thing from the hardware side — "resolution depends on the size of the motors
# used" — which is exactly why they built the 2.4 cm prototype as well.

def _rig(device):
    d = _dev(device)
    return d, d["n"] * d["link"], d["link"]


def curve_shape(kind="sine", device=DEFAULT_DEVICE, n=90):
    """Plain 2D/3D curves — the first display primitive."""
    _, total, link = _rig(device)
    out = []
    if kind == "line":
        return [(t / (n - 1) * total - total / 2, 0.0, 0.0) for t in range(n)]
    if kind == "helix":
        r = link * 2.2
        turns = max(1.0, total / (2 * math.pi * r) * 0.55)
        rise = math.sqrt(max(total ** 2 / (turns ** 2) - (2 * math.pi * r) ** 2, 0.0))
        for i in range(n):
            t = i / (n - 1)
            a = t * math.pi * 2 * turns
            out.append((r * math.cos(a), (t - 0.5) * rise * turns, r * math.sin(a)))
        return out
    amp = link * 2.0                                  # gentle enough to hold
    waves = 3.0
    span = total * 0.80                               # arc > chord on a wave
    for i in range(n):
        t = i / (n - 1)
        out.append(((t - 0.5) * span, amp * math.sin(t * math.pi * 2 * waves), 0.0))
    return out


def serpentine_surface(device=DEFAULT_DEVICE):
    """A surface made "by creating tight serpentine curves" (paper, p.3).

    One continuous curve, boustrophedon. Row pitch is two link lengths — the
    tightest U-turn the chain can actually hold — and the row count follows
    from how much chain is left, so the surface is as dense as the hardware
    allows and no denser.
    """
    _, total, link = _rig(device)
    pitch = link * 2.0
    rows = max(3, int(math.sqrt(total / pitch)))
    width = max(link * 2, (total - rows * pitch) / rows)
    out = []
    for r in range(rows):
        y = r * pitch - rows * pitch / 2
        xs = [-width / 2, width / 2] if r % 2 == 0 else [width / 2, -width / 2]
        out.append((xs[0], y, 0.0))
        out.append((xs[1], y, 0.0))
    return out


def space_filling_solid(device=DEFAULT_DEVICE):
    """"Solid forms with 3D physical geometry by using a space filling
    technique" (paper, p.3).

    Requires alternating axes — the planar prototype physically cannot do
    this, and `clip_from_shapes` refuses it there rather than faking it.
    """
    _, total, link = _rig(device)
    pitch = link * 2.0
    layers = 3
    rows = max(2, int((total / layers) / (pitch * 2.2)))
    width = max(link * 2, (total / layers - rows * pitch) / rows)
    out = []
    for L in range(layers):
        z = L * pitch - layers * pitch / 2
        for r in range(rows):
            rr = r if L % 2 == 0 else rows - 1 - r
            y = rr * pitch - rows * pitch / 2
            xs = [-width / 2, width / 2] if (rr + L) % 2 == 0 else [width / 2, -width / 2]
            out.append((xs[0], y, z))
            out.append((xs[1], y, z))
    return out


def data_curve(values, width=1.1, height=0.5):
    """"Data from underlying models can be physically rendered ... a fit curve
    from a statistical analysis, or a line chart" (paper, p.3).

    This is the one that matters for the tutor: any series the lesson is about
    — a joint's speed over time, a temperature, a graph edge count — becomes a
    physical curve the chain holds up.
    """
    v = list(values) or [0.0]
    lo, hi = min(v), max(v)
    rng = (hi - lo) or 1.0
    return [(i / max(len(v) - 1, 1) * width - width / 2,
             (x - lo) / rng * height - height / 2, 0.0) for i, x in enumerate(v)]


# ---- Iconic Forms (paper, p.3 and Fig. 1) --------------------------------

def _poly(pts, close=True, dense=3):
    """Corner list -> densely sampled path, so resampling keeps the corners."""
    out = []
    seq = list(pts) + ([pts[0]] if close else [])
    for i in range(len(seq) - 1):
        a, b = seq[i], seq[i + 1]
        for k in range(dense):
            t = k / dense
            out.append(tuple(a[j] + (b[j] - a[j]) * t for j in range(3)))
    out.append(tuple(seq[-1]))
    return out


WRIST_R = 0.055                       # a wrist is a wrist on either prototype


def iconic(name="phone", device=DEFAULT_DEVICE):
    """The paper's Figure 1 transformations, as vector outlines.

    "when the user wants to make a call, it transforms into the vector icon of
    a telephone which lets the user both understand the mode and functionally
    allows her to easily grip and hold it up to her mouth and ear."
    """
    _, total, link = _rig(device)
    if name == "wristband":                       # Fig. 1b — closed loop + tab
        # A wrist does not scale with the chain, so the surplus length becomes
        # extra turns — which is what wrapping 186 cm around an arm looks like.
        turns = max(1.0, (total - 0.09) / (2 * math.pi * WRIST_R))
        steps = max(24, int(turns * 26))
        out = [(WRIST_R * math.cos(i / steps * math.pi * 2 * turns),
                WRIST_R * math.sin(i / steps * math.pi * 2 * turns),
                i / steps * turns * link * 0.35) for i in range(steps + 1)]
        out += [(WRIST_R + 0.03, 0.02, out[-1][2]),
                (WRIST_R + 0.09, 0.05, out[-1][2])]        # the peel-off flag
        return out
    if name == "phone":                           # Fig. 1d — handset outline
        return _poly([(-0.20, -0.05, 0.0), (-0.12, 0.07, 0.0), (0.12, 0.07, 0.0),
                      (0.20, -0.05, 0.0), (0.11, -0.09, 0.0), (0.06, -0.02, 0.0),
                      (-0.06, -0.02, 0.0), (-0.11, -0.09, 0.0)], dense=4)
    if name == "surface":
        return serpentine_surface(device)
    if name == "solid":
        return space_filling_solid(device)
    if name == "lamp":                            # Fig. 7 — stand + shade + lever
        stem = [(0.0, -0.30 + i * 0.05, 0.0) for i in range(9)]
        shade = [(0.13 * math.cos(math.pi * (0.15 + i / 12 * 0.7)),
                  0.16 + 0.09 * math.sin(math.pi * (0.15 + i / 12 * 0.7)), 0.0)
                 for i in range(13)]
        lever = [(0.10, -0.05, 0.0), (0.20, -0.02, 0.0)]
        return stem + shade + lever
    if name == "ruler":                           # Fig. 10 — drawing guide
        return curve_shape("sine", device)
    if name == "slider":                          # "User Interface Elements"
        return _poly([(-0.25, 0.0, 0.0), (0.25, 0.0, 0.0), (0.25, 0.05, 0.0),
                      (0.18, 0.05, 0.0), (0.18, 0.0, 0.0)], close=False, dense=5)
    return curve_shape("sine", device)


SHAPES = ["curve", "helix", "surface", "solid", "phone", "wristband",
          "lamp", "ruler", "slider", "data"]


def shape_path(name, values=None, device=DEFAULT_DEVICE):
    """Name -> vector path, the single lookup the server and the board share."""
    if name == "data":
        return data_curve(values or [0.2, 0.5, 0.35, 0.8, 0.6, 0.95, 0.7])
    if name in ("curve", "sine"):
        return curve_shape("sine", device)
    if name in ("helix", "line"):
        return curve_shape(name, device)
    return iconic(name, device)


# --------------------------------------------------------------------------
# CAD: Bezier / NURBS vector data (paper, p.6)
# --------------------------------------------------------------------------
# "LineFORM can be used to physically render and manipulate bezier or NURBs
# curves in 3D. Users can freely modify the model through direct deformation."

def bezier(ctrl, n=64):
    """de Casteljau over 3D control points -> polyline."""
    out = []
    for k in range(n):
        t = k / (n - 1)
        pts = [tuple(p) for p in ctrl]
        while len(pts) > 1:
            pts = [tuple(pts[i][j] + (pts[i + 1][j] - pts[i][j]) * t for j in range(3))
                   for i in range(len(pts) - 1)]
        out.append(pts[0])
    return out


def nurbs(ctrl, degree=3, weights=None, n=96):
    """Rational B-spline with a clamped uniform knot vector -> polyline.

    Enough to render what a CAD package hands over; the point is that the
    chain approximates the *curve*, not the control cage.
    """
    m = len(ctrl)
    if m <= degree:
        return bezier(ctrl, n)
    w = list(weights or [1.0] * m)
    knots = ([0.0] * (degree + 1) +
             [i / (m - degree) for i in range(1, m - degree)] +
             [1.0] * (degree + 1))

    def basis(i, k, t):
        if k == 0:
            return 1.0 if (knots[i] <= t < knots[i + 1] or
                           (t >= 1.0 and knots[i + 1] >= 1.0 > knots[i])) else 0.0
        a = b = 0.0
        d1 = knots[i + k] - knots[i]
        d2 = knots[i + k + 1] - knots[i + 1]
        if d1 > 1e-12:
            a = (t - knots[i]) / d1 * basis(i, k - 1, t)
        if d2 > 1e-12:
            b = (knots[i + k + 1] - t) / d2 * basis(i + 1, k - 1, t)
        return a + b

    out = []
    for s in range(n):
        t = s / (n - 1)
        num, den = [0.0, 0.0, 0.0], 0.0
        for i in range(m):
            bw = basis(i, degree, t) * w[i]
            den += bw
            for j in range(3):
                num[j] += ctrl[i][j] * bw
        out.append(tuple(x / den for x in num) if den > 1e-12 else tuple(ctrl[-1]))
    return out


def from_svg_path(d, scale=0.0016, flip_y=True):
    """A subset of the SVG `d` grammar (M/L/H/V/C/Q/Z, absolute + relative).

    This is the paper's "extract outline from binary image data as series of
    vectors" arriving by the route a modern toolchain actually uses: whatever
    the CAD app or icon set exports. Curves are flattened before fitting.
    """
    import re
    toks = re.findall(r"([MmLlHhVvCcQqZz])|(-?\d*\.?\d+(?:e-?\d+)?)", d)
    cmds, cur = [], None
    for letter, num in toks:
        if letter:
            cur = [letter]
            cmds.append(cur)
        elif cur is not None:
            cur.append(float(num))
    pts, pos, start = [], (0.0, 0.0), (0.0, 0.0)
    for c in cmds:
        op, args = c[0], c[1:]
        rel = op.islower()
        o = op.upper()
        i = 0
        while True:
            if o == "Z":
                if pts:
                    pts.append(start)
                pos = start
                break
            need = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "Q": 4}[o]
            if i + need > len(args):
                break
            a = args[i:i + need]
            i += need
            if o in ("M", "L"):
                p = (a[0] + pos[0], a[1] + pos[1]) if rel else (a[0], a[1])
                if o == "M" and not pts:
                    start = p
                pts.append(p)
                pos = p
            elif o == "H":
                pos = (a[0] + pos[0] if rel else a[0], pos[1])
                pts.append(pos)
            elif o == "V":
                pos = (pos[0], a[1] + pos[1] if rel else a[1])
                pts.append(pos)
            else:
                pairs = [(a[k] + (pos[0] if rel else 0.0),
                          a[k + 1] + (pos[1] if rel else 0.0)) for k in range(0, need, 2)]
                ctrl = [(pos[0], pos[1], 0.0)] + [(x, y, 0.0) for x, y in pairs]
                for q in bezier(ctrl, 12)[1:]:
                    pts.append((q[0], q[1]))
                pos = pairs[-1]
            if o == "M":
                o = "L"                              # repeated pairs are lineto
            if i >= len(args):
                break
    if not pts:
        return curve_shape("sine")
    sy = -1.0 if flip_y else 1.0                     # SVG y grows downward
    path = [(x * scale, y * scale * sy, 0.0) for x, y in pts]
    return _scale_to(path, 0.9)


# --------------------------------------------------------------------------
# constraint: wrap the curve around a limb (paper, Fig. 9)
# --------------------------------------------------------------------------
# "This application demonstrates how whole body motion can be constrained by
# wrapping the actuated curve interface around limbs or joints like bandages
# so that it acts as an exoskeleton ... it can also record motion and replay
# back on your body ... to learn kinesthetic motion such as sports and dances
# as an external motor memory."
#
# This is the join between the paper and the tutor: the how-to gesture is the
# recorded motion, and the wrap is what replays it onto the learner.

def wrap_limb(a, b, radius=0.055, turns=3.5, samples=64, lead=0.10):
    """Helix around the segment a->b, plus a straight lead-in. -> path.

    Feed it a forearm (elbow -> wrist) from the skeleton and the chain
    bandages that forearm, following it frame by frame.
    """
    axis = _sub(b, a)
    L = _len(axis)
    if L < 1e-6:
        return curve_shape("sine")
    f = _norm(axis)
    ref = (0.0, 1.0, 0.0) if abs(f[1]) < 0.9 else (1.0, 0.0, 0.0)
    u = _norm(_cross(f, ref))
    v = _cross(f, u)
    out = [_sub(a, _mul(f, lead))]
    for i in range(samples):
        t = i / (samples - 1)
        ang = t * math.pi * 2 * turns
        c = _add(a, _mul(f, t * L))
        out.append(_add(c, _add(_mul(u, radius * math.cos(ang)),
                                _mul(v, radius * math.sin(ang)))))
    return out


def constraint_stiffness(n, locked, softness=0.12):
    """Variable stiffness per joint: "the interface can allow users to deform
    only specific part of the line" (paper, p.3).

    `locked` is the fraction of the chain held rigid from the base; the rest
    stays compliant so the learner can still move through it. That is the
    hinge behaviour the paper renders by changing PID terms per joint.
    """
    k = int(n * _clamp(locked, 0.0, 1.0))
    return [1.0] * k + [softness] * (n - k)


# --------------------------------------------------------------------------
# clips: the transport format the board renders
# --------------------------------------------------------------------------

def _pose_from_path(path, device, base, heading):
    fit = fit_curve(path, device, base=base, heading=heading)
    return fit


def clip_from_paths(paths, labels=None, seconds=None, fps=12,
                    device=DEFAULT_DEVICE, stiffness=None, snap=False,
                    base=(0.0, 0.9, 0.0), title="", source=""):
    """Morph the chain through a sequence of target shapes. -> clip dict.

    Frames are produced by the mass-spring settle, not by interpolating
    positions, so what the board draws is a trajectory the hardware could
    follow: the joints nearest the base arrive first and the free end trails.
    """
    d = _dev(device)
    paths = [p for p in paths if p] or [curve_shape("sine")]
    labels = list(labels or [])
    labels += [""] * (len(paths) - len(labels))
    seconds = seconds or max(3.0, 2.6 * len(paths))
    n_frames = max(int(round(seconds * fps)), 2)
    per = max(n_frames // len(paths), 2)

    heading = DEFAULT_HEADING
    stiff = list(stiffness or [1.0] * d["n"])
    targets, errors, clipped = [], [], []
    for p in paths:
        fit = _pose_from_path(p, device, base, heading)
        ang = snap_to_grid(fit["angles"], 90.0, device) if snap else fit["angles"]
        targets.append(ang)
        errors.append(round(fit["error"], 4))
        clipped.append(fit["clipped"])

    state = [(a, 0.0) for a in targets[0]]
    frames, angle_log, seg = [], [], []
    dt = 1.0 / fps
    for i in range(n_frames):
        k = min(i // per, len(targets) - 1)
        # sub-step the solver so a 12 fps clip still settles like a 60 Hz loop
        for _ in range(max(int(CONTROL_HZ * dt), 1)):
            state = relax(state, targets[k], stiff, 1.0 / CONTROL_HZ, device)
        ang = [a for a, _ in state]
        pts = forward_kinematics(ang, device, base=base, heading=heading)
        frames.append([[round(c, 4) for c in p] for p in pts])
        angle_log.append([round(math.degrees(a), 1) for a in ang])
        seg.append(k)
    return {
        "version": 1, "kind": "lineform", "fps": fps,
        "seconds": round(n_frames / fps, 2),
        "title": title, "source": source,
        # The board projects every clip through the same camera maths, and
        # the tutor reads grid coordinates off it to annotate the chain, so
        # a clip without one cannot be pointed at.
        "camera": {"yaw": 0.5, "pitch": 0.12, "dist": 3.0,
                   "target": [0.0, float(base[1]), 0.0], "fov": 40.0},
        "device": {"key": device, "name": d["name"], "joints": d["n"],
                   "link_m": d["link"], "limit_deg": round(math.degrees(d["limit"]), 1),
                   "axes": d["axes"], "servo": d["servo"], "total_m": d["total_m"],
                   "control_hz": CONTROL_HZ},
        "frames": frames, "angles": angle_log, "segment": seg,
        "labels": labels, "stiffness": [round(s, 2) for s in stiff],
        "fit_error": errors, "clipped": clipped, "snap": bool(snap),
        "paper": "Nakagaki, Follmer & Ishii, LineFORM, UIST '15",
    }


def clip_from_shapes(names, seconds=None, fps=12, device=DEFAULT_DEVICE,
                     snap=False, values=None, title=""):
    """Convenience: shape names -> clip. Rejects solids on the planar rig."""
    names = [n for n in names] or ["curve"]
    if _dev(device)["axes"] == "single":
        names = [n for n in names if n != "solid"] or ["curve"]
    return clip_from_paths([shape_path(n, values, device) for n in names],
                           labels=names, seconds=seconds, fps=fps,
                           device=device, snap=snap,
                           title=title or " -> ".join(names))


def describe(clip) -> str:
    """Clip -> text for prompt injection, so the tutor can talk about the
    device it is showing instead of narrating over it blindly."""
    d = clip["device"]
    lines = [f"Actuated curve interface: {d['name']} — {d['joints']} 1-DOF servos "
             f"({d['servo']}), {d['link_m']*100:.1f} cm per joint, "
             f"{d['total_m']} m total, +/-{d['limit_deg']} deg, {d['axes']} axes."]
    if clip.get("labels"):
        lines.append("Shapes shown, in order: " +
                     ", ".join(x for x in clip["labels"] if x))
    if clip.get("fit_error"):
        worst = max(clip["fit_error"])
        lines.append(f"Shape-fit residual: {worst*100:.1f} cm worst case — the gap "
                     f"between the drawn curve and what {d['joints']} single-axis "
                     f"joints can physically hold.")
    if any(clip.get("clipped", [])):
        lines.append(f"{max(clip['clipped'])} joints hit their {d['limit_deg']} deg "
                     f"limit and were clamped.")
    if clip.get("snap"):
        lines.append("Snap-to-grid is on: the curve is constrained to right angles.")
    return "\n".join(lines)


if __name__ == "__main__":
    import json
    for key in DEVICES:
        c = clip_from_shapes(["curve", "phone", "wristband", "surface", "solid"],
                             seconds=10, device=key)
        print(describe(c))
        print("  frames:", len(c["frames"]), " payload:",
              len(json.dumps(c)) // 1024, "KB")
        print()
    svg = from_svg_path("M 10 10 L 90 10 C 120 40 120 70 90 100 L 10 100 Z")
    f = fit_curve(svg)
    print("SVG outline -> chain: error", round(f["error"] * 100, 2), "cm,",
          f["clipped"], "joints clamped")
