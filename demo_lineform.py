"""
demo_lineform.py - the LineFORM control algorithm, one servo at a time.

Stdlib only (math), no dependencies, ~100 lines: it pastes straight into an
online runner. This is the same solve that lineform.fit_curve does for the
real 28-joint device, cut down to a planar 12-joint chain so a person can
watch it happen.

    run it            programiz.com/python-programming/online-compiler
    step through it   pythontutor.com/visualize.html   <- every variable,
                      forward and back, one line at a time

The paper (Nakagaki, Follmer & Ishii, UIST '15, p.4):

    "we extract outline from binary image data as series of vectors, then
     calculate the angle for each servo motor according to length of each
     joint ... this is a slightly different, and more simple, task than the
     inverse kinematic control needed to move a serpentine robot towards a
     goal, as we do not care about the position of the end effector"

So: no IK, no solver, no optimiser. Walk the target curve at the joint
pitch, and at each joint turn as far towards the next sample as the servo
is physically allowed to turn. What comes back is the shape the hardware
can actually hold, plus the honest gap between that and what was asked for.
"""

import math
import sys

# pythontutor.com stops after ~1000 executed steps, and the full run is well
# past that. Set this True before pasting there: it shrinks the curve and
# runs only the first demo, which fits the budget and still shows every
# angle being decided. Leave it False anywhere that just runs the file.
TRACE = False

# "id" = Bahasa Indonesia, "en" = English.  Also: python demo_lineform.py id
LANG = "id"
if len(sys.argv) > 1 and sys.argv[1] in ("id", "en"):
    LANG = sys.argv[1]

JOINTS = 12                       # servos in the chain
LINK = 0.12                       # metres between joints
LIMIT = math.radians(103.0)       # Dynamixel AX-18A travel, per the paper

TEXT = {
    "en": {
        "title": "LineFORM control: one angle per servo, no inverse kinematics.",
        "rig": "%d servos, %.0f cm links, +/-%.0f deg travel",
        "how": ["", "  How it works, three steps:",
                "   1. walk the target curve, dropping a point every %.0f cm of",
                "      arc length - the JOINT SPACING decides where the samples",
                "      land, not the curve",
                "   2. at each joint, take the turn between one sample segment",
                "      and the next  (never from where the tip has drifted to)",
                "   3. clamp that turn to what the servo can physically do, then",
                "      run it forward to see the shape it can actually hold",
                "  The tip position is never solved for. That is the paper's point.",
                ""],
        "can": "a shape it can hold",
        "cannot": "a shape it cannot",
        "head": "  servo    wanted   allowed   at limit   tip after this joint",
        "legend": "\n  asked for: the dotted curve      held: the chain of 'o'",
        "score": "  worst gap: %.1f cm   servos at their limit: %d of %d",
        "yes": "YES",
        "close": ["", "  The second gap is not a bug, and not a failed solve. A chain",
                  "  of 1-DOF hinges physically cannot pass through every point of",
                  "  an arbitrary curve. The residual and the count of clamped",
                  "  servos are what the tutor reports on screen instead of quietly",
                  "  drawing a shape the hardware could never take."],
    },
    "id": {
        "title": "Kontrol LineFORM: satu sudut per motor servo, tanpa kinematika invers.",
        "rig": "%d motor servo, tautan %.0f cm, jangkauan putar +/-%.0f derajat",
        "how": ["", "  Cara kerjanya, tiga langkah:",
                "   1. susuri kurva target, jatuhkan satu titik setiap %.0f cm",
                "      panjang busur - JARAK ANTAR SENDI yang menentukan letak",
                "      titiknya, bukan kurvanya",
                "   2. di tiap sendi, ambil sudut belok antara satu segmen dan",
                "      segmen berikutnya  (bukan dari posisi ujung yang melenceng)",
                "   3. potong sudut itu sebatas kemampuan fisik servo, lalu",
                "      jalankan maju untuk melihat bentuk yang benar-benar bisa",
                "      ditahan",
                "  Posisi ujung tidak pernah dihitung. Itulah inti makalahnya.",
                ""],
        "can": "bentuk yang bisa ditahan",
        "cannot": "bentuk yang tidak bisa ditahan",
        "head": "  servo   diminta  diizinkan  kena batas  ujung setelah sendi ini",
        "legend": "\n  yang diminta: kurva bertitik     yang ditahan: rantai 'o'",
        "score": "  selisih terburuk: %.1f cm   servo yang mentok: %d dari %d",
        "yes": "YA",
        "close": ["", "  Selisih pada kasus kedua bukan bug, dan bukan solusi yang gagal.",
                  "  Rantai engsel satu derajat kebebasan memang tidak mungkin",
                  "  melewati setiap titik pada kurva sembarang. Nilai selisih dan",
                  "  jumlah servo yang mentok itulah yang ditampilkan tutor di layar,",
                  "  alih-alih diam-diam menggambar bentuk yang tidak sanggup",
                  "  dibentuk perangkat kerasnya."],
    },
}


def t(key, *args):
    s = TEXT[LANG][key]
    return (s % args) if args else s


def target_curve(samples=160, width=1.30, height=0.34):
    """The shape we ASK for: a sine wave. Any polyline works."""
    return [(width * i / (samples - 1),
             height * math.sin(2 * math.pi * i / (samples - 1)))
            for i in range(samples)]


def resample(path, step, count):
    """Walk the path, dropping a point every `step` metres of ARC LENGTH.
    This is what makes it the paper's problem instead of an IK problem: the
    chain's own joint pitch decides where the samples land, not the curve.

    `remaining` has to carry across segments - a segment shorter than the
    step is consumed, not skipped, or the walk slides backwards.
    """
    out, i, here, remaining = [path[0]], 0, path[0], step
    while len(out) <= count and i < len(path) - 1:
        ax, ay = here
        bx, by = path[i + 1]
        d = math.hypot(bx - ax, by - ay)
        if d < remaining:                  # not far enough: carry the rest on
            remaining -= d
            i += 1
            here = path[i]
            continue
        t = remaining / d                  # land exactly `remaining` along it
        here = (ax + (bx - ax) * t, ay + (by - ay) * t)
        out.append(here)
        remaining = step
    while len(out) <= count:               # curve ran out: hold the last point
        out.append(out[-1])
    return out


def solve(path):
    """-> [(wanted, allowed, clamped)], the points reached, the samples.

    One angle per servo, each decided from the TARGET's own bend - the angle
    between one resampled segment and the next - never from where the tip
    has drifted to. Measuring from the tip makes every joint chase its own
    accumulated error and the chain thrashes between its limits.

    The turn is measured against the chain's REAL heading though, so when a
    servo runs out of travel the next one inherits the shortfall and tries
    to make it up. Local, greedy, and self-correcting: the paper's method.
    """
    want = resample(path, LINK, JOINTS)
    segs = [math.atan2(want[j + 1][1] - want[j][1],
                       want[j + 1][0] - want[j][0]) for j in range(JOINTS)]

    angles, heading = [], 0.0              # base points along +x
    for j in range(JOINTS):
        turn = (segs[j] - heading + math.pi) % (2 * math.pi) - math.pi
        allowed = max(-LIMIT, min(LIMIT, turn))
        angles.append((turn, allowed, abs(allowed - turn) > 1e-9))
        heading += allowed                 # the chain's heading, not the curve's

    x, y = want[0]                         # forward kinematics with what it can do
    heading, points = 0.0, [(x, y)]
    for _, allowed, _ in angles:
        heading += allowed
        x, y = x + LINK * math.cos(heading), y + LINK * math.sin(heading)
        points.append((x, y))
    return angles, points, want


def plot(points, curve, w=58, h=17):
    """ASCII: '.' is the curve we asked for, 'o' the servos, '#' the base."""
    xs = [p[0] for p in curve + points]
    ys = [p[1] for p in curve + points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    sx = (w - 1) / (x1 - x0 or 1)
    sy = (h - 1) / (y1 - y0 or 1)
    grid = [[" "] * w for _ in range(h)]

    def put(p, ch):
        c = int((p[0] - x0) * sx)
        r = h - 1 - int((p[1] - y0) * sy)
        if 0 <= r < h and 0 <= c < w:
            grid[r][c] = ch

    for p in curve:
        put(p, ".")
    for p in points:
        put(p, "o")
    put(points[0], "#")
    return "\n".join("  |" + "".join(row) + "|" for row in grid)


def tight_loop(samples=400, radius=0.06, turns=4):
    """A shape the hardware CANNOT hold: a 6 cm circle needs 115 deg of
    travel per 12 cm link and the servo has 103, so every joint runs out.

    It winds `turns` times so there is 1.5 m of curve for 1.44 m of chain -
    otherwise the chain simply runs off the end of a short path, which is a
    different failure and a less interesting one.
    """
    return [(radius * math.cos(turns * 2 * math.pi * i / (samples - 1)),
             radius * math.sin(turns * 2 * math.pi * i / (samples - 1)))
            for i in range(samples)]


def run(title, path):
    angles, points, want = solve(path)

    # --- step by step: one servo per line -------------------------------
    print("\n=== %s ===\n" % title)
    print(t("head"))
    for j, (turn, allowed, hit) in enumerate(angles, 1):
        px, py = points[j]
        print("   %2d     %+7.1f  %+7.1f      %-7s   (%+.2f, %+.2f)"
              % (j, math.degrees(turn), math.degrees(allowed),
                 t("yes") if hit else "-", px, py))

    # --- the honest part -------------------------------------------------
    err = max(math.hypot(p[0] - q[0], p[1] - q[1])
              for p, q in zip(points, want))
    clipped = sum(1 for _, _, hit in angles if hit)
    print("\n" + plot(points, path))
    print(t("legend"))
    print(t("score", err * 100, clipped, JOINTS))


def main():
    print(t("title"))
    print(t("rig", JOINTS, LINK * 100, math.degrees(LIMIT)))
    for line in TEXT[LANG]["how"]:
        print(line % (LINK * 100) if "%" in line else line)

    run(t("can"), target_curve(20 if TRACE else 160))
    if not TRACE:
        run(t("cannot"), tight_loop())

    for line in TEXT[LANG]["close"]:
        print(line)


main()
