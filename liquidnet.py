"""liquidnet.py — a liquid time-constant network, dependency-free.

The reference implementation of the LTC cell the page runs during camera
verification (index.html mirrors this byte-for-byte in spirit, the way the
board mirrors motion.py). The model is Hasani et al. 2021, "Liquid
Time-constant Networks", integrated with the paper's fused ODE solver:

    f       = sigmoid(W x + U u + b)
    x(t+dt) = (x + dt * f * A) / (1 + dt * (1/tau + f))     (elementwise)

The name is the middle line: the effective time constant of neuron i is

    tau_sys_i = tau_i / (1 + tau_i * f_i(x, u))

so it DEPENDS ON THE INPUT — a busy input tightens the neuron's memory, a
quiet one relaxes it. That is the property the verification loop wants:
when the tracked body is moving and scoring, the state follows closely;
when the tracker flickers for a frame, the state coasts instead of
snapping to zero.

Training follows the reservoir discipline: the recurrent core (W, U, b,
tau, A) is FIXED from a seeded generator, identical for every student, so
the dynamics are reproducible and there is nothing to overfit; only the
readout adapts, online, one cross-entropy gradient step per frame against
the teacher signal the angle-band physics provides. The readout sees the
CONCATENATION [gain*x ; u] — liquid state plus the direct input — the
standard reservoir skip connection: the direct path makes the mapping
learnable immediately, the state contributes the temporal judgement, and
neither can silently label the lagged transition paths of the other (the
failure mode a state-only readout was measured to have here). The network
personalises without ever becoming the judge — coach.py's measured angles
remain the authority.

    from liquidnet import LTC
    net = LTC()
    y = net.step(u=[...11 floats...], dt=1/30, teach=1.0)

Run this file for the self-test:

    python liquidnet.py
"""

from __future__ import annotations

import math

N_HIDDEN = 24
N_INPUT = 13          # 9 joint-band deviations + score/100 + seen flag
#                       + hold fraction + hand-step flag
SEED = 0xC0FFEE
READOUT_GAIN = 4.0    # LTC states live near +-0.3; the readout needs reach


def _mulberry32(seed):
    """The page's PRNG, bit-exact, so both sides grow the same core."""
    a = seed & 0xFFFFFFFF

    def rand():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = (a ^ (a >> 15)) * (1 | a) & 0xFFFFFFFF
        t = (t + ((t ^ (t >> 7)) * (61 | t) & 0xFFFFFFFF)) ^ t
        t &= 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    return rand


def _sig(v):
    if v < -60:
        return 0.0
    if v > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-v))


class LTC:
    """One liquid time-constant layer with an online-adapted readout."""

    def __init__(self, n=N_HIDDEN, m=N_INPUT, seed=SEED, eta=0.02):
        r = _mulberry32(seed)
        mat = lambda rows, cols, s: [[(r() * 2 - 1) * s for _ in range(cols)]
                                     for _ in range(rows)]
        self.n, self.m, self.eta = n, m, eta
        self.W = mat(n, n, 0.9 / math.sqrt(n))
        self.U = mat(n, m, 1.4 / math.sqrt(m))
        self.b = [(r() * 2 - 1) * 0.3 for _ in range(n)]
        self.tau = [0.3 + r() * 2.2 for _ in range(n)]     # 0.3 - 2.5 s
        self.A = [r() * 2 - 1 for _ in range(n)]
        self.x = [0.0] * n
        self.wo = [(r() * 2 - 1) * 0.1 for _ in range(n + m)]
        self.bo = 0.0
        # the second readout: ENGAGEMENT — is a student actually here and
        # working? Taught by the tracker's own seen flag, it lets the loop
        # pause its coaching clock when nobody is in front of the camera
        # instead of burning rounds against an empty room.
        self.we = [(r() * 2 - 1) * 0.1 for _ in range(n + m)]
        self.be = 0.0
        # the third readout: OUTCOME PREDICTION — "will this round be hit?"
        # It cannot be taught per frame (the label arrives only when the
        # round resolves), so the loop keeps the round's feature vectors
        # and replays them against the actual outcome via teach_outcome():
        # the network experiments on its own forecasts and learns from
        # being wrong. Applied: a confident predicted miss triggers the
        # coach EARLY instead of waiting out the timeout.
        self.wp = [(r() * 2 - 1) * 0.1 for _ in range(n + m)]
        self.bp = 0.0
        # the fourth readout: DETAIL NEED — "does THIS student need finer-
        # grained instruction?" Taught by replaying each resolved round's
        # trajectory against whether it took coached rounds to land
        # (teach_need), and distilled into a slow per-student trait
        # (self.need) that the page's interrupt path APPLIES: it sizes the
        # steps of every generated answer chapter, on any topic. The
        # self-experimentation loop reaching out of the camera and into
        # the explanations themselves.
        self.wd = [(r() * 2 - 1) * 0.1 for _ in range(n + m)]
        self.bd = 0.0
        self.y = 0.0
        self.y_eng = 0.0
        self.y_pred = 0.0
        self.y_need = 0.0
        self.need = 0.5               # the trait: slow EMA of y_need
        self.z = [0.0] * (n + m)      # the last feature vector, replayable

    def reset(self):
        self.x = [0.0] * self.n
        self.y = 0.0
        self.y_eng = 0.0
        self.y_pred = 0.0
        self.y_need = 0.0
        # self.need survives on purpose — it is a TRAIT, not a state

    def _features(self, u):
        """The readout's view: [gain*x ; u] — state plus skip connection."""
        return ([READOUT_GAIN * v for v in self.x]
                + [u[k] if k < len(u) else 0.0 for k in range(self.m)])

    def step(self, u, dt, teach=None, teach_eng=None):
        """One fused-solver step; optionally online readout updates."""
        dt = min(max(dt, 0.005), 0.1)
        f = []
        for i in range(self.n):
            a = self.b[i]
            a += sum(self.W[i][j] * self.x[j] for j in range(self.n))
            a += sum(self.U[i][k] * u[k]
                     for k in range(min(self.m, len(u))))
            f.append(_sig(a))
        for i in range(self.n):                # the fused ODE solver step
            self.x[i] = ((self.x[i] + dt * f[i] * self.A[i])
                         / (1 + dt * (1 / self.tau[i] + f[i])))
        z = self._features(u)
        self.z = z
        self.y = _sig(self.bo + sum(w * v for w, v in zip(self.wo, z)))
        self.y_eng = _sig(self.be + sum(w * v for w, v in zip(self.we, z)))
        self.y_pred = _sig(self.bp + sum(w * v for w, v in zip(self.wp, z)))
        self.y_need = _sig(self.bd + sum(w * v for w, v in zip(self.wd, z)))
        # the trait distils slowly from the per-frame readout — a few
        # seconds of struggle should not rewrite who the student is
        self.need = 0.98 * self.need + 0.02 * self.y_need
        if teach is not None:
            # online readout adaptation — cross-entropy gradient, which
            # does not vanish when the readout is confidently wrong (the
            # squared-error delta rule was measured to saturate here)
            g = self.eta * (teach - self.y)
            self.wo = [w + g * v for w, v in zip(self.wo, z)]
            self.bo += g
        if teach_eng is not None:
            g = self.eta * (teach_eng - self.y_eng)
            self.we = [w + g * v for w, v in zip(self.we, z)]
            self.be += g
        return self.y

    def teach_outcome(self, zs, label):
        """The delayed-label update: replay a round's stored feature
        vectors against how the round actually ended (1 hit, 0 missed).
        One cross-entropy step per vector — the self-experimentation loop
        closing over the network's own predictions."""
        for z in zs:
            p = _sig(self.bp + sum(w * v for w, v in zip(self.wp, z)))
            g = self.eta * (label - p)
            self.wp = [w + g * v for w, v in zip(self.wp, z)]
            self.bp += g

    def teach_need(self, zs, label):
        """Same replay discipline for the DETAIL-NEED readout: label 1
        when the round needed coaching to land (or never landed), 0 on a
        clean first-round hit."""
        for z in zs:
            p = _sig(self.bd + sum(w * v for w, v in zip(self.wd, z)))
            g = self.eta * (label - p)
            self.wd = [w + g * v for w, v in zip(self.wd, z)]
            self.bd += g

    def tau_effective(self, u):
        """The liquid property, measurable: tau/(1 + tau*f) per neuron."""
        f = []
        for i in range(self.n):
            a = self.b[i]
            a += sum(self.W[i][j] * self.x[j] for j in range(self.n))
            a += sum(self.U[i][k] * u[k]
                     for k in range(min(self.m, len(u))))
            f.append(_sig(a))
        return [self.tau[i] / (1 + self.tau[i] * f[i])
                for i in range(self.n)]


def _self_test():
    net = LTC()
    quiet = [0.0] * N_INPUT
    busy = [1.0] * 9 + [0.9, 1.0, 0.5, 0.0]

    # 1. the liquid property: busy input must tighten the time constants
    t_quiet = sum(net.tau_effective(quiet)) / net.n
    t_busy = sum(net.tau_effective(busy)) / net.n
    assert t_busy < t_quiet, (t_busy, t_quiet)

    # 2. bounded, finite dynamics under sustained drive
    for _ in range(2000):
        net.step(busy, 1 / 30)
    assert all(math.isfinite(v) and abs(v) < 10 for v in net.x), net.x

    # 3. online learning: both readouts separate their regimes.
    # Phases last ~3 s, as real pose phases do — shorter than the slowest
    # time constants and the states never reach the attractors they are
    # judged by, which is not how the loop is used.
    net = LTC()
    good = [0.05] * 9 + [0.9, 1.0, 0.5, 0.0]   # small deviations, high score
    bad = [0.8] * 9 + [0.3, 1.0, 0.0, 0.0]     # large deviations, low score
    gone = [0.0] * 9 + [0.0, 0.0, 0.0, 0.0]    # nobody in front of the camera
    for _ in range(400):
        for _ in range(90):
            net.step(good, 1 / 30, teach=1.0, teach_eng=1.0)
        for _ in range(90):
            net.step(bad, 1 / 30, teach=0.0, teach_eng=1.0)
        for _ in range(30):
            net.step(gone, 1 / 30, teach_eng=0.0)
    for _ in range(90):
        y_good = net.step(good, 1 / 30)
    e_good = net.y_eng
    for _ in range(90):
        y_bad = net.step(bad, 1 / 30)
    e_bad = net.y_eng
    for _ in range(60):
        net.step(gone, 1 / 30)
    e_gone = net.y_eng
    assert y_good > 0.9 and y_bad < 0.1, (y_good, y_bad)
    assert e_good > 0.8 and e_bad > 0.8 and e_gone < 0.3, \
        (e_good, e_bad, e_gone)      # engaged whether right or wrong;
    #                                  disengaged only when absent

    # 3b. the outcome predictor learns from its own forecasts: rounds where
    # the deviations shrink end hit (label 1), rounds stuck high end missed
    # (label 0). Labels arrive only at round end and are replayed over the
    # round's stored feature vectors — after training, the predictor
    # separates the two MID-round, early enough to act on.
    def round_(net_, kind, frames=60, teach=True):
        zs = []
        for i in range(frames):
            t = i / frames
            dev = (0.6 * (1 - t) if kind == "approach" else 0.7)
            u = [dev] * 9 + [1 - dev, 1.0, t if kind == "approach" else 0.0,
                             0.0]
            net_.step(u, 1 / 30)
            zs.append(list(net_.z))
        if teach:
            net_.teach_outcome(zs, 1.0 if kind == "approach" else 0.0)
            # a clean approach needed no coaching (0); a stuck round did (1)
            net_.teach_need(zs, 0.0 if kind == "approach" else 1.0)
    for _ in range(60):
        round_(net, "approach")
        round_(net, "stuck")
    round_(net, "approach", frames=30, teach=False)   # judged MID-round
    p_hit = net.y_pred
    round_(net, "stuck", frames=30, teach=False)
    p_miss = net.y_pred
    assert p_hit > 0.65 and p_miss < 0.35, (p_hit, p_miss)

    # 3c. the detail-need readout separates the same regimes with inverse
    # labels, mid-round — and the slow trait drifts UP through a run of
    # coached rounds, which is what sizes the page's answer chapters.
    round_(net, "stuck", frames=30, teach=False)
    n_stuck = net.y_need
    round_(net, "approach", frames=30, teach=False)
    n_clean = net.y_need
    assert n_stuck > 0.65 and n_clean < 0.35, (n_stuck, n_clean)
    before = net.need
    for _ in range(20):
        round_(net, "stuck", frames=30, teach=False)
    assert net.need > before, (before, net.need)
    after_need = net.need     # snapshot NOW — section 4's clean frames
    #                           will (correctly) pull the trait back down

    # 4. flicker robustness: confidence coasts through a two-frame tracker
    # dropout instead of collapsing — the whole reason the loop wants a
    # liquid state rather than an instantaneous threshold
    for _ in range(90):
        net.step(good, 1 / 30)
    held = net.y
    for _ in range(2):
        net.step(gone, 1 / 30)                     # tracker lost the body
    assert net.y > 0.6 and abs(net.y - held) < 0.3, (net.y, held)

    print("liquidnet self-test OK: "
          f"tau busy {t_busy:.2f}s < quiet {t_quiet:.2f}s, "
          f"split good={y_good:.2f}/bad={y_bad:.2f}, "
          f"engagement {e_good:.2f}/{e_bad:.2f}/gone {e_gone:.2f}, "
          f"outcome mid-round hit={p_hit:.2f}/miss={p_miss:.2f}, "
          f"detail-need stuck={n_stuck:.2f}/clean={n_clean:.2f} "
          f"(trait {before:.2f}->{after_need:.2f}), "
          f"state bounded, flicker bridged at y={net.y:.2f}")


if __name__ == "__main__":
    _self_test()
