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

N_HIDDEN = 12
N_INPUT = 11          # 9 joint-band deviations + score/100 + body-seen flag
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
        self.y = 0.0

    def reset(self):
        self.x = [0.0] * self.n
        self.y = 0.0

    def _features(self, u):
        """The readout's view: [gain*x ; u] — state plus skip connection."""
        return ([READOUT_GAIN * v for v in self.x]
                + [u[k] if k < len(u) else 0.0 for k in range(self.m)])

    def step(self, u, dt, teach=None):
        """One fused-solver step; optionally one online readout update."""
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
        self.y = _sig(self.bo + sum(w * v for w, v in zip(self.wo, z)))
        if teach is not None:
            # online readout adaptation — cross-entropy gradient, which
            # does not vanish when the readout is confidently wrong (the
            # squared-error delta rule was measured to saturate here)
            g = self.eta * (teach - self.y)
            self.wo = [w + g * v for w, v in zip(self.wo, z)]
            self.bo += g
        return self.y

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
    busy = [1.0] * 9 + [0.9, 1.0]

    # 1. the liquid property: busy input must tighten the time constants
    t_quiet = sum(net.tau_effective(quiet)) / net.n
    t_busy = sum(net.tau_effective(busy)) / net.n
    assert t_busy < t_quiet, (t_busy, t_quiet)

    # 2. bounded, finite dynamics under sustained drive
    for _ in range(2000):
        net.step(busy, 1 / 30)
    assert all(math.isfinite(v) and abs(v) < 10 for v in net.x), net.x

    # 3. online learning: the readout separates two input regimes.
    # Phases last ~3 s, as real pose phases do — shorter than the slowest
    # time constants and the states never reach the attractors they are
    # judged by, which is not how the loop is used.
    net = LTC()
    good = [0.05] * 9 + [0.9, 1.0]      # small deviations, high score
    bad = [0.8] * 9 + [0.3, 1.0]        # large deviations, low score
    for _ in range(400):
        for _ in range(90):
            net.step(good, 1 / 30, teach=1.0)
        for _ in range(90):
            net.step(bad, 1 / 30, teach=0.0)
    for _ in range(90):
        y_good = net.step(good, 1 / 30)
    for _ in range(90):
        y_bad = net.step(bad, 1 / 30)
    assert y_good > 0.9 and y_bad < 0.1, (y_good, y_bad)

    # 4. flicker robustness: confidence coasts through a two-frame tracker
    # dropout instead of collapsing — the whole reason the loop wants a
    # liquid state rather than an instantaneous threshold
    for _ in range(90):
        net.step(good, 1 / 30)
    held = net.y
    for _ in range(2):
        net.step([0.0] * 9 + [0.0, 0.0], 1 / 30)   # tracker lost the body
    assert net.y > 0.6 and abs(net.y - held) < 0.3, (net.y, held)

    print("liquidnet self-test OK: "
          f"tau busy {t_busy:.2f}s < quiet {t_quiet:.2f}s, "
          f"learned split good={y_good:.2f} bad={y_bad:.2f}, "
          f"state bounded, flicker bridged at y={net.y:.2f}")


if __name__ == "__main__":
    _self_test()
