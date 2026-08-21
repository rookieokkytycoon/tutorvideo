"""physics.py — the narrow expert that reads instructions and objects.

Corroboration counts how many sources SAID something. It cannot tell you
whether the thing is true: forty videos copying one wrong tutorial produce
forty units of agreement. This module is the other half — a check that does
not care how many people said it, only whether it is physically possible.

Every rule here is deterministic and cites the law it applies. That is
deliberate: a narrow expert whose verdicts cannot be audited is just another
opinion, and an opinion that overrides forty sources had better be able to
show its work. Nothing is probabilistic, nothing is learned, and a rule that
does not fire returns nothing rather than guessing.

    from physics import audit
    for v in audit("Turn the left pedal clockwise to remove it"):
        print(v.law, v.explain)

Severity drives what the graph does with a finding:
    "impossible"  the step cannot work as written    -> auto-dispute
    "unsafe"      it can work and may injure you     -> auto-dispute + warn
    "suspect"     probably wrong, context-dependent  -> flag for review only
"""

import re

# --------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------

# "counter-clockwise" CONTAINS "clockwise", so a naive search for the latter
# matches the former and inverts every verdict. Direction is therefore read by
# testing anti-clockwise FIRST and cutting those phrases out before looking for
# clockwise at all. Same class of bug as "disconnect" containing "connect" —
# every term below is anchored on \b for exactly that reason.
CW = r"(?:\bclockwise\b|\bcw\b|\brighty\b|\bto the right\b|\brightward\b)"
CCW = (r"(?:\bcounter[\s-]?clockwise\b|\banti[\s-]?clockwise\b|\bccw\b|"
       r"\blefty\b|\bto the left\b|\bleftward\b)")
LOOSEN = (r"(?:\bloosen\b|\bundo\b|\bunscrew\b|\bremove\b|\bback off\b|"
          r"\btake off\b|\bfree\b|\brelease\b|\bdetach\b|\bslacken\b|"
          r"\bextract\b)")
TIGHTEN = (r"(?:\btighten\b|\bscrew in\b|\bdo up\b|\binstall\b|\bfasten\b|"
           r"\bsecure\b|\btorque\b|\battach\b)")


def _directions(t):
    """-> (clockwise, counter_clockwise). Order matters; see CW/CCW above."""
    ccw = bool(re.search(CCW, t, re.I))
    cw = bool(re.search(CW, re.sub(CCW, " ", t, flags=re.I), re.I))
    return cw, ccw

# Parts with a genuine left-hand thread. Kept deliberately short: every entry
# is a case where reversing the rule is the documented, standard fact, not a
# maybe. A false "this is left-hand" would be worse than staying silent.
LEFT_HAND_THREAD = [
    (r"left\s+(?:side\s+)?pedal|non[\s-]?drive[\s-]?side\s+pedal",
     "the left bicycle pedal"),
    (r"(?:propane|lpg|butane|fuel\s+gas|acetylene)\s+(?:fitting|regulator|"
     r"hose|nut|connector)", "propane and fuel-gas fittings"),
    (r"(?:bench\s+)?grinder\s+left[\s-]?(?:hand\s+)?(?:wheel|nut|arbor)",
     "the left-hand wheel nut on a bench grinder"),
    (r"turnbuckle\s+(?:left|one)\s+end", "one end of a turnbuckle"),
]


def _has(pattern, text):
    return re.search(pattern, text, re.I) is not None


# --------------------------------------------------------------------------
# structured facts — parse once, then reason over the parse
# --------------------------------------------------------------------------
#
# Matching regexes against raw prose caps this module at a few dozen rules and
# makes every one of them brittle (see the counter-clockwise/disconnect bugs
# that inverted two verdicts before they were caught). Everything below pulls
# the instruction apart ONCE into the handful of things physics actually cares
# about — what part, what action, which way, how much of what unit — and the
# rules then read fields instead of re-parsing English.

UNITS = {
    # torque
    "nm": ("torque", 1.0), "n·m": ("torque", 1.0), "n-m": ("torque", 1.0),
    "newton metre": ("torque", 1.0), "newton meter": ("torque", 1.0),
    "lb-ft": ("torque", 1.3558), "ft-lb": ("torque", 1.3558),
    "lbft": ("torque", 1.3558), "ftlb": ("torque", 1.3558),
    "foot pound": ("torque", 1.3558), "pound foot": ("torque", 1.3558),
    "in-lb": ("torque", 0.113), "lb-in": ("torque", 0.113),
    "inch pound": ("torque", 0.113),
    # pressure
    "psi": ("pressure", 6894.76), "bar": ("pressure", 100000.0),
    "kpa": ("pressure", 1000.0), "mpa": ("pressure", 1e6),
    "pa": ("pressure", 1.0), "atm": ("pressure", 101325.0),
    # temperature handled separately (offset scales, not multiplicative)
    "c": ("temperature", 1.0), "°c": ("temperature", 1.0),
    "celsius": ("temperature", 1.0), "centigrade": ("temperature", 1.0),
    "f": ("temperature", None), "°f": ("temperature", None),
    "fahrenheit": ("temperature", None),
    # length / mass
    "mm": ("length", 0.001), "cm": ("length", 0.01), "m": ("length", 1.0),
    "in": ("length", 0.0254), "inch": ("length", 0.0254),
    "kg": ("mass", 1.0), "g": ("mass", 0.001), "lb": ("mass", 0.4536),
    # electrical
    "v": ("voltage", 1.0), "volt": ("voltage", 1.0),
    "a": ("current", 1.0), "amp": ("current", 1.0), "ampere": ("current", 1.0),
}
_UNIT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:-\s*(\d+(?:\.\d+)?)\s*)?"
    r"(n·m|n-m|nm|newton\s+met(?:re|er)s?|lb-?ft|ft-?lbs?|foot\s+pounds?|"
    r"pound\s+feet|in-?lbs?|lb-?in|inch\s+pounds?|psi|bar|kpa|mpa|pa|atm|"
    r"°?\s*c\b|celsius|centigrade|°?\s*f\b|fahrenheit|mm|cm|kg|lbs?\b|"
    r"amps?|amperes?|volts?|[va]\b|inch(?:es)?|in\b|m\b|g\b)",
    re.I)


def quantities(text):
    """Every measured value in the step, normalised to SI where possible.

    -> [{"raw", "value", "high", "unit", "dimension", "si"}]. A range
    ("torque to 20-25 Nm") keeps both ends, because a spec is usually a
    window and checking only the midpoint hides the interesting failures.
    """
    out = []
    for m in _UNIT_RE.finditer(text):
        lo, hi, raw_unit = m.group(1), m.group(2), m.group(3)
        key = re.sub(r"[\s°]", "", raw_unit.lower())
        key = {"ftlbs": "ft-lb", "ftlb": "ft-lb", "lbft": "lb-ft",
               "lbs": "lb", "amps": "amp", "amperes": "amp",
               "volts": "v", "inches": "inch", "newtonmetres": "nm",
               "newtonmeters": "nm", "footpounds": "foot pound",
               "inlbs": "in-lb", "inlb": "in-lb"}.get(key, key)
        dim_scale = UNITS.get(key)
        if not dim_scale:
            continue
        dim, scale = dim_scale
        val = float(lo)
        si = None
        if dim == "temperature":
            si = (val - 32.0) * 5.0 / 9.0 if scale is None else val
        elif scale:
            si = val * scale
        out.append({"raw": m.group(0).strip(), "value": val,
                    "high": float(hi) if hi else None,
                    "unit": key, "dimension": dim, "si": si})
    return out


ACTIONS = {
    "loosen": LOOSEN, "tighten": TIGHTEN,
    "heat": r"\bheat\w*\b|\btorch\b|\bwarm up\b|\bflame\b",
    "cool": r"\bcool\w*\b|\bchill\b|\bfreeze\b|\bice\b",
    "open": r"\bopen\b|\buncap\b|\bcrack\b(?!\s*ed)",
    "disconnect": r"\bdisconnect\w*\b|\bunhook\b|\bunplug\b",
    "connect": r"\bconnect\b|\bconnecting\b|\breconnect\w*\b|\bhook up\b",
    "mix": r"\bmix\b|\bcombine\b|\badd\b.*\bto\b|\bpour\b.*\binto\b",
    "lift": r"\blift\b|\braise\b|\bhoist\b|\bjack\b",
    "measure": r"\bmeasure\b|\bcheck\b|\bset\b|\btorque to\b|\binflate\b",
}


def facts(text):
    """One instruction -> the structured view every rule reads."""
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    cw, ccw = _directions(t)
    lh = next((name for pat, name in LEFT_HAND_THREAD if _has(pat, t)), None)
    acts = {k for k, pat in ACTIONS.items() if _has(pat, t)}
    if "disconnect" in acts:
        acts.discard("connect")          # "disconnect" contains "connect"
    return {"text": t, "clockwise": cw, "counter_clockwise": ccw,
            "left_hand_thread": lh, "actions": acts,
            "quantities": quantities(t)}


class Violation:
    """One physical objection to one instruction."""

    __slots__ = ("rule", "law", "severity", "explain", "suggest", "quote",
                 "confidence", "cite")

    def __init__(self, rule, law, severity, explain, suggest="", quote="",
                 confidence=1.0, cite=""):
        self.rule, self.law, self.severity = rule, law, severity
        self.explain, self.suggest, self.quote = explain, suggest, quote
        # A rule that is a published standard and a rule that is a range
        # heuristic must not carry the same weight when they overrule forty
        # sources. `cite` is what a reviewer checks; `confidence` is what
        # decides whether the graph acts or merely flags.
        self.confidence, self.cite = confidence, cite

    def as_dict(self):
        return {"rule": self.rule, "law": self.law, "severity": self.severity,
                "explain": self.explain, "suggest": self.suggest,
                "quote": self.quote, "confidence": self.confidence,
                "cite": self.cite}

    def __repr__(self):
        return f"<{self.severity}:{self.rule}>"


# --------------------------------------------------------------------------
# rules
# --------------------------------------------------------------------------

def _rule_thread_handedness(t):
    """Right-hand threads loosen counter-clockwise; left-hand ones do not.

    This is the single most-repeated wrong instruction in bicycle and small-
    engine content, because "righty-tighty" is stated as if it were universal.
    """
    lh = next((name for pat, name in LEFT_HAND_THREAD if _has(pat, t)), None)
    loosening, tightening = _has(LOOSEN, t), _has(TIGHTEN, t)
    cw, ccw = _directions(t)
    if not (cw or ccw) or not (loosening or tightening):
        return None
    if cw and ccw:            # both named — a comparison, not an instruction
        return None

    if lh:
        # left-hand thread: loosens clockwise, tightens counter-clockwise
        if loosening and ccw:
            return Violation(
                "thread_handedness",
                "Left-hand thread reverses the screw's helix direction",
                "impossible",
                f"This is {lh}, which has a LEFT-HAND thread. Turning it "
                f"counter-clockwise tightens it — the step as written jams "
                f"the part rather than removing it, and is how these get "
                f"seized or stripped.",
                "Turn it CLOCKWISE to loosen.")
        if tightening and cw:
            return Violation(
                "thread_handedness",
                "Left-hand thread reverses the screw's helix direction",
                "impossible",
                f"This is {lh} — a LEFT-HAND thread. Clockwise loosens it, "
                f"so it will never tighten this way.",
                "Turn it COUNTER-CLOCKWISE to tighten.")
        return None

    # ordinary right-hand thread
    if loosening and cw:
        return Violation(
            "thread_handedness",
            "Right-hand thread: counter-clockwise backs the helix out",
            "impossible",
            "A standard right-hand thread tightens clockwise. Turning "
            "clockwise to loosen drives the fastener further in.",
            "Turn it COUNTER-CLOCKWISE to loosen.")
    if tightening and ccw:
        return Violation(
            "thread_handedness",
            "Right-hand thread: clockwise drives the helix in",
            "impossible",
            "A standard right-hand thread loosens counter-clockwise, so it "
            "cannot be tightened that way.",
            "Turn it CLOCKWISE to tighten.")
    return None


def _rule_thermal_expansion(t):
    """Heat expands. Which member you heat decides whether a joint frees."""
    if not _has(r"seiz|stuck|rusted|frozen|corroded|won't budge|wont budge", t) \
            and not _has(r"interference|press[\s-]?fit|shrink[\s-]?fit", t):
        return None
    heating_inner = _has(r"heat(?:ing)?\s+(?:up\s+)?the\s+(?:bolt|screw|stud|"
                         r"shaft|pin|axle|inner)", t)
    cooling_outer = _has(r"(?:cool|chill|freeze|ice)\s+(?:down\s+)?the\s+"
                         r"(?:nut|housing|hub|collar|outer|bore|socket)", t)
    if heating_inner:
        return Violation(
            "thermal_expansion",
            "Thermal expansion: a heated part grows into its mating surface",
            "impossible",
            "Heating the inner member (the bolt or shaft) expands it INTO "
            "the thread or bore, tightening the joint. Seized fasteners are "
            "freed by expanding the outer member away from the inner one.",
            "Heat the NUT or housing instead, not the bolt.")
    if cooling_outer:
        return Violation(
            "thermal_expansion",
            "Thermal contraction: a cooled part shrinks onto what it holds",
            "impossible",
            "Cooling the outer member shrinks it onto the inner one, "
            "gripping harder.",
            "Heat the outer member, or cool the INNER one, to free it.")
    return None


def _rule_battery_order(t):
    """Chassis ground makes terminal order a short-circuit question."""
    if not _has(r"batter|terminal", t):
        return None
    # \b matters twice over here: "disconnect" contains "connect", and
    # mistaking one for the other reverses the safe order exactly.
    disconnecting = _has(r"\bdisconnect\w*|\bremove\b|\bundo\b|\bdetach\b|"
                         r"\btake off\b|\bunhook\b", t)
    connecting = _has(r"\bconnect\b|\bconnecting\b|\breconnect\w*|\battach\b|"
                      r"\binstall\b|\brefit\b|\bput back\b|\bhook up\b", t)
    if disconnecting:
        connecting = False        # "disconnect X then Y" is one operation
    pos_first = _has(r"(?:positive|\+|red)\s+(?:terminal\s+)?(?:cable\s+|"
                     r"lead\s+|clamp\s+)?first", t)
    neg_first = _has(r"(?:negative|\-|black|ground|earth)\s+(?:terminal\s+)?"
                     r"(?:cable\s+|lead\s+|clamp\s+)?first", t)
    if disconnecting and pos_first:
        return Violation(
            "battery_terminal_order",
            "The chassis is bonded to the negative terminal",
            "unsafe",
            "With the negative still connected, the whole vehicle body is at "
            "battery negative. A spanner touching any metal while on the "
            "positive post completes a dead short across the battery — high "
            "current, arcing and a burst-battery risk.",
            "Disconnect the NEGATIVE terminal first.")
    if connecting and neg_first:
        return Violation(
            "battery_terminal_order",
            "The chassis is bonded to the negative terminal",
            "unsafe",
            "Fitting the negative first re-bonds the chassis, so any slip "
            "while fitting the positive shorts through the body.",
            "Connect the POSITIVE terminal first, negative last.")
    return None


def _rule_pressure_before_opening(t):
    """Pressure raises boiling point; releasing it flashes liquid to steam."""
    hot_pressurised = _has(r"(?:radiator|coolant|expansion\s+tank|pressure)\s*"
                           r"(?:cap|valve)?", t) and _has(r"hot|warm|running|"
                                                          r"just\s+driven", t)
    if hot_pressurised and _has(r"open|remove|undo|unscrew|loosen", t):
        return Violation(
            "pressure_before_opening",
            "Pressure raises the boiling point of the coolant",
            "unsafe",
            "A hot cooling system holds coolant above 100 °C as a liquid "
            "ONLY because it is pressurised. Releasing that pressure boils it "
            "instantly and ejects scalding coolant and steam.",
            "Let it cool before opening, or release pressure remotely.")
    if _has(r"fuel\s+(?:line|rail|hose|filter|injector)", t) and \
            _has(r"open|disconnect|remove|undo|crack", t) and \
            not _has(r"depressuris|depressuriz|relieve|bleed|release\s+the\s+"
                     r"pressure|fuse|pump\s+off", t):
        return Violation(
            "pressure_before_opening",
            "A fuel rail stays pressurised after the engine stops",
            "unsafe",
            "Fuel injection holds residual pressure with the engine off. "
            "Opening the line sprays atomised fuel.",
            "Relieve fuel pressure first.")
    return None


def _rule_stored_spring_energy(t):
    """A compressed spring is stored energy; the retainer is what holds it."""
    spring = _has(r"spring|strut|shock\s+absorber|coil[\s-]?over|valve\s+"
                  r"spring|clutch\s+spring", t)
    releasing = _has(r"remove|undo|unbolt|take off|release|unscrew|pop off", t)
    retainer = _has(r"retain(?:er|ing)|circlip|c[\s-]?clip|top\s+nut|"
                    r"gland\s+nut|collar|cap", t)
    guarded = _has(r"compressor|compress|relieve|decompress|unload|"
                   r"after\s+releasing|once\s+the\s+tension", t)
    if spring and releasing and retainer and not guarded:
        return Violation(
            "stored_spring_energy",
            "Elastic potential energy is released the instant it is unretained",
            "unsafe",
            "The retainer is the only thing holding the spring's stored "
            "energy. Removing it while the spring is compressed releases that "
            "energy in one motion.",
            "Fit a spring compressor and relieve the tension first.")
    return None


def _rule_lever_torque(t):
    """tau = F x r. A shorter lever cannot give more torque for the same force."""
    if _has(r"short(?:er)?\s+(?:wrench|spanner|bar|lever|breaker)", t) and \
            _has(r"more\s+(?:torque|leverage|force)|extra\s+(?:torque|"
                 r"leverage)|easier\s+to\s+(?:undo|loosen|break)", t):
        return Violation(
            "lever_torque",
            "Torque = force x lever arm (tau = F x r)",
            "impossible",
            "Torque is the applied force times the distance from the axis. A "
            "shorter wrench REDUCES torque for the same hand force.",
            "Use a LONGER bar for more torque.")
    return None


def _rule_siphon_and_gravity(t):
    """Fluid does not flow uphill unless something drives it."""
    if _has(r"siphon|drain", t) and _has(r"above|higher\s+than|uphill|"
                                         r"raise\s+the\s+(?:outlet|hose\s+end)", t) \
            and not _has(r"pump|pressuris|pressuriz|vacuum", t):
        return Violation(
            "gravity_flow",
            "A gravity siphon needs the outlet below the source surface",
            "impossible",
            "A siphon is driven by the height difference between the source "
            "surface and the outlet. With the outlet above the source, flow "
            "stops.",
            "Put the outlet BELOW the level of the liquid being drained.")
    return None


def _rule_chemical_incompatibility(t):
    """Some mixtures produce a toxic gas. This is not a judgement call."""
    PAIRS = [
        (r"\bbleach\b|\bhypochlorite\b", r"\bammonia\b|\bwindow cleaner\b",
         "chloramine gas", "Bleach + ammonia releases chloramine vapour."),
        (r"\bbleach\b|\bhypochlorite\b", r"\bvinegar\b|\backd?etic acid\b|"
         r"\bacid\b|\bdescal\w+\b", "chlorine gas",
         "Bleach + acid releases chlorine gas."),
        (r"\bhydrogen peroxide\b", r"\bvinegar\b|\bacetic acid\b",
         "peracetic acid", "Peroxide + vinegar forms corrosive peracetic acid."),
    ]
    if not _has(ACTIONS["mix"], t):
        return None
    for a, b, gas, why in PAIRS:
        if _has(a, t) and _has(b, t):
            return Violation(
                "chemical_incompatibility",
                f"Reaction produces {gas}",
                "unsafe",
                f"{why} It is toxic at low concentration and this reaction "
                f"happens immediately at room temperature.",
                "Never combine these. Ventilate and use one product alone.",
                confidence=1.0,
                cite="US CDC / NIOSH guidance on household chemical mixing")
    return None


def _rule_live_circuit(t):
    """Work on a de-energised circuit, verified dead — not assumed dead."""
    working = _has(r"\bcut\b|\bstrip\b|\bsplice\b|\breplace\b|\brewire\b|"
                   r"\bwork(?:ing)? on\b|\btouch\b", t)
    live = _has(r"\blive\b|\benergi[sz]ed\b|\bpower(?:ed)? on\b|"
                r"\bwithout (?:switching|turning) (?:it )?off\b|"
                r"\bbreaker (?:still )?on\b", t)
    circuit = _has(r"\bwire\b|\bwiring\b|\bcircuit\b|\bmains\b|\boutlet\b|"
                   r"\bsocket\b|\bconsumer unit\b|\bbreaker\b|\bfuse box\b", t)
    if working and live and circuit:
        return Violation(
            "live_circuit",
            "Current follows any path to earth, including through a person",
            "unsafe",
            "Mains voltage across the body is lethal well below the current "
            "a domestic breaker will trip at; an RCD is a backstop, not a "
            "permission.",
            "Isolate at the breaker, lock it off, and prove dead with a "
            "tester before touching the conductors.",
            confidence=1.0,
            cite="Safe isolation procedure (lock-out/tag-out)")
    return None


def _rule_torque_plausibility(t):
    """A torque figure orders of magnitude off spec is a transcription error."""
    f = facts(t)
    if "measure" not in f["actions"] and "tighten" not in f["actions"]:
        return None
    for q in f["quantities"]:
        if q["dimension"] != "torque" or q["si"] is None:
            continue
        # small fasteners on bicycles/electronics live in 1-60 Nm; wheel nuts
        # and crank bolts reach ~250 Nm. Beyond 1500 Nm is heavy industrial and
        # essentially never appears in a consumer how-to.
        if q["si"] > 1500:
            return Violation(
                "torque_plausibility",
                "Torque figure far outside any hand-tool range",
                "suspect",
                f"{q['raw']} is {q['si']:.0f} N·m — beyond what a hand tool "
                f"or consumer fastener can take. Usually a unit mix-up "
                f"(N·m written where in-lb was meant) or a typo.",
                "Check the figure and its unit against the service spec.",
                confidence=0.6, cite="range heuristic, not a published spec")
        if 0 < q["si"] < 0.2:
            return Violation(
                "torque_plausibility",
                "Torque figure below any meaningful fastener spec",
                "suspect",
                f"{q['raw']} is {q['si']:.3f} N·m — far below a real "
                f"tightening spec.",
                "Check the unit; this looks like a decimal or unit slip.",
                confidence=0.6, cite="range heuristic, not a published spec")
    return None


def _rule_water_boiling(t):
    """Water does not exceed 100 °C at atmospheric pressure."""
    f = facts(t)
    if not _has(r"\bwater\b|\bcoolant\b", t) or "heat" not in f["actions"]:
        return None
    if _has(r"pressuri[sz]|sealed|autoclave|under pressure", t):
        return None
    for q in f["quantities"]:
        if q["dimension"] == "temperature" and q["si"] and q["si"] > 100.5:
            return Violation(
                "phase_change",
                "Water boils at 100 °C at atmospheric pressure",
                "impossible",
                f"{q['raw']} is above water's boiling point at ambient "
                f"pressure — it turns to steam rather than reaching that "
                f"temperature as a liquid.",
                "Either the system must be pressurised, or the figure is wrong.",
                confidence=0.9, cite="phase diagram of water at 1 atm")
    return None


RULES = (_rule_thread_handedness, _rule_thermal_expansion, _rule_battery_order,
         _rule_pressure_before_opening, _rule_stored_spring_energy,
         _rule_lever_torque, _rule_siphon_and_gravity,
         _rule_chemical_incompatibility, _rule_live_circuit,
         _rule_torque_plausibility, _rule_water_boiling)

AUTO_DISPUTE = ("impossible", "unsafe")   # "suspect" is flagged, never acted on


# --------------------------------------------------------------------------
# api
# --------------------------------------------------------------------------

def audit(text):
    """One instruction -> [Violation]. Empty when nothing objects."""
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    if not t:
        return []
    out = []
    for rule in RULES:
        try:
            v = rule(t)
        except re.error:
            continue
        if v:
            v.quote = t[:160]
            out.append(v)
    return out


def audit_steps(steps):
    """[str] -> [{"index", "text", "violations":[dict]}] for offending steps."""
    found = []
    for i, s in enumerate(steps or []):
        vs = audit(s)
        if vs:
            found.append({"index": i, "text": str(s)[:200],
                          "violations": [v.as_dict() for v in vs]})
    return found


def audit_doc(doc):
    """A mined doc -> the same list, over its steps."""
    return audit_steps([s.get("text") if isinstance(s, dict) else s
                        for s in (doc.get("steps") or [])])


def dispute_in_graph(hm, tid, steps=None, graph_path=None):
    """Run the audit over a task already in the graph and open disputes.

    This is the join between the two halves: corroboration says how many
    sources agreed, physics says whether they can all be right. A physics
    dispute is authored by "physics" and carries the law, so a reviewer can
    check the reasoning rather than trusting the label.
    """
    sids = steps if steps is not None else hm._steps_of(tid)
    opened = []
    for sid in sids:
        text = hm.g.nodes[sid].get("text", "")
        for v in audit(text):
            # only act on a finding that is both serious AND well-grounded;
            # a range heuristic gets recorded for review, never auto-disputed
            if v.severity not in AUTO_DISPUTE or v.confidence < 0.85:
                continue
            cid = hm.correct(
                sid, kind="wrong", author="physics", auto=True,
                text=f"[{v.law}] {v.explain}",
                replacement=v.suggest or "")
            hm.g.nodes[cid]["rule"] = v.rule
            hm.g.nodes[cid]["severity"] = v.severity
            opened.append({"correction": cid, "step": sid, **v.as_dict()})
    if opened and graph_path:
        hm.save(graph_path)
    return opened


if __name__ == "__main__":
    CASES = [
        "Turn the left pedal clockwise to remove it",
        "Turn the left pedal counter-clockwise to remove it",
        "Loosen the axle nut by turning it clockwise",
        "Loosen the axle nut by turning it counter-clockwise",
        "Heat the bolt with a torch to free the seized fastener",
        "Heat the nut with a torch to free the seized fastener",
        "Disconnect the positive terminal first, then the negative",
        "Disconnect the negative terminal first, then the positive",
        "Open the radiator cap while the engine is still hot",
        "Undo the strut top nut and remove the retaining collar",
        "Undo the strut top nut with a spring compressor fitted",
        "Use a shorter wrench for more torque on a stubborn bolt",
        "Pinch the chain between thumb and finger",
    ]
    for c in CASES:
        vs = audit(c)
        if vs:
            v = vs[0]
            print(f"  {v.severity.upper():10} {c}")
            print(f"             law: {v.law}")
            print(f"             fix: {v.suggest}")
        else:
            print(f"  {'ok':10} {c}")
