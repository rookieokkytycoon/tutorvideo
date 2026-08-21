"""screenread.py — read what the video SHOWS, not only what it says.

openclaw.py mines the transcript. But the numbers that decide whether a repair
is done correctly are almost never spoken: a torque spec, a part number, a
clearance, a capacity, a warning label. They are put on screen as text, and a
transcript-only miner throws all of it away.

This is the other half of the crawl — OCR over sampled frames, deduplicated,
with the measured quantities pulled out and normalised by physics.quantities().
Those numbers are what turn the physics layer from directional ("tighten it")
into quantitative ("35 Nm, and the spec says 25").

Runs on rapidocr-onnxruntime: pip-only, no system binary, CPU.

    python screenread.py "https://www.youtube.com/watch?v=..." 30
"""

import os
import re
import sys

SAMPLE_FPS = 0.5        # one frame every 2s; OCR is the slow part, not decode
MAX_SECONDS = 120
MIN_SCORE = 0.55        # rapidocr confidence floor
MIN_CHARS = 3

# Text that appears on nearly every how-to video and carries no procedure
# information. Dropping it here keeps the extractor's context window for the
# things that matter.
CHROME = re.compile(
    r"^(?:subscribe|like|share|comment|follow|click|link in (?:the )?"
    r"(?:bio|description)|patreon|instagram|facebook|twitter|tiktok|"
    r"youtube|www\.|https?:|@\w+|\d{1,2}:\d{2}|hd|cc|live|ad|sponsored)",
    re.I)


class ScreenError(Exception):
    pass


def available():
    """-> (bool, reason)."""
    try:
        import rapidocr_onnxruntime          # noqa: F401
        import cv2                           # noqa: F401
    except ImportError as e:
        return False, f"pip install rapidocr-onnxruntime opencv-python ({e})"
    return True, ""


_ENGINE = None


def _engine():
    global _ENGINE
    if _ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR
        _ENGINE = RapidOCR()
    return _ENGINE


def _clean(s):
    s = re.sub(r"\s+", " ", str(s or "")).strip()
    # OCR routinely confuses these inside otherwise-numeric strings
    if re.search(r"\d", s):
        s = re.sub(r"(?<=\d)[OoIl](?=\d)", lambda m: "0" if m.group(0) in "Oo"
                   else "1", s)
    return s


def read_frames(path, start=0.0, seconds=MAX_SECONDS, fps=SAMPLE_FPS):
    """OCR sampled frames of a local video. -> [{"t", "text", "score"}].

    Deduplicated: a caption burned on screen for ten seconds is one finding,
    not twenty, and the first timestamp it appeared at is the useful one.
    """
    ok, why = available()
    if not ok:
        raise ScreenError(f"OCR unavailable — {why}")
    import cv2
    from posetrack import _frames

    eng, seen, out = _engine(), {}, []
    for ms, img in _frames(path, start, seconds, fps):
        h = img.shape[0]
        if h > 720:                       # OCR gains nothing above 720p
            scale = 720.0 / h
            img = cv2.resize(img, None, fx=scale, fy=scale)
        try:
            res, _ = eng(img)
        except Exception:
            continue
        for item in (res or []):
            if len(item) < 3:
                continue
            _box, txt, score = item[0], item[1], item[2]
            txt = _clean(txt)
            if (not txt or len(txt) < MIN_CHARS or float(score) < MIN_SCORE
                    or CHROME.search(txt)):
                continue
            key = re.sub(r"[^a-z0-9]", "", txt.lower())
            if not key or key in seen:
                continue
            seen[key] = True
            out.append({"t": round(ms / 1000.0, 1), "text": txt,
                        "score": round(float(score), 3)})
    return out


def specs(readings):
    """Pull the measured values out of OCR text. -> [{"t","text",...quantity}]

    This is the payload: a torque figure shown on screen and never spoken is
    invisible to a transcript miner and is exactly what the physics layer
    needs to check a step quantitatively.
    """
    import physics
    out = []
    for r in readings:
        for q in physics.quantities(r["text"]):
            out.append({"t": r["t"], "text": r["text"], **q})
    return out


def read_url(url, start=0.0, seconds=MAX_SECONDS, fps=SAMPLE_FPS):
    """YouTube URL -> {"readings", "specs"}. Reuses posetrack's video cache."""
    import posetrack
    ok, why = posetrack.available()
    if not ok:
        raise ScreenError(f"cannot fetch video — {why}")
    path = posetrack.download(url)
    readings = read_frames(path, start, seconds, fps)
    return {"readings": readings, "specs": specs(readings),
            "video": os.path.basename(path)}


def as_context(readings, limit=40):
    """Fold OCR findings into a block the extractor can read alongside the
    transcript, timestamped so a step can be tied to what was on screen."""
    lines = []
    for r in readings[:limit]:
        m, s = int(r["t"]) // 60, int(r["t"]) % 60
        lines.append(f"[{m}:{s:02d}] {r['text']}")
    return "\n".join(lines)


def _main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    ok, why = available()
    print("OCR available:", ok, "" if ok else f"({why})")
    if not ok:
        return 1
    start = float(argv[2]) if len(argv) > 2 else 0.0
    out = read_url(argv[1], start=start, seconds=40)
    print(f"\n{len(out['readings'])} distinct on-screen strings:")
    for r in out["readings"][:25]:
        print(f"  [{r['t']:>6.1f}s] {r['text'][:70]}")
    print(f"\n{len(out['specs'])} measured values found on screen:")
    for s in out["specs"]:
        si = f" = {s['si']:.2f} SI" if s["si"] is not None else ""
        print(f"  [{s['t']:>6.1f}s] {s['raw']}  ({s['dimension']}{si})")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
