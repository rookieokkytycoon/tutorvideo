"""posetrack.py — the pixels, not the words.

openclaw.py mines what a video *says*. This mines what the person in it
actually *does*: BlazePose over the frames for the body, MediaPipe Hands for
the fingers, both landed straight into the clip contract the board already
renders.

    YouTube URL -> yt-dlp (cached, low-res)
                -> PoseLandmarker  33 world landmarks -> motion.clip_from_landmarks
                -> HandLandmarker  21 world landmarks -> handform.clip_from_frames

The landmarks come back in metres in both cases, which is the space these two
modules already work in, so nothing is re-authored on the way through —
gestures are recognised from the tracked motion rather than from the text,
and a clip's `backend` field says which route produced it.

MediaPipe 1.x removed `mp.solutions`, so everything here uses the Tasks API
and downloads its own .task models on first use.

    python posetrack.py "https://www.youtube.com/watch?v=..."
"""

import os
import shutil
import sys

import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "models")
CACHE_DIR = os.path.join(ROOT, ".videocache")

MODELS = {
    "pose": ("pose_landmarker.task",
             "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"),
    "hand": ("hand_landmarker.task",
             "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task"),
}

TRACK_FPS = 12          # what the board plays at; no reason to track faster
MAX_SECONDS = 20        # a chapter is seconds long, so never track minutes


class TrackError(Exception):
    """Anything that stops a video becoming a track, phrased for a user."""


def available():
    """Is real tracking possible in this interpreter? -> (bool, reason)."""
    try:
        import mediapipe            # noqa: F401
        import cv2                  # noqa: F401
    except ImportError as e:
        return False, f"pip install mediapipe opencv-python ({e})"
    try:
        import yt_dlp               # noqa: F401
    except ImportError:
        return False, "pip install yt-dlp"
    return True, ""


def ensure_model(kind):
    """Model path, downloading it on first use. Tasks needs a local file."""
    name, url = MODELS[kind]
    path = os.path.join(MODEL_DIR, name)
    if os.path.exists(path) and os.path.getsize(path) > 100_000:
        return path
    os.makedirs(MODEL_DIR, exist_ok=True)
    try:
        r = requests.get(url, timeout=300, stream=True)
        r.raise_for_status()
        tmp = path + ".part"
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        os.replace(tmp, path)
    except requests.RequestException as e:
        raise TrackError(f"could not download the {kind} model: {e}")
    return path


# --------------------------------------------------------------- download

def download(url, max_height=480):
    """Fetch a video once and keep it. -> local path.

    Cached by video id: mining the same video for body and for hands must
    not download it twice, and re-mining it later must not download it at
    all. Without ffmpeg yt-dlp cannot cut a section, so the whole file comes
    down and the trackers seek into it instead.
    """
    try:
        import yt_dlp
    except ImportError:
        raise TrackError("yt-dlp is not installed")
    os.makedirs(CACHE_DIR, exist_ok=True)

    from openclaw import video_id
    vid = video_id(url)
    for ext in ("mp4", "webm", "mkv"):
        hit = os.path.join(CACHE_DIR, f"{vid}.{ext}")
        if os.path.exists(hit) and os.path.getsize(hit) > 10_000:
            return hit

    have_ffmpeg = bool(shutil.which("ffmpeg"))
    opts = {
        "outtmpl": os.path.join(CACHE_DIR, "%(id)s.%(ext)s"),
        "quiet": True, "no_warnings": True, "noprogress": True,
        "format": (f"bestvideo[height<={max_height}][ext=mp4]+bestaudio/"
                   f"best[height<={max_height}][ext=mp4]/"
                   f"best[height<={max_height}]/best")
        if have_ffmpeg else
                  (f"best[height<={max_height}][ext=mp4]/"
                   f"best[height<={max_height}]/best[ext=mp4]/best"),
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={vid}",
                                    download=True)
            path = ydl.prepare_filename(info)
    except Exception as e:
        raise TrackError(f"could not download the video: "
                         f"{str(e).splitlines()[0][:180]}")
    if not os.path.exists(path):
        for ext in ("mp4", "webm", "mkv"):
            alt = os.path.join(CACHE_DIR, f"{vid}.{ext}")
            if os.path.exists(alt):
                return alt
        raise TrackError("the download produced no file")
    return path


# ----------------------------------------------------------------- track

def _frames(path, start=0.0, seconds=MAX_SECONDS, fps=TRACK_FPS):
    """Decode a window of a video at a fixed rate. Yields (ms, BGR frame)."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise TrackError(f"cannot open the downloaded video: {path}")
    try:
        src = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if start > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
        step = max(int(round(src / float(fps))), 1)
        want = int(seconds * fps)
        i = got = 0
        while got < want:
            ok, img = cap.read()
            if not ok:
                break
            if i % step:
                i += 1
                continue
            i += 1
            yield int((start + got / float(fps)) * 1000), img
            got += 1
    finally:
        cap.release()


def _detector(kind):
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    cls, opt = ((vision.PoseLandmarker, vision.PoseLandmarkerOptions)
                if kind == "pose" else
                (vision.HandLandmarker, vision.HandLandmarkerOptions))
    kw = {} if kind == "pose" else {"num_hands": 1}
    return mp, cls.create_from_options(opt(
        base_options=BaseOptions(model_asset_path=ensure_model(kind)),
        running_mode=vision.RunningMode.VIDEO, **kw))


def track_body(path, start=0.0, seconds=MAX_SECONDS, fps=TRACK_FPS):
    """-> [[ (x,y,z) x 33 ], ...] in BlazePose world metres. May be short."""
    import cv2
    mp, det = _detector("pose")
    out = []
    try:
        for ms, img in _frames(path, start, seconds, fps):
            res = det.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), ms)
            lms = getattr(res, "pose_world_landmarks", None)
            if not lms:
                continue
            out.append([(p.x, p.y, p.z) for p in lms[0]])
    finally:
        det.close()
    return out


def track_hand(path, start=0.0, seconds=MAX_SECONDS, fps=TRACK_FPS):
    """-> [[ (x,y,z) x 21 ], ...] wrist-relative metres, handform's own space."""
    import cv2
    mp, det = _detector("hand")
    out = []
    try:
        for ms, img in _frames(path, start, seconds, fps):
            res = det.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), ms)
            lms = getattr(res, "hand_world_landmarks", None)
            if not lms:
                continue
            out.append([(p.x, -p.y, p.z) for p in lms[0]])   # y down -> y up
    finally:
        det.close()
    return out


MIN_FRAMES = 6          # fewer than this is noise, not a demonstration


def video_to_skeleton(path, start=0.0, seconds=MAX_SECONDS, steps=None,
                      title="", source="", fps=TRACK_FPS):
    """Tracked body -> the same skeleton clip the synthesiser makes."""
    import motion
    raw = track_body(path, start, seconds, fps)
    if len(raw) < MIN_FRAMES:
        raise TrackError(f"no body visible in that part of the video "
                         f"({len(raw)} usable frames)")
    return motion.clip_from_landmarks(
        motion.from_landmarks(raw), fps=fps, steps=steps, title=title,
        source=source, backend="blazepose-tracked")


def video_to_hand(path, start=0.0, seconds=MAX_SECONDS, steps=None,
                  title="", source="", fps=TRACK_FPS):
    """Tracked fingers -> the same hand clip the synthesiser makes."""
    import handform
    raw = track_hand(path, start, seconds, fps)
    if len(raw) < MIN_FRAMES:
        raise TrackError(f"no hand visible in that part of the video "
                         f"({len(raw)} usable frames)")
    # handform.plan_from_steps, not a literal dict: clip_from_frames reads a
    # "pose" off every plan entry, and only the planner sets it (it classifies
    # the pose from the step text). Building the entries here by hand raised
    # KeyError: 'pose' for every tracked hand chapter that carried steps —
    # which is every mined chapter with tracking on.
    plan = handform.plan_from_steps(steps, len(raw) / float(fps))[0] \
        if steps else None
    return handform.clip_from_frames(raw, fps, title=title, source=source,
                                     backend="mediapipe-tracked", plan=plan)


def track_url(url, start=0.0, seconds=MAX_SECONDS, want=("body", "hand"),
              steps=None, title="", source=""):
    """URL -> {"skeleton": clip|None, "hand": clip|None, "errors": {...}}.

    Never raises for a rig that simply was not visible: a video of a bench
    with two hands in it has no body to track, and that is an ordinary
    outcome rather than a failure of the whole lesson.
    """
    ok, why = available()
    if not ok:
        raise TrackError(f"real tracking is not installed — {why}")
    path = download(url)
    out, errs = {}, {}
    if "body" in want:
        try:
            out["skeleton"] = video_to_skeleton(path, start, seconds, steps,
                                                title, source)
        except (TrackError, Exception) as e:
            errs["skeleton"] = str(e)[:200]
    if "hand" in want:
        try:
            out["hand"] = video_to_hand(path, start, seconds, steps,
                                        title, source)
        except (TrackError, Exception) as e:
            errs["hand"] = str(e)[:200]
    out["errors"] = errs
    return out


def _main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    ok, why = available()
    print("tracking available:", ok, "" if ok else f"({why})")
    if not ok:
        return 1
    start = float(argv[2]) if len(argv) > 2 else 0.0
    print("downloading…")
    path = download(argv[1])
    print("  ->", path, os.path.getsize(path), "bytes")
    for name, fn in (("body", video_to_skeleton), ("hand", video_to_hand)):
        try:
            clip = fn(path, start=start, seconds=8, title=f"tracked {name}")
            print(f"  {name}: {clip['kind']} backend={clip['backend']} "
                  f"frames={len(clip['frames'])} "
                  f"gestures={[g['name'] for g in clip.get('gestures', [])]}")
        except Exception as e:
            print(f"  {name}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
