# Agentic Video Tutor — Python backend + hivemind graph

One Python backend serves the website, proxies the Claude API,
and hosts the how-to knowledge graph ("hivemind") the tutor answers from.

    pyweb/
      server.py         Flask: page + /api/claude proxy + /api/graph + /api/ingest
      hivemind.py       knowledge graph: ingest, cross-link, retrieve, persist
      coach.py          the return path: your camera scored against the rig
      index.html        the full app (player, interrupts, agentic chapters, recording)
      mpt/              MoneyPrinterTurbo, vendored and imported as a library:
                        the neural voice, stock step footage and the MP4 export
      requirements.txt  flask, requests, networkx, gunicorn

## Hivemind endpoints

    GET  /health              -> includes graph node/edge counts
    POST /api/graph           {"question": "..."}
                              -> {"context": grounded how-to text or ""}
    POST /api/ingest          {"title": "...", "text": "1. step...\n2. step..."}
                              -> parses steps/tools/warnings, grows the graph,
                                 saves hivemind_graph.json

The frontend automatically calls /api/graph before every interruption
and injects the retrieved how-to into the tutor's answer — grounded
answers with source titles, and honest "not covered yet" otherwise.

Grow it at scale: point an OpenClaw crawler skill (or any script) at
/api/ingest — one POST per article. For messy sources (YouTube
transcripts), use hivemind.extract_with_llm() instead of the plain parser.

## Run locally (2 minutes)

    cd pyweb
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=sk-ant-...     # from console.anthropic.com
    python server.py

Open http://localhost:8000 — press Demo, then try a real topic.
Windows: use  set ANTHROPIC_API_KEY=sk-ant-...  instead of export.

Sanity check without opening a browser:
    curl localhost:8000/health
    -> {"key_configured": true, "ok": true}

## Put it on the public internet

Any host that runs Python works. The pattern is always:
set the env var, run gunicorn.

    gunicorn -w 2 -b 0.0.0.0:8000 server:app

- **Render / Railway** (easiest): new Web Service from this folder,
  start command = the gunicorn line above, add ANTHROPIC_API_KEY in
  the dashboard's environment settings. You get an https URL.
- **Fly.io**: `fly launch`, `fly secrets set ANTHROPIC_API_KEY=...`.
- **Your own VPS**: run gunicorn behind nginx/caddy for TLS.

## Built-in protections

- Key lives only in the server environment; the page calls /api/claude.
- 60 KB request cap and a model allowlist (claude-sonnet-4-6 /
  claude-haiku-4-5) so strangers can't pump arbitrary traffic
  through your key.
- Clean JSON errors surface in the app's status line
  (missing key, upstream down, oversized request).

Still recommended before sharing the URL widely:
- Set a monthly spend limit at console.anthropic.com -> Billing.
- Add rate limiting if traffic grows, e.g. flask-limiter:
      pip install flask-limiter
  and decorate the proxy route with @limiter.limit("30/minute").

## Costs

Every lesson ≈ 1 director call + 1 sketch call per chapter (3-4);
every interruption ≈ 1 more call; an agentic new chapter ≈ 2.
All on your key. Pricing: https://docs.claude.com/en/api/overview

## MoneyPrinterTurbo: the voice, the footage and the file  (`mpt/`)

Two of this README's own "next steps" are now done, and neither is a new
service to deploy: MoneyPrinterTurbo is vendored whole in `mpt/` and imported
as a **library**. Nothing else runs — no FastAPI, no Streamlit, no second port.

    POST /api/tts      {"text": "...", "lang": "en"|"id"}
      -> {"audio": "/api/mpt/media?f=...", "seconds": 2.45,
          "cues": [{"t": 0.1, "d": 0.25, "w": "Grip"}, ...]}

    POST /api/stock    {"terms": ["seat the chain", ...], "seconds": 5}
      -> {"videos": [url|null, ...], "errors": [...], "source": "pexels"}

    POST /api/render   {"topic": "...", "chapters": [{"narration": "..."}, ...]}
      -> {"job": "<id>", "poll": "/api/render/status?job=<id>"}

**The voice is the one free thing in this app.** Every other generated asset
here bills someone — Claude per lesson, Replicate per clip. edge-tts needs no
API key at all, so narration costs nothing and the page prefers it, keeping
`speechSynthesis` only as the fallback. That matters most in Bahasa: a machine
with no `id-ID` voice installed narrates an Indonesian lesson in an American
accent and says nothing about it. `id-ID-GadisNeural` is always there.

**The cues are why `/api/tts` returns JSON instead of just audio.** The
avatar's mouth is driven by word boundaries, which `speechSynthesis` emits and
a plain `<audio>` element does not. Without them the mouth would flap on an
average while a real voice said something else; with them the lipsync is
tighter than the browser's ever was, because the timings come from the
synthesiser rather than from a word count.

**Stock footage is the cheap sibling of `/api/videos`.** Diffusion *invents*
footage of the step and bills per clip; a stock search finds footage that was
really filmed, in seconds rather than 30-90s, for nothing. Both answer the
same `{"videos": [url|null, ...]}`, so a `world` chapter's `step_videos`
machinery does not care which one filled the screen — a miss comes back null,
that step's screen goes quiet, and the lesson carries on. Stock is preferred
when it is configured; an explicit `"step_videos": ["cinematic sentence", ...]`
is a *diffusion* instruction, so that stays on diffusion.

**The export is the thing the browser genuinely cannot do.** `Record` captures
the board with `canvas.captureStream()`, which carries pixels and no audio —
the file it makes is a mute animation of a lesson that was mostly talking.
`Export MP4` renders server-side instead: the same narration in the same
neural voice, real footage cut under it, subtitles burned in, muxed by ffmpeg.
It is minutes rather than seconds, so it returns a job id and the button
polls; the render state lives in a file, not in memory, because gunicorn runs
two workers and a poll lands on whichever one is free.

### Setup

    pip install -r mpt/requirements.txt

Narration works immediately after that — no key, no config. Stock footage and
the export additionally need a free Pexels key in `mpt/config.toml`:

    pexels_api_keys = ["your-key"]        # https://www.pexels.com/api/

ffmpeg does **not** need to be on PATH: moviepy pulls in `imageio-ffmpeg`,
which ships a binary, and `mpt/` resolves that automatically.

### It degrades in three independent pieces

The tiers fail separately and are reported separately, so a box that can
narrate but not export says exactly that instead of hiding all three:

    GET /health -> {"mpt": {"ok": true, "tts": true,
                            "stock": false, "stock_why": "...no api key...",
                            "render": true, "voices": {...}}}

With `mpt/` absent entirely, every route answers `{"disabled": true, "why":
...}` — the same contract `/api/video` uses without `REPLICATE_API_TOKEN` —
and the page falls back rather than erroring. The board still runs with no
keys of any kind, which is still the point.

Overrides: `MPT_VOICE_EN`, `MPT_VOICE_ID`.

Still open: AVI-style indexing of finished lessons for deeper question
answering.

## One 3D room: `mode:"world"`  (LineFORM's third affordance)

Panes are three separate rooms. A world chapter puts the same clips in ONE
space — placed at real positions, occluding each other, filmed by a single
camera — which is what the paper's constraint needs:

    "whole body motion can be constrained by wrapping the actuated curve
     interface around limbs or joints like bandages"       (LineFORM, p.7)

    POST /api/world
    {"actors": [{"kind": "skeleton", "steps": [...]},
                {"kind": "hand",     "steps": [...]},
                {"kind": "lineform", "shapes": [...], "wrap": true}],
     "seconds": 16, "screen": true}
    -> {"kind": "world", "actors": [...placed clips...],
        "stage": {"ground": ..., "screen": ...}, "shots": [...camera track...]}

What the room adds over `compose`:

- **binding** — an actor rides another actor's joint. A hand and a body are
  automatically the same person's hand; `"wrap": true` bandages the servo
  chain to the moving forearm. Placement is `world.py`'s job, not a rig's.
- **the footage is furniture** — a diffusion clip plays on a screen standing
  in the room (`"video_prompt"`), so the body can walk in front of it and
  the chain can hang between you and it. No token: the screen is simply not
  part of the room, everything else still runs.
- **footage cut to the steps** — `"step_videos": true` on a world chapter
  generates one clip per step and cuts it onto that screen exactly when the
  rig reaches that step, so real film of the pinch plays behind the servo
  chain doing the pinch. The cues come off the SOLVED clip's segment
  boundaries, not off the director's text, so they cannot drift out of sync.

      POST /api/videos {"prompts": ["...", "..."]}
      -> {"videos": [url|null, ...], "errors": [...]}

  Generated concurrently in one request (serialising them would cost 30-90s
  per step and starve a 2-worker deploy), capped at 4 clips per chapter, and
  a clip that fails comes back null — that step's screen goes quiet and the
  lesson carries on. You can write the prompts yourself by passing an array
  instead of `true`. **This is one paid generation per step**; without a
  token the screen simply says so and everything else still runs.
- **a camera with opinions** — shots are planned from the how-to text, not a
  fixed rhythm: `motion.classify_step`'s primitive (or the hand pose) decides
  close, medium or wide, so twisting a cap cuts to the fingers and lifting a
  wheel pulls back.
- **the student holds it too** — drag to orbit, wheel to zoom, double-click
  to reset. The drag is an offset on top of the director's shot, so the
  lesson keeps cutting and you keep your angle.
- **the tutor can answer by moving** — an interruption may come back with
  `<camera>{"target":"the grip index tip","dist":0.8}</camera>`, which is
  the honest answer to "show me that from the side". Annotation strokes bind
  to the anchor they were drawn around and follow it: circle the wrist,
  orbit ninety degrees, the circle is still on the wrist.

Try it with no key and no token at all:

    http://localhost:8000/?preset=world

## Live coaching: the rig watches you back  (`coach.py`)

Every other mode points one way — text becomes a motion, the motion plays,
you watch. This is the return path, and it is the half of Figure 9 the rest
of the codebase was missing:

    "it can also record motion and replay back on your body ... to learn
     kinesthetic motion such as sports and dances as an external motor
     memory"                                              (LineFORM, Fig. 9)

Press **Coach me** (or open `/?coach=squat%20down`). The board splits: the
rig performs the current step on a loop on the left, your camera is on the
right with the tracked skeleton drawn on top of it, and a strip underneath
scores the two against each other and tells you what to change.

    POST /api/coach          {"question": "..."} | {"steps": [...]} | {"text": "..."}
      -> {"clip": <the skeleton clip to copy>,
          "reference": {"steps": [{"targets": {...}, "key": [...]}, ...], ...}}

    POST /api/coach/score    {"reference": ..., "step": 0, "angles": {...}}
      -> {"score": 71, "cues": [{"text": "straighten your right elbow", ...}]}

    POST /api/coach/report   {"reference": ..., "samples": [...]}
      -> per-step best/mean/in-range, the fault to work on, a verdict

**Nine angles, not landmark positions.** Where you stand, how tall you are
and which way the camera points all change the landmarks and none of them
change a joint angle, so a 1.9 m adult filmed from the side is comparable
to a 1.5 m one filmed head-on with no calibration step. Eight joints plus
trunk lean — the things a person can actually be told to change.

**A step is a band, not a pose.** Each step's target is the interval each
angle travels through while the reference performs it. Scoring against the
step's mean pose would mark you down for copying the demonstration exactly:
the elbow is meant to move 60 degrees, so 60 degrees of movement cannot be
the error.

**The tracking is in your browser.** MediaPipe's pose landmarker runs in the
tab (loaded from a CDN on first use — about 15 MB, so the first start takes
a few seconds). The video is never uploaded; the only thing that reaches the
server is a few angles per second, and only when the session ends and asks
for a summary. It needs a secure context, so `https://` or `localhost`.

**Both sides compute the same numbers from the same definitions.** The joint
triples, the tolerances and the phrasing table live in `coach.py` and are
shipped to the page inside the reference; the browser scores at frame rate
(30 fps over HTTP would be absurd) and `/api/coach/score` re-runs the same
code when an authoritative number is wanted. They agree to the digit —
`float(x)+.5` flooring included, so a 16.5 degree error is not reported as
16 in the tutor's answer and 17 on the overlay beside it.

**The take becomes a chapter.** Coaching records the raw landmark streams as
it goes — BlazePose's 33 body points and MediaPipe's 21 hand points, at the
board's own 12 fps. Press Finish and they go to `/api/replay`, which hands
them to the same functions `/api/track` feeds, so what comes back is an
ordinary clip: your motion performed by the servo chains, with the gestures
recognised from *your* frames rather than from a text prompt.

    POST /api/replay   {"body": [[[x,y,z] x33], ...],
                        "hand": [[[x,y,z] x21], ...],
                        "fps": 12, "kind": "compose", "steps": [...]}
      -> the same clip contract /api/motion, /api/hand and /api/track return

Unlike `/api/track` this needs no mediapipe, no OpenCV, no yt-dlp and no
download — the tracking already happened on the other side of the wire. A
deployment where `/api/track` is unavailable can still do this, which is most
deployments.

The clip is pushed into the timeline as an ordinary chapter, which is what
makes the rest work for free: **pause, interrupt, ask, annotate and resume all
behave exactly as they do on a synthesised chapter, because it is not a
special case.** The same Pause button drives a coaching session too — asking
"am I leaning too far forward?" mid-squat pauses the session, answers against
your live joint angles, and resumes.

A pane that barely saw the take is dropped rather than allowed to shorten the
others: panes share one playhead, so a hand that drifted through shot for half
a second would otherwise truncate a fifteen-second body replay to half a
second. It comes back in `missing` instead.

Other things worth knowing:

- **Mirror is on by default.** You are facing the demonstration, so copying
  it with the opposite arm is the natural thing to do; the score accepts
  that, and the fault is drawn on the limb that is actually wrong. Turn it
  off to be graded on the same side the rig uses.
- **Partly visible is not a fail.** A measure whose joints are out of frame
  is dropped rather than guessed, so sitting at a desk gets you graded on
  your arms. Those joints go grey, not green — grey means "not measured",
  never "correct".
- **The hivemind grounds it only on a confident match.** Retrieval ranks and
  does not threshold, so an unknown topic still comes back with whatever
  scored least badly. Everywhere else that is harmless; here it would grade
  your squat against "patch a bicycle inner tube", so `/api/coach` requires
  the retrieved title to actually be about what you asked and synthesises
  the movement otherwise.
- **It says when the reference is meaningless.** A topic with no action verb
  in it produces a body standing still waving its hands. `reference.physical`
  is false and the page says so rather than issuing a number with nothing
  behind it.
- **You can interrupt mid-session.** The current step, your score and the
  specific fault go into the prompt, so "what am I doing wrong?" is answered
  about your actual right elbow rather than in general.

## Diffusion video scenes (CogVideoX-class, combined with SketchAgent)

Set one more env var and the Director starts planning "hybrid" scenes:
generated film footage playing on the board with SketchAgent annotation
strokes (arrows, circles) drawn OVER the moving video in red with a
white halo. Interrupting freezes the footage on the current frame and
the tutor annotates that exact frame.

    export REPLICATE_API_TOKEN=r8_...        # replicate.com/account
    # optional; default is the fast model (~39s per 5s clip at 480p):
    export VIDEO_MODEL=wan-video/wan-2.2-t2v-fast
    # or the actual CogVideoX paper model (slower, ~6 min per clip):
    # export VIDEO_MODEL=thudm/cogvideox-t2v

No token set -> /api/video answers {"disabled": true} and every scene
falls back to sketch-only automatically (hybrid plans are coerced to
plain sketches, and a scene whose clip fails mid-lesson is redrawn as
a full diagram rather than orphan annotations). Clip generation runs
in parallel with sketch generation, the NEXT chapter's clip is
prefetched while the current one plays (so no 30-90s stall between
chapters), and student interruptions can spawn hybrid chapters too —
"show me a real one" gets footage + annotations, not just a diagram.
Clips cost money per generation (see replicate.com pricing), so mind
the meter on public deployments.
