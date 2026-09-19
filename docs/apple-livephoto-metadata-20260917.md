<!-- v1.0.0 -->
# Apple `LivePhotoInfo` and the per-frame Core Motion track — Wed, 17 Sep 2026

Notes toward decoding the metadata iPhone captures carry alongside the video.
Nothing here is wired into the pipeline yet. It is written down because one
field in it — a per-frame optical-stabilisation offset — would explain a defect
we have measured and worked around twice.

## Why this matters to the reconstruction

`camera_calibration.py` records that DA3 reports a **different focal length for
every frame** of a video shot on a lens that never moved: 881.7–915.1 px in
July, 875.3–913.8 px in September, a 4.3% spread. We collapse that to a median
and write one camera, which is the right fix for a value that should not vary.

But the spread may not be pure estimator noise. If the phone's optical image
stabiliser physically shifts the lens per frame, the principal point genuinely
moves per frame, and a per-frame estimator would correctly report a per-frame
camera. That offset is plausibly recorded in the file. If it is, we could
subtract it rather than average over it.

## What is actually in the file

Measured on `thorough-backyard-155sec.MOV` (1920×1080, 59.94 fps, 9332 frames).

| track | samples | contents |
|---|---|---|
| video / audio | 9332 / 7298 | — |
| mebx #1 (`Track3`) | 1 | `VideoOrientation: Rotate 180` |
| mebx #2 (`Track4`) | 1 | static |
| **mebx #3 (`Track5`)** | **9332** | **one record per video frame**, `CoreMotionVersion 3077.0.4` |
| mebx #4 (`Track6`) | 1 | static |

**GPS is a single static point**, not a track: one `GPSCoordinates` in `Keys`
with `LocationAccuracyHorizontal 19.79` m. There is no positional trajectory to
recover from these files, and the three clips of 17 Sep differ from each other
by about 8 m of longitude, which is inside that error bar.

**The per-frame track is fixed 144-byte records**, 9332 of them, exactly one per
video frame. What ExifTool prints as `LivePhotoInfo` is record 0 of this track:
its float values match byte-for-byte.

## Field anatomy

The structure below was contributed by the user and is reproduced as the working
hypothesis; the measurements against it are mine.

**1. Temporal synchronisation.** Pairs of `int64` / `CMTime` structs:
`StillImageTimeValue` / `StillImageTimeScale` give the rational timestamp where
the full-resolution still aligns with the video track, plus trim-in/trim-out
offsets bounding the loop segment.

*Consistent with:* slots 28/29 and 30/31 hold values that read as denormal
floats around 1.21e-40, the signature of a 64-bit integer split across two
32-bit slots. Slots 29 and 31 take only 37 distinct values across 9332 frames,
which is what a slowly-incrementing high word looks like.

**2. Motion vitality.** `vitality-score` (is the motion intentional and
coherent), `vitality-scoring-version` (integer, which `CMCaptureCore` heuristic),
`vitality-transition-score` (jitter and acceleration around the shutter moment;
a jerk yields a low score and suppresses the UI bounce).

*Consistent with:* several constant integer slots (2 → `3`, 18 → `7`, 12 →
`16711684`) that look like version and flag words rather than measurements.

**3. Spatial alignment transform.** Six to nine floats forming an affine or
homography that maps the video frame's coordinate space onto the still frame,
because the still readout and the video stream differ in crop, rolling-shutter
profile and **OIS offset**:

```
| a   b   0 |
| c   d   0 |
| tx  ty  1 |
```

*This is the part worth chasing.* Record 0 carries exactly six floats in a run:

```
7.39610958334089e-15   -1.78372772010726e-15
0.0158999189734459     -0.110848918557167
1.0253164768219         1.00390625
```

Two near-zero, two small, two near one — the shape of a transform that is
almost identity. And `1.00390625` is **exactly 257/256**, a dyadic rational,
which is a fixed hardware ratio rather than anything measured. `1.0253164768219`
is **constant across all 9332 frames** (slot 10), consistent with a fixed
still-versus-video crop ratio.

## What varies per frame, and how cleanly

Lag-1 autocorrelation of each varying slot, over 9332 frames. A physical signal
is smooth; noise is not.

| slot | range | autocorr | reading |
|---|---|---|---|
| 8 | −16.5 … 30.0 | 0.990 | plausible rotation rate, rad/s |
| 9 | −5.41 … 4.69 | 0.916 | plausible rotation rate |
| 19, 20, 22, 23 | within ±0.9 | 1.0000 | attitude-like, but see below |
| **25, 26** | **±0.00068** | **1.0000** | **best OIS-offset candidates** |
| 27 | −1.24 … 0.86 | 1.0000 | — |
| 6, 7, 21, 24 | hundreds to thousands | 1.0000 | pixel or timestamp scale |

All of these are smooth to four decimal places, so the track carries real
physical signal rather than padding.

**What it is not.** No contiguous triple or quadruple among the unit-scale slots
has norm ≈ 1, so slots 19/20/22/23 are not a plain gravity vector or quaternion
in that layout. Either the components are non-adjacent, or they are scaled, or
they are transform terms rather than attitude.

## The two things worth doing next, in order

1. **Test slots 25 and 26 as an OIS offset.** They are tiny, perfectly smooth,
   and in the right numeric range for a normalised principal-point shift. The
   test is discriminative and free: DA3 already reports a per-frame `cx`/`cy`
   for every frame of a capture. If these slots correlate with the residual
   after the median is removed, the OIS reading is confirmed and we can subtract
   a measured offset instead of averaging over an unexplained spread.
2. **Recover per-frame attitude.** If slots 8 and 9 are rotation rates, they
   integrate to relative orientation, which would constrain the scale drift
   measured at 1.7–2.4× by `tools/measure_scale_drift.py`, and would replace
   `gravity_align`'s PCA-skewness heuristic — the one that "occasionally guesses
   up-down backwards".

Neither needs a GPU, a job, or a Space rebuild. Both need the layout pinned
down further than this document pins it.
