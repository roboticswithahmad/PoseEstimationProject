# Pose Estimation (Work in Progress)

## Status

Work in progress — pose detection is working, rep-counting logic is not yet built.

## What it does

Detects and draws 33 body landmarks on video using MediaPipe's Tasks API (`PoseLandmarker`), running in `VIDEO` mode for frame-to-frame temporal consistency.

![33 pose landmarks reference](Poses/Pose_Landmarks.jpg)

## Tools used

- Python
- OpenCV
- MediaPipe Tasks API

## How to run

```bash
source cv-env/bin/activate
pip install -r requirements.txt
python3 PoseModuleVideo.py
```

## Key decision: API version

MediaPipe 1.0 removed the legacy `mp.solutions` API used in most tutorials. Built on the new Tasks API (`PoseLandmarker`) instead of downgrading, including switching from `IMAGE` mode to `VIDEO` mode for frame-to-frame consistency.

## Key decision: model size

<!-- PENDING: Ahmad to fill in — lite/full/heavy comparison notes -->

## Known limitations

<!-- PENDING: Ahmad to fill in -->

## Next steps

Rep-counting logic (joint angle calculation) not yet built.
