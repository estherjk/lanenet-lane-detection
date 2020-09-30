# README_SUPPLEMENT

Additional info (didn't want to modify original README).

## Setup

### Installing dependencies

```bash
pip install -r requirements.txt
```

### Updating PYTHONPATH

```bash
export PYTHONPATH="/path/to/lanenet-lane-detection:$PYTHONPATH"
```

## Testing LaneNet model

### Testing a single image

```bash
python3 tools/test_lanenet.py --weights_path ./model/tusimple_lanenet/tusimple_lanenet.ckpt --image_path ./data/tusimple_test_image/0.jpg
```

### Testing a video source

```bash
python3 tools/test_lanenet_video.py --weights_path ./model/tusimple_lanenet/tusimple_lanenet.ckpt --video_src ./data/tusimple_test_video/0.mp4
```