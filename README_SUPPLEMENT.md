# README_SUPPLEMENT

Additional info (didn't want to modify original README).

## Docker Usage

### Build

```bash
docker build -t lanenet:latest .
```

### Run

```bash
docker run -it --rm \
    --gpus all \
    -u $(id -u):$(id -g) \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -v $PWD:/code \
    lanenet:latest bash
```

## Testing LaneNet model

### Testing a single image

```bash
python tools/test_lanenet.py --weights_path ./model/tusimple_lanenet/tusimple_lanenet.ckpt --image_path ./data/tusimple_test_image/0.jpg
```

### Testing a video source

```bash
python tools/test_lanenet_video.py --weights_path ./model/tusimple_lanenet/tusimple_lanenet.ckpt --video_src ./data/tusimple_test_video/0.mp4
```