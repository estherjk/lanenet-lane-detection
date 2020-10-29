# jetson

To run LaneNet on Jetson (e.g. Xavier NX, Nano), a custom Docker container is needed.

## Docker Usage

### Build

```bash
docker build -t lanenet-jetson:latest .
```

### Run

Allow external applications to connect to the host’s X display:

```bash
xhost +
```

Start an interactive session in the container:

```bash
docker run -it --rm \
    --runtime nvidia \
    --network host \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -v $PWD/..:/code \
    lanenet-jetson:latest
```

## Inference with TensorRT

Using the sample test video:

```bash
python3 tensorrt/trt_inference.py \
    --onnx_file ./model/lanenet.onnx \
    --video_src ./data/tusimple_test_video/0.mp4 \
    --engine_file ./tensorrt/jetson.engine
```