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

## Using TensorRT for inference

### Converting TensorFlow model to TensorRT

Freeze meta & checkpoint files:

```bash
python tensorrt/freeze_graph.py --weights_path model/tusimple_lanenet/tusimple_lanenet.ckpt --save_path model/lanenet.pb
```

### Converting frozen graph to ONNX

```bash
python -m tf2onnx.convert \
    --input ./model/lanenet.pb \
    --output ./model/lanenet.onnx \
    --inputs lanenet/input_tensor:0 \
    --outputs lanenet/final_binary_output:0,lanenet/final_pixel_embedding_output:0
```

### Running inference with a video source

```bash
python3 tensorrt/trt_inference.py \
    --onnx_file ./model/lanenet.onnx \
    --video_src ./data/tusimple_test_video/0.mp4 \
    --engine_file ./tensorrt/jetson.engine
```