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
    -p 6006:6006 \
    -u $(id -u):$(id -g) \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -v $PWD:/code \
    lanenet:latest
```

## Testing Pre-trained TensorFlow LaneNet model

### Testing a single image

```bash
python tools/test_lanenet.py --weights_path ./model/tusimple_lanenet/tusimple_lanenet.ckpt --image_path ./data/tusimple_test_image/0.jpg
```

### Testing a video source

```bash
python tools/test_lanenet_video.py --weights_path ./model/tusimple_lanenet/tusimple_lanenet.ckpt --video_src ./data/tusimple_test_video/0.mp4
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
python tensorrt/trt_inference.py \
    --onnx_file ./model/lanenet.onnx \
    --video_src ./data/tusimple_test_video/0.mp4 \
    --engine_file ./tensorrt/pc.engine
```
