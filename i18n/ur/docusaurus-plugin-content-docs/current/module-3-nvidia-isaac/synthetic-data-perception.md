---
slug: /module-3-nvidia-isaac/synthetic-data-perception
title: "Synthetic Data for Perception"
hide_table_of_contents: false
---

# Synthetic Data for Perception

Isaac Sim synthetic data generation provide کرتا ہے جو ML training کے لیے useful ہے۔

## Synthetic Data Generation

### Randomization

```python
from isaac_sim import SimulationApp

app = SimulationApp()
app.randomize_objects()

# Randomize positions
app.randomize_positions(min=[-1, -1, 0], max=[1, 1, 1])
```

### Dataset Generation

```python
from omni.isaac.synthetic_utils import SyntheticDataset

dataset = SyntheticDataset(
    num_images=10000,
    output_dir="./dataset",
    categories=["person", "robot", "obstacle"]
)

for i in range(10000):
    image, annotations = dataset.next()
    dataset.save(i, image, annotations)
```

## Perception Pipeline

### Object Detection

```python
from isaac_ros_object_detection import ObjectDetector

detector = ObjectDetector(
    model_path="resnet50.onnx",
    confidence_threshold=0.5
)

def detect_objects(image):
    detections = detector.infer(image)
    return detections
```

### Semantic Segmentation

```python
from isaac_ros_segmentation import SemanticSegmenter

segmenter = SemanticSegmenter(
    model_path="deeplabv3.onnx"
)

def segment_image(image):
    segmentation = segmenter.infer(image)
    return segmentation
```

## اگلے steps

[navigation-planning.md](./navigation-planning.md) پڑھیں۔
