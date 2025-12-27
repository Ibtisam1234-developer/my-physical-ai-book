# Synthetic Data and Perception

## Introduction to Synthetic Data Generation

Synthetic data generation is a revolutionary approach to creating training datasets for AI systems. Instead of collecting real-world data, which can be expensive, time-consuming, and limited, synthetic data is generated from simulations with full ground truth labels.

### Advantages of Synthetic Data

- **Infinite Variety**: Generate unlimited variations of scenes and objects
- **Ground Truth Labels**: Perfect annotations for training AI models
- **Cost Effective**: Reduce expensive data collection and labeling
- **Safety**: Train on dangerous scenarios without risk
- **Control**: Precise control over environmental conditions
- **Speed**: Generate data much faster than real collection

## Isaac Sim for Synthetic Data

Isaac Sim provides powerful tools for synthetic data generation:

### Synthetic Data Helper

```python
# Example: Using Isaac Sim's synthetic data generation
from omni.isaac.synthetic_utils import SyntheticDataHelper
from omni.isaac.synthetic_utils.annotation_configs import (
    InstanceSegmentationCfg,
    BoundingBox2DCfg,
    BoundingBox3DCfg,
    SemanticSegmentationCfg
)
import carb
import numpy as np
from PIL import Image


class SyntheticDataGenerator:
    """
    Generates synthetic training data using Isaac Sim.
    """

    def __init__(self):
        self.sd_helper = SyntheticDataHelper()
        self.setup_annotation_configs()

    def setup_annotation_configs(self):
        """Set up annotation configurations."""
        # Instance segmentation configuration
        self.instance_seg_cfg = InstanceSegmentationCfg(
            colorize_instance_id=True,
            return_colors=True
        )

        # 2D bounding box configuration
        self.bbox_2d_cfg = BoundingBox2DCfg(
            return_bboxes=True,
            return_labels=True
        )

        # 3D bounding box configuration
        self.bbox_3d_cfg = BoundingBox3DCfg(
            return_bboxes=True,
            return_labels=True
        )

        # Semantic segmentation configuration
        self.semantic_seg_cfg = SemanticSegmentationCfg(
            colorize_instance_id=True,
            return_colors=True
        )

    def capture_annotations(self, output_dir, num_samples=1000):
        """
        Capture synthetic annotations for AI training.

        Args:
            output_dir: Directory to save annotations
            num_samples: Number of samples to generate
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        for i in range(num_samples):
            # Randomize environment for variation
            self.randomize_environment()

            # Capture RGB image
            rgb_data = self.sd_helper.get_rgb_data()

            # Capture instance segmentation
            instance_seg_data = self.sd_helper.get_instance_segmentation_data(
                config=self.instance_seg_cfg
            )

            # Capture 2D bounding boxes
            bbox_2d_data = self.sd_helper.get_2d_bounding_box_data(
                config=self.bbox_2d_cfg
            )

            # Capture semantic segmentation
            semantic_seg_data = self.sd_helper.get_semantic_segmentation_data(
                config=self.semantic_seg_cfg
            )

            # Save data
            self.save_sample(
                rgb_data,
                instance_seg_data,
                bbox_2d_data,
                semantic_seg_data,
                f"{output_dir}/sample_{i:06d}"
            )

    def randomize_environment(self):
        """Randomize environment for domain randomization."""
        # Randomize object positions
        self.randomize_object_poses()

        # Randomize lighting
        self.randomize_lighting()

        # Randomize textures
        self.randomize_materials()

        # Randomize camera positions
        self.randomize_camera_poses()

    def save_sample(self, rgb, instance_seg, bbox_2d, semantic_seg, path):
        """Save a complete synthetic data sample."""
        import os
        os.makedirs(path, exist_ok=True)

        # Save RGB image
        rgb_img = Image.fromarray(rgb)
        rgb_img.save(f"{path}/rgb.png")

        # Save instance segmentation
        instance_seg_img = Image.fromarray(instance_seg["data"])
        instance_seg_img.save(f"{path}/instance_segmentation.png")

        # Save semantic segmentation
        semantic_seg_img = Image.fromarray(semantic_seg["data"])
        semantic_seg_img.save(f"{path}/semantic_segmentation.png")

        # Save bounding box annotations
        import json
        bbox_annotations = {
            "bbox_2d": bbox_2d["data"].tolist(),
            "labels": bbox_2d["labels"]
        }
        with open(f"{path}/bbox_annotations.json", "w") as f:
            json.dump(bbox_annotations, f, indent=2)

        # Save metadata
        metadata = {
            "rgb_resolution": rgb.shape[:2],
            "instance_seg_resolution": instance_seg["data"].shape[:2],
            "semantic_seg_resolution": semantic_seg["data"].shape[:2],
            "num_objects": len(bbox_2d["data"])
        }
        with open(f"{path}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
```

## Domain Randomization for Robust Perception

### Physics-Based Domain Randomization

```python
# Example: Domain randomization for robust perception
import random
import numpy as np


class DomainRandomizer:
    """
    Implements domain randomization techniques to improve sim-to-real transfer.
    """

    def __init__(self):
        self.randomization_params = {
            'materials': {
                'roughness_range': (0.1, 0.9),
                'metallic_range': (0.0, 0.2),
                'albedo_range': (0.2, 1.0),
                'normal_map_strength_range': (0.0, 1.0)
            },
            'lighting': {
                'intensity_range': (0.3, 2.0),
                'color_temperature_range': (3000, 8000),
                'shadow_softness_range': (0.1, 1.0)
            },
            'camera': {
                'exposure_range': (-2.0, 2.0),
                'iso_range': (100, 1600),
                'aperture_range': (1.4, 16.0),
                'focus_distance_range': (0.5, 10.0)
            },
            'objects': {
                'scale_variance': 0.1,
                'rotation_variance': 0.1,
                'position_variance': 0.05
            }
        }

    def randomize_material_properties(self, material_prim):
        """Randomize material properties."""
        # Randomize roughness
        roughness = random.uniform(
            self.randomization_params['materials']['roughness_range'][0],
            self.randomization_params['materials']['roughness_range'][1]
        )
        material_prim.GetRoughnessAttr().Set(roughness)

        # Randomize metallic
        metallic = random.uniform(
            self.randomization_params['materials']['metallic_range'][0],
            self.randomization_params['materials']['metallic_range'][1]
        )
        material_prim.GetMetallicAttr().Set(metallic)

        # Randomize albedo
        albedo = [
            random.uniform(
                self.randomization_params['materials']['albedo_range'][0],
                self.randomization_params['materials']['albedo_range'][1]
            ) for _ in range(3)
        ]
        material_prim.GetDiffuseAttr().Set(albedo)

    def randomize_lighting(self, light_prim):
        """Randomize lighting properties."""
        # Randomize intensity
        intensity = random.uniform(
            self.randomization_params['lighting']['intensity_range'][0],
            self.randomization_params['lighting']['intensity_range'][1]
        )
        light_prim.GetIntensityAttr().Set(intensity)

        # Randomize color temperature (convert to RGB)
        color_temp = random.uniform(
            self.randomization_params['lighting']['color_temperature_range'][0],
            self.randomization_params['lighting']['color_temperature_range'][1]
        )
        color_rgb = self.color_temperature_to_rgb(color_temp)
        light_prim.GetColorAttr().Set(color_rgb)

    def color_temperature_to_rgb(self, temp_kelvin):
        """
        Convert color temperature in Kelvin to RGB.

        Args:
            temp_kelvin: Color temperature in Kelvin

        Returns:
            RGB color tuple (0-1 range)
        """
        temp_kelvin /= 100
        if temp_kelvin <= 66:
            red = 1.0
            green = 0.3900816764212907 * np.log(temp_kelvin) - 0.631841443162656
        else:
            red = 1.593017578125 * pow(temp_kelvin - 60, -0.1332047592)
            green = 0.11298908650786 * np.log(temp_kelvin - 10) - 0.2021413942

        if temp_kelvin >= 66:
            blue = 1.0
        elif temp_kelvin <= 19:
            blue = 0.0
        else:
            blue = 0.2370407477006835 * np.log(temp_kelvin - 10) - 0.2335206936025876

        return (np.clip(red, 0, 1), np.clip(green, 0, 1), np.clip(blue, 0, 1))

    def randomize_camera_properties(self, camera_prim):
        """Randomize camera properties."""
        # Randomize exposure
        exposure = random.uniform(
            self.randomization_params['camera']['exposure_range'][0],
            self.randomization_params['camera']['exposure_range'][1]
        )
        camera_prim.GetExposureAttr().Set(exposure)

        # Randomize ISO
        iso = random.uniform(
            self.randomization_params['camera']['iso_range'][0],
            self.randomization_params['camera']['iso_range'][1]
        )
        camera_prim.GetISOAttr().Set(iso)

        # Randomize aperture
        aperture = random.uniform(
            self.randomization_params['camera']['aperture_range'][0],
            self.randomization_params['camera']['aperture_range'][1]
        )
        camera_prim.GetFStopAttr().Set(aperture)
```

## Perception Algorithms with Isaac ROS

### Isaac ROS Perception Pipeline

```python
# Example: Isaac ROS perception pipeline
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose
from std_msgs.msg import Header
import cv2
import numpy as np


class IsaacPerceptionPipeline(Node):
    """
    Perception pipeline using Isaac ROS packages.
    """

    def __init__(self):
        super().__init__('isaac_perception_pipeline')

        # Image subscription
        self.image_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_rect_color',
            self.image_callback,
            10
        )

        # Camera info subscription
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/rgb/camera_info',
            self.camera_info_callback,
            10
        )

        # Object detection publisher
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            '/perception/detections',
            10
        )

        # Point cloud publisher (for 3D perception)
        self.point_cloud_pub = self.create_publisher(
            PointStamped,
            '/perception/point_cloud',
            10
        )

        # Initialize camera matrix
        self.camera_matrix = None
        self.distortion_coeffs = None

        self.get_logger().info('Isaac Perception Pipeline initialized')

    def image_callback(self, msg):
        """Process incoming camera images."""
        # Convert ROS Image to OpenCV
        image = self.ros_image_to_cv2(msg)

        # Perform object detection
        detections = self.detect_objects(image)

        # Publish detections
        self.publish_detections(detections, msg.header)

        # Perform 3D reconstruction if depth is available
        if self.has_depth_available():
            self.perform_3d_reconstruction(image, msg.header)

    def detect_objects(self, image):
        """Detect objects in the image using Isaac ROS packages."""
        # This would typically use Isaac ROS detection packages
        # such as Isaac ROS DNN Image Encoding or Isaac ROS AprilTag
        # For demonstration, we'll use a simple OpenCV-based approach

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply simple blob detection
        detector = cv2.SimpleBlobDetector_create()
        keypoints = detector.detect(gray)

        # Convert to vision_msgs format
        detections = []
        for kp in keypoints:
            detection = Detection2D()
            detection.bbox.center.x = kp.pt[0]
            detection.bbox.center.y = kp.pt[1]
            detection.bbox.size_x = kp.size
            detection.bbox.size_y = kp.size

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.id = "object"
            hypothesis.score = 0.8  # Confidence score

            detection.results.append(hypothesis)
            detections.append(detection)

        return detections

    def perform_3d_reconstruction(self, rgb_image, header):
        """Perform 3D reconstruction using depth information."""
        # This would typically use Isaac ROS Stereo DNN or similar
        # For demonstration, we'll create a simple point cloud

        # Simulate depth data (in practice, this comes from depth sensor)
        height, width = rgb_image.shape[:2]
        depth_map = np.ones((height, width), dtype=np.float32) * 2.0  # 2 meters away

        # Create point cloud from depth map
        points = []
        for v in range(height):
            for u in range(width):
                if depth_map[v, u] > 0:  # Valid depth
                    # Convert pixel coordinates to 3D world coordinates
                    z = depth_map[v, u]
                    x = (u - self.camera_matrix[0, 2]) * z / self.camera_matrix[0, 0]
                    y = (v - self.camera_matrix[1, 2]) * z / self.camera_matrix[1, 1]

                    point_msg = PointStamped()
                    point_msg.header = header
                    point_msg.point.x = x
                    point_msg.point.y = y
                    point_msg.point.z = z

                    self.point_cloud_pub.publish(point_msg)

    def camera_info_callback(self, msg):
        """Update camera parameters from camera info."""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.d)

    def ros_image_to_cv2(self, img_msg):
        """Convert ROS Image message to OpenCV image."""
        # Convert ROS image format to OpenCV format
        if img_msg.encoding == 'rgb8':
            return np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
                img_msg.height, img_msg.width, 3
            )
        elif img_msg.encoding == 'bgr8':
            return np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
                img_msg.height, img_msg.width, 3
            )
        else:
            # Handle other encodings as needed
            raise ValueError(f"Unsupported image encoding: {img_msg.encoding}")

    def publish_detections(self, detections, header):
        """Publish object detections."""
        detection_array = Detection2DArray()
        detection_array.header = header
        detection_array.detections = detections

        self.detection_pub.publish(detection_array)

    def has_depth_available(self):
        """Check if depth information is available."""
        # In a real system, this would check for depth sensor data
        return self.camera_matrix is not None
```

## Deep Learning Integration

### Training Neural Networks with Synthetic Data

```python
# Example: Training neural network with synthetic data
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import json


class SyntheticDataset(Dataset):
    """
    Dataset class for loading synthetic training data.
    """

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []

        # Load all sample directories
        for sample_dir in os.listdir(data_dir):
            sample_path = os.path.join(data_dir, sample_dir)
            if os.path.isdir(sample_path):
                rgb_path = os.path.join(sample_path, 'rgb.png')
                seg_path = os.path.join(sample_path, 'instance_segmentation.png')

                if os.path.exists(rgb_path) and os.path.exists(seg_path):
                    self.samples.append({
                        'rgb': rgb_path,
                        'segmentation': seg_path,
                        'metadata': os.path.join(sample_path, 'metadata.json')
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load RGB image
        rgb_image = Image.open(sample['rgb']).convert('RGB')

        # Load segmentation mask
        seg_mask = Image.open(sample['segmentation']).convert('L')  # Grayscale

        # Apply transforms
        if self.transform:
            rgb_image = self.transform(rgb_image)
            seg_mask = self.transform(seg_mask)

        # Load metadata if needed
        if os.path.exists(sample['metadata']):
            with open(sample['metadata'], 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}

        return {
            'rgb': rgb_image,
            'segmentation': seg_mask,
            'metadata': metadata
        }


class PerceptionNet(nn.Module):
    """
    Simple convolutional neural network for perception tasks.
    """

    def __init__(self, num_classes=10):
        super(PerceptionNet, self).__init__()

        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_perception_model(data_dir, num_epochs=10, batch_size=32, learning_rate=0.001):
    """
    Train perception model using synthetic data.

    Args:
        data_dir: Directory containing synthetic training data
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
    """
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Create dataset and dataloader
    dataset = SyntheticDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create model, loss, and optimizer
    model = PerceptionNet(num_classes=20)  # Example: 20 object classes
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            # Get data
            rgb_images = batch['rgb']
            labels = batch['segmentation']  # This would be class labels in practice

            # Forward pass
            outputs = model(rgb_images)
            loss = criterion(outputs, labels.squeeze(1))  # Remove channel dimension

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Batch [{batch_idx}/{len(dataloader)}], '
                      f'Loss: {loss.item():.4f}')

        epoch_loss = running_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}] Average Loss: {epoch_loss:.4f}')

    # Save trained model
    torch.save(model.state_dict(), 'trained_perception_model.pth')
    print("Model saved as 'trained_perception_model.pth'")
```

## Isaac ROS Perception Packages

### Isaac ROS AprilTag Detection

```python
# Example: Isaac ROS AprilTag detection
from apriltag_msgs.msg import AprilTagDetectionArray
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
import rclpy
from rclpy.node import Node


class AprilTagDetector(Node):
    """
    AprilTag detection using Isaac ROS packages.
    """

    def __init__(self):
        super().__init__('apriltag_detector')

        # Subscribe to camera image
        self.image_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_rect_color',
            self.image_callback,
            10
        )

        # Publish detected tags
        self.tag_pub = self.create_publisher(
            AprilTagDetectionArray,
            '/perception/apriltags',
            10
        )

        # Publish tag poses
        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/perception/apriltag_pose',
            10
        )

        self.get_logger().info('AprilTag Detector initialized')

    def image_callback(self, msg):
        """Process image and detect AprilTags."""
        # Isaac ROS handles the detection internally
        # This is a simplified interface
        pass
```

### Isaac ROS Stereo DNN

```python
# Example: Isaac ROS stereo depth estimation
from stereo_msgs.msg import DisparityImage
from sensor_msgs.msg import Image
import rclpy
from rclpy.node import Node


class StereoDepthEstimator(Node):
    """
    Stereo depth estimation using Isaac ROS packages.
    """

    def __init__(self):
        super().__init__('stereo_depth_estimator')

        # Subscribe to stereo pair
        self.left_sub = self.create_subscription(
            Image,
            '/camera/left/image_rect_color',
            self.left_image_callback,
            10
        )

        self.right_sub = self.create_subscription(
            Image,
            '/camera/right/image_rect_color',
            self.right_image_callback,
            10
        )

        # Publish disparity map
        self.disparity_pub = self.create_publisher(
            DisparityImage,
            '/perception/disparity',
            10
        )

        # Publish depth image
        self.depth_pub = self.create_publisher(
            Image,
            '/perception/depth',
            10
        )

        self.latest_left = None
        self.latest_right = None

    def left_image_callback(self, msg):
        """Store left camera image."""
        self.latest_left = msg
        if self.latest_right is not None:
            self.process_stereo_pair()

    def right_image_callback(self, msg):
        """Store right camera image."""
        self.latest_right = msg
        if self.latest_left is not None:
            self.process_stereo_pair()

    def process_stereo_pair(self):
        """Process stereo image pair to generate depth."""
        # Isaac ROS stereo packages handle the heavy computation
        # This is a simplified interface
        pass
```

## Exercise: Create Synthetic Data Pipeline

Create a complete synthetic data generation pipeline that:
1. Sets up a varied scene with multiple objects
2. Implements domain randomization for materials and lighting
3. Captures RGB, depth, and segmentation data
4. Saves annotations in a format suitable for training
5. Tests the generated data with a simple perception model

## Learning Outcomes

After completing this section, students will be able to:
1. Generate synthetic training data using Isaac Sim
2. Implement domain randomization techniques
3. Use Isaac ROS perception packages
4. Train neural networks with synthetic data
5. Evaluate perception model performance
6. Apply sim-to-real transfer techniques

## Next Steps

Continue to [Navigation Planning](./navigation-planning.md) to learn about Isaac ROS integration and Nav2 for humanoid navigation.