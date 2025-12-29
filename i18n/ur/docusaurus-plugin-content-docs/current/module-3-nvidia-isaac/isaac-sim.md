---
slug: /module-3-nvidia-isaac/isaac-sim
title: "Isaac Sim"
hide_table_of_contents: false
---

# Isaac Sim

Isaac Sim NVIDIA Omniverse پر based high-fidelity robotics simulation platform ہے۔

## Getting Started

### Installation

```bash
# Isaac Sim requirements
conda create -n isaac-sim python=3.8
conda activate isaac-sim

# Install Isaac Sim
pip install isaac-sim
```

### Basic Scene Setup

```python
import omni.isaac.core
from omni.isaac.core import World

# Create world
world = World()

# Add ground plane
world.add_ground_plane()

# Add light
world.add_light(light_type=LightType.DIRECTIONAL)
```

## Robot Setup

### URDF Import

```python
from omni.isaac.urdf import _urdf

# Import robot from URDF
robot = _urdf.create_robot_from_urdf(
    urdf_path="humanoid.urdf",
    import_paths=["/path/to/meshes"]
)
```

### Joint Control

```python
from omni.isaac.core.articulations import ArticulationView

# Create articulation view
robot_view = ArticulationView(robot_path="/World/humanoid")

# Set joint positions
robot_view.set_joint_positions([0.0, 0.0, 0.5, 0.0])
```

## Simulation Features

### Physics

```python
from pxr import PhysxSchema

# Add physics
world.add_physics_callback("physics", callback_fn)
```

### Sensors

```python
from omni.isaac.sensor import Camera

# Add camera
camera = Camera(
    prim_path="/World/humanoid/head/camera",
    resolution=(640, 480)
)
```

## اگلے steps

[isaac-ros-integration.md](./isaac-ros-integration.md) پڑھیں۔
