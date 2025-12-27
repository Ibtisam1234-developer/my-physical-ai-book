# Isaac Sim

## Introduction to Isaac Sim

Isaac Sim is NVIDIA's photorealistic simulation application built on the Omniverse platform. It provides a comprehensive environment for simulating robots in realistic 3D worlds, with accurate physics, high-fidelity rendering, and GPU acceleration.

### Key Features of Isaac Sim

- **Photorealistic Rendering**: NVIDIA RTX technology for lifelike visuals
- **GPU-Accelerated Physics**: PhysX for realistic physics simulation
- **USD-based Scenes**: Universal Scene Description for scalable content
- **Synthetic Data Generation**: Create massive datasets for AI training
- **ROS 2 Integration**: Native support for ROS 2 workflows
- **AI Training Environment**: Built-in tools for reinforcement learning

## Installing and Setting Up Isaac Sim

### System Requirements

```bash
# Minimum requirements
- NVIDIA GPU with CUDA support (RTX series recommended)
- 16GB+ RAM
- Ubuntu 20.04 LTS or Windows 10/11
- 50GB+ free disk space

# Recommended for humanoid simulation
- RTX 4090 or A6000 (24GB VRAM)
- 64GB+ RAM
- 1TB+ SSD storage
```

### Installation Process

```bash
# Download Isaac Sim from NVIDIA Developer website
# Or install via Omniverse Launcher

# Set up environment
export ISAACSIM_PATH="/path/to/isaac-sim"
export PYTHONPATH="$ISAACSIM_PATH/python:$PYTHONPATH"

# Verify installation
python -c "import omni; print('Isaac Sim imported successfully')"
```

## Creating Humanoid Robot Scenes

### USD Scene Structure

```python
# Example: Creating a humanoid robot scene in USD
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf
import carb


def create_humanoid_scene(scene_path, robot_usd_path):
    """
    Create a humanoid robot scene in Isaac Sim.

    Args:
        scene_path: Path to save the scene USD file
        robot_usd_path: Path to humanoid robot USD file
    """
    # Create stage
    stage = Usd.Stage.CreateNew(scene_path)

    # Create world prim
    world = UsdGeom.Xform.Define(stage, "/World")

    # Add ground plane
    ground = UsdGeom.Mesh.Define(stage, "/World/ground")
    ground.CreatePointsAttr([(-10, 0, -10), (10, 0, -10), (10, 0, 10), (-10, 0, 10)])
    ground.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
    ground.CreateFaceVertexCountsAttr([3, 3])

    # Add lighting
    dome_light = UsdLux.DomeLight.Define(stage, "/World/domeLight")
    dome_light.CreateIntensityAttr(1.0)
    dome_light.CreateColorAttr((0.8, 0.8, 0.9))

    # Add humanoid robot from external USD
    robot_prim = stage.OverridePrim("/World/HumanoidRobot")
    robot_prim.GetReferences().AddReference(robot_usd_path)

    stage.Save()
    print(f"Scene saved to: {scene_path}")
    return stage


def setup_physics_properties(stage, robot_path):
    """
    Set up physics properties for the humanoid robot.
    """
    robot_prim = stage.GetPrimAtPath(robot_path)

    # Set up rigid body properties
    for child in robot_prim.GetAllChildren():
        if child.GetTypeName() == "Xform":
            # Add rigid body to links
            rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(child)
            rigid_body_api.CreateRigidBodyEnabledAttr(True)

            # Add collision API
            collision_api = UsdPhysics.CollisionAPI.Apply(child)
            collision_api.CreateCollisionEnabledAttr(True)


def add_sensors_to_robot(stage, robot_path):
    """
    Add sensors to the humanoid robot.
    """
    # Add RGB camera
    camera_prim = UsdGeom.Camera.Define(stage, f"{robot_path}/head/camera")
    camera_prim.GetFocalLengthAttr().Set(24.0)
    camera_prim.GetHorizontalApertureAttr().Set(36.0)
    camera_prim.GetVerticalApertureAttr().Set(24.0)

    # Add IMU sensor
    # (Implementation depends on Isaac Sim extensions)
```

## Isaac Sim Extensions for Robotics

### Core Extensions

Isaac Sim uses extensions to provide robotics-specific functionality:

```python
# Example: Using Isaac Sim extensions for robotics
import omni
import omni.kit.commands
from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage


class IsaacSimRobotManager:
    """
    Manages humanoid robots in Isaac Sim.
    """

    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.robots = {}

    def add_humanoid_robot(self, robot_path, position, orientation):
        """
        Add a humanoid robot to the simulation.

        Args:
            robot_path: Path to robot USD file
            position: (x, y, z) position
            orientation: (qx, qy, qz, qw) quaternion
        """
        # Add robot to stage
        add_reference_to_stage(
            usd_path=robot_path,
            prim_path=f"/World/Humanoid_{len(self.robots)}"
        )

        # Create robot object
        robot = Robot(
            prim_path=f"/World/Humanoid_{len(self.robots)}",
            name=f"humanoid_{len(self.robots)}",
            position=position,
            orientation=orientation
        )

        self.robots[f"humanoid_{len(self.robots)}"] = robot
        return robot

    def setup_sensors(self, robot_name):
        """Set up sensors for the robot."""
        robot = self.robots[robot_name]

        # Add camera sensor
        from omni.isaac.sensor import Camera

        camera = Camera(
            prim_path=f"{robot.prim_path}/head/camera",
            frequency=30,
            resolution=(640, 480)
        )

        # Add IMU sensor
        # Additional sensor setup...

    def run_simulation(self, steps):
        """Run the simulation for specified steps."""
        self.world.reset()

        for i in range(steps):
            self.world.step(render=True)

            # Get sensor data
            for robot_name, robot in self.robots.items():
                # Process robot data
                joint_positions = robot.get_joints_state().positions
                # Additional processing...
```

## Synthetic Data Generation

### Generating Training Data

```python
# Example: Synthetic data generation pipeline
import numpy as np
from PIL import Image
import json
from omni.isaac.synthetic_utils import SyntheticDataHelper


class SyntheticDataGenerator:
    """
    Generates synthetic training data for AI models.
    """

    def __init__(self, sim_world):
        self.world = sim_world
        self.data_helper = SyntheticDataHelper()

    def capture_training_data(self, num_samples, output_dir):
        """
        Capture synthetic training data with variations.

        Args:
            num_samples: Number of samples to generate
            output_dir: Directory to save data
        """
        for i in range(num_samples):
            # Randomize environment
            self.randomize_scene()

            # Randomize lighting
            self.randomize_lighting()

            # Randomize textures
            self.randomize_textures()

            # Randomize robot appearance
            self.randomize_robot_appearance()

            # Capture RGB image
            rgb_image = self.capture_rgb_image()

            # Capture depth image
            depth_image = self.capture_depth_image()

            # Generate segmentation mask
            seg_mask = self.generate_segmentation_mask()

            # Save data
            self.save_sample(
                rgb_image,
                depth_image,
                seg_mask,
                f"{output_dir}/sample_{i:06d}"
            )

    def randomize_scene(self):
        """Randomize scene elements."""
        # Randomize object positions
        # Randomize floor materials
        # Randomize obstacles
        pass

    def randomize_lighting(self):
        """Randomize lighting conditions."""
        # Randomize dome light intensity and color
        # Add random point lights
        # Randomize shadows
        pass

    def save_sample(self, rgb, depth, seg, path):
        """Save a synthetic data sample."""
        import os
        os.makedirs(path, exist_ok=True)

        # Save RGB image
        rgb.save(f"{path}/rgb.png")

        # Save depth image
        depth.save(f"{path}/depth.png")

        # Save segmentation mask
        seg.save(f"{path}/segmentation.png")

        # Save metadata
        metadata = {
            "rgb_path": "rgb.png",
            "depth_path": "depth.png",
            "seg_path": "segmentation.png",
            "capture_settings": self.get_capture_settings()
        }

        with open(f"{path}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
```

## Domain Randomization Techniques

### Improving Sim-to-Real Transfer

```python
class DomainRandomizer:
    """
    Implements domain randomization techniques.
    """

    def __init__(self, sim_world):
        self.world = sim_world
        self.randomization_params = {
            'textures': {
                'roughness_range': (0.1, 0.9),
                'metallic_range': (0.0, 0.2),
                'albedo_range': (0.2, 1.0)
            },
            'lighting': {
                'intensity_range': (0.5, 2.0),
                'color_temperature_range': (4000, 8000),
                'direction_variance': 0.2
            },
            'physics': {
                'friction_range': (0.3, 0.8),
                'restitution_range': (0.0, 0.2),
                'mass_variance': 0.1
            },
            'sensors': {
                'noise_level_range': (0.0, 0.05),
                'bias_range': (-0.01, 0.01)
            }
        }

    def randomize_environment(self, step):
        """Randomize environment parameters."""
        # Randomize textures
        self.randomize_materials()

        # Randomize lighting
        self.randomize_lighting_conditions()

        # Randomize physics properties
        self.randomize_physics_params()

        # Randomize sensor parameters
        self.randomize_sensor_params()

    def randomize_materials(self):
        """Randomize material properties."""
        # Iterate through all materials in scene
        # Apply random roughness, metallic, albedo values
        pass

    def randomize_physics_params(self):
        """Randomize physics parameters."""
        # Apply random friction coefficients
        # Apply random restitution values
        # Apply random mass variations
        pass

    def randomize_sensor_params(self):
        """Randomize sensor parameters."""
        # Apply random noise levels
        # Apply random biases
        # Apply random calibration parameters
        pass
```

## Isaac Sim Python API

### Controlling Simulation

```python
# Example: Isaac Sim Python API usage
import omni
from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.viewports import set_camera_view
import numpy as np


def setup_humanoid_simulation():
    """
    Set up a humanoid robot simulation in Isaac Sim.
    """
    # Create world
    world = World(stage_units_in_meters=1.0)

    # Get assets root path
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        carb.log_error("Could not find Isaac Sim assets folder")
        return

    # Add humanoid robot
    robot_asset_path = f"{assets_root_path}/Isaac/Robots/Humanoid/humanoid_instanceable.usd"

    add_reference_to_stage(
        usd_path=robot_asset_path,
        prim_path="/World/Humanoid"
    )

    # Create robot object
    robot = Robot(
        prim_path="/World/Humanoid",
        name="humanoid_robot",
        position=np.array([0.0, 0.0, 0.0]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0])
    )

    # Reset world
    world.reset()

    # Set camera view
    set_camera_view(eye=np.array([2.0, 2.0, 2.0]), target=np.array([0.0, 0.0, 1.0]))

    # Simulation loop
    for i in range(1000):
        # Get robot state
        joint_positions = robot.get_joints_state().positions
        joint_velocities = robot.get_joints_state().velocities

        # Apply control commands
        # (This would be your control algorithm)
        target_positions = np.zeros_like(joint_positions)

        # Set joint positions
        robot.set_joints_state(positions=target_positions)

        # Step simulation
        world.step(render=True)

        # Print info every 100 steps
        if i % 100 == 0:
            print(f"Step {i}: Joint positions = {joint_positions[:3]}...")

    return world, robot


def main():
    """Main function to run the simulation."""
    try:
        world, robot = setup_humanoid_simulation()
        print("Humanoid simulation completed successfully!")
    except Exception as e:
        print(f"Error running simulation: {e}")
    finally:
        # Clean up
        world.clear()
```

## Performance Optimization

### Optimizing Isaac Sim Scenes

```python
class IsaacSimOptimizer:
    """
    Optimizes Isaac Sim scenes for performance.
    """

    def __init__(self, stage):
        self.stage = stage

    def optimize_for_training(self):
        """Optimize scene for synthetic data generation."""
        # Reduce rendering quality during training
        self.set_render_quality("low")

        # Increase physics substeps for stability
        self.set_physics_substeps(8)

        # Use simpler collision meshes
        self.simplify_collision_meshes()

        # Disable unnecessary visual elements
        self.disable_non_essential_visuals()

    def optimize_for_inference(self):
        """Optimize scene for real-time inference."""
        # Set rendering quality to medium
        self.set_render_quality("medium")

        # Optimize for frame rate
        self.optimize_for_fps(60)

        # Use level-of-detail for distant objects
        self.setup_lod_system()

    def set_render_quality(self, quality_level):
        """Set rendering quality."""
        # Implementation depends on Isaac Sim API
        pass

    def optimize_for_fps(self, target_fps):
        """Optimize for target frame rate."""
        # Adjust physics settings
        # Simplify geometries
        # Reduce shadow resolution
        pass
```

## Exercise: Create Complete Humanoid Scene

Create a complete humanoid robot scene in Isaac Sim that includes:
1. Photorealistic humanoid robot model
2. Varied environment with obstacles
3. Proper lighting setup
4. Sensor configurations (camera, IMU, LiDAR)
5. Physics properties for realistic movement
6. Domain randomization capabilities

## Learning Outcomes

After completing this section, students will be able to:
1. Install and configure Isaac Sim
2. Create USD scenes for humanoid robots
3. Set up physics properties for realistic simulation
4. Generate synthetic data for AI training
5. Apply domain randomization techniques
6. Optimize scenes for performance

## Next Steps

Continue to [Synthetic Data and Perception](./synthetic-data-perception.md) to learn about generating training data and AI-powered perception systems.