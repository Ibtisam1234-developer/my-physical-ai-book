# Isaac Sim Practical Exercises

## Exercise 1: Basic Humanoid Robot Simulation

### Objective
Create a basic humanoid robot simulation in Isaac Sim and verify functionality.

### Steps

1. **Install Isaac Sim** (if not already installed)
   ```bash
   # Download from NVIDIA Developer website
   # Follow installation guide for your OS
   ```

2. **Create a new Isaac Sim scene**
   ```python
   # Create basic_scene.py
   import omni
   from omni.isaac.core import World
   from omni.isaac.core.utils.stage import add_reference_to_stage
   from omni.isaac.core.utils.prims import create_prim
   import numpy as np

   # Create world
   world = World(stage_units_in_meters=1.0)

   # Add ground plane
   create_prim(
       prim_path="/World/ground_plane",
       prim_type="Plane",
       position=np.array([0, 0, 0]),
       orientation=np.array([0, 0, 0, 1])
   )

   # Add simple humanoid robot
   add_reference_to_stage(
       usd_path="path/to/humanoid_robot.usd",
       prim_path="/World/HumanoidRobot"
   )

   # Reset world
   world.reset()

   # Run simulation
   for i in range(1000):
       world.step(render=True)

   # Cleanup
   world.clear()
   ```

3. **Run the simulation**
   ```bash
   python basic_scene.py
   ```

4. **Verify the simulation runs without errors**
   - Check that the robot appears in the scene
   - Verify the ground plane is visible
   - Ensure the simulation runs smoothly

### Expected Outcome
A basic Isaac Sim scene with a humanoid robot on a ground plane, running for 1000 simulation steps.

## Exercise 2: Sensor Integration in Isaac Sim

### Objective
Add sensors to the humanoid robot and capture sensor data.

### Steps

1. **Create a sensor-equipped humanoid scene**
   ```python
   # sensors_scene.py
   import omni
   from omni.isaac.core import World
   from omni.isaac.core.utils.stage import add_reference_to_stage
   from omni.isaac.core.utils.prims import create_prim
   from omni.isaac.sensor import Camera
   import numpy as np

   # Create world
   world = World(stage_units_in_meters=1.0)

   # Add ground plane
   create_prim(
       prim_path="/World/ground_plane",
       prim_type="Plane",
       position=np.array([0, 0, 0])
   )

   # Add humanoid robot
   add_reference_to_stage(
       usd_path="path/to/humanoid_robot.usd",
       prim_path="/World/HumanoidRobot"
   )

   # Create camera sensor
   camera = Camera(
       prim_path="/World/HumanoidRobot/head/camera",
       frequency=30,
       resolution=(640, 480)
   )

   # Reset world
   world.reset()

   # Run simulation and capture data
   for i in range(300):  # 10 seconds at 30Hz
       world.step(render=True)

       # Capture camera data
       if i % 10 == 0:  # Every 10 steps
           image = camera.get_rgba()
           depth = camera.get_depth()
           print(f"Step {i}: Image shape: {image.shape}, Depth range: {depth.min():.2f}-{depth.max():.2f}")

   # Cleanup
   world.clear()
   ```

2. **Run the sensor simulation**
   ```bash
   python sensors_scene.py
   ```

3. **Verify sensor data capture**
   - Check that camera images are captured
   - Verify depth data is generated
   - Ensure no runtime errors occur

### Expected Outcome
A humanoid robot with camera sensor capturing images and depth data during simulation.

## Exercise 3: Synthetic Data Generation

### Objective
Generate synthetic training data using Isaac Sim.

### Steps

1. **Create synthetic data generator**
   ```python
   # synthetic_data_generator.py
   import omni
   from omni.isaac.core import World
   from omni.isaac.core.utils.stage import add_reference_to_stage
   from omni.isaac.core.utils.prims import create_prim
   from omni.isaac.synthetic_utils import SyntheticDataHelper
   import numpy as np
   import random
   import os
   from PIL import Image

   class SyntheticDataGenerator:
       def __init__(self, output_dir="synthetic_data"):
           self.output_dir = output_dir
           self.world = World(stage_units_in_meters=1.0)
           self.sd_helper = SyntheticDataHelper()

           # Create output directory
           os.makedirs(output_dir, exist_ok=True)

       def setup_scene(self):
           """Set up the synthetic data scene."""
           # Add ground plane
           create_prim(
               prim_path="/World/ground_plane",
               prim_type="Plane",
               position=np.array([0, 0, 0])
           )

           # Add humanoid robot
           add_reference_to_stage(
               usd_path="path/to/humanoid_robot.usd",
               prim_path="/World/HumanoidRobot"
           )

           # Add objects for data generation
           self.add_random_objects()

           # Add lighting
           self.add_lighting()

       def add_random_objects(self):
           """Add random objects to the scene."""
           object_shapes = ["Cube", "Sphere", "Cylinder"]
           for i in range(5):
               shape = random.choice(object_shapes)
               position = np.array([
                   random.uniform(-3, 3),
                   random.uniform(-3, 3),
                   0.5
               ])

               create_prim(
                   prim_path=f"/World/Object_{i}",
                   prim_type=shape,
                   position=position,
                   scale=np.array([0.3, 0.3, 0.3])
               )

       def add_lighting(self):
           """Add random lighting to the scene."""
           from omni.isaac.core.utils.prims import get_prim_at_path
           from pxr import UsdLux

           # Add dome light
           create_prim(
               prim_path="/World/DomeLight",
               prim_type="DomeLight",
               attributes={"color": (random.uniform(0.8, 1.0), random.uniform(0.8, 1.0), random.uniform(0.8, 1.0)),
                         "intensity": random.uniform(500, 1500)}
           )

       def generate_sample(self, sample_id):
           """Generate a single synthetic data sample."""
           # Randomize environment
           self.randomize_environment()

           # Capture RGB image
           rgb_data = self.sd_helper.get_rgb_data()

           # Capture depth
           depth_data = self.sd_helper.get_depth_data()

           # Capture segmentation
           seg_data = self.sd_helper.get_segmentation_data()

           # Save sample
           sample_dir = os.path.join(self.output_dir, f"sample_{sample_id:06d}")
           os.makedirs(sample_dir, exist_ok=True)

           # Save RGB image
           rgb_image = Image.fromarray(rgb_data)
           rgb_image.save(os.path.join(sample_dir, "rgb.png"))

           # Save depth
           depth_image = Image.fromarray(depth_data)
           depth_image.save(os.path.join(sample_dir, "depth.png"))

           # Save segmentation
           seg_image = Image.fromarray(seg_data)
           seg_image.save(os.path.join(sample_dir, "segmentation.png"))

           print(f"Saved sample {sample_id}")

       def randomize_environment(self):
           """Randomize environment parameters."""
           # Randomize object positions
           for i in range(5):
               new_pos = np.array([
                   random.uniform(-3, 3),
                   random.uniform(-3, 3),
                   0.5
               ])
               # Update object position (implementation depends on specific API)

           # Randomize lighting
           # Update dome light properties

       def generate_dataset(self, num_samples=100):
           """Generate synthetic dataset."""
           self.setup_scene()
           self.world.reset()

           for i in range(num_samples):
               self.generate_sample(i)
               self.world.step(render=True)

           self.world.clear()

   # Run the data generator
   if __name__ == "__main__":
       generator = SyntheticDataGenerator("humanoid_synthetic_data")
       generator.generate_dataset(num_samples=50)  # Generate 50 samples
   ```

2. **Run synthetic data generation**
   ```bash
   python synthetic_data_generator.py
   ```

3. **Verify dataset generation**
   - Check that samples are saved in the output directory
   - Verify RGB, depth, and segmentation images are generated
   - Ensure randomization is working properly

### Expected Outcome
A dataset of 50 synthetic samples with RGB, depth, and segmentation images for training AI models.

## Exercise 4: Isaac ROS Integration

### Objective
Integrate Isaac Sim with ROS 2 using Isaac ROS bridges.

### Steps

1. **Set up Isaac ROS environment**
   ```bash
   # Source ROS 2 and Isaac ROS
   source /opt/ros/humble/setup.bash
   source /path/to/isaac_ros_ws/install/setup.bash
   ```

2. **Create Isaac ROS launch file**
   ```python
   # isaac_ros_launch.py
   from launch import LaunchDescription
   from launch_ros.actions import Node
   from launch.actions import DeclareLaunchArgument
   from launch.substitutions import LaunchConfiguration
   from launch.conditions import IfCondition
   from launch.substitutions import PythonExpression

   def generate_launch_description():
       """Generate Isaac ROS launch description."""

       # Declare launch arguments
       use_sim_time = LaunchConfiguration('use_sim_time')
       camera_enabled = LaunchConfiguration('camera_enabled')

       # Isaac ROS camera bridge node
       camera_bridge = Node(
           package='isaac_ros_image_proc',
           executable='image_format_converter',
           name='image_format_converter',
           parameters=[{
               'use_sim_time': use_sim_time,
               'input_format': 'rgba8',
               'output_format': 'rgb8'
           }],
           condition=IfCondition(camera_enabled)
       )

       # Isaac ROS depth bridge node
       depth_bridge = Node(
           package='isaac_ros_depth_image_proc',
           executable='depth_image_proc',
           name='depth_image_processor',
           parameters=[{
               'use_sim_time': use_sim_time
           }]
       )

       # Isaac ROS perception pipeline
       perception_pipeline = Node(
           package='isaac_ros_detectnet',
           executable='detectnet_node',
           name='detectnet',
           parameters=[{
               'use_sim_time': use_sim_time,
               'input_topic': '/camera/image_rect_color',
               'model_name': 'ssd_mobilenet_v2_coco',
               'confidence_threshold': 0.5
           }]
       )

       return LaunchDescription([
           DeclareLaunchArgument('use_sim_time', default_value='true'),
           DeclareLaunchArgument('camera_enabled', default_value='true'),
           camera_bridge,
           depth_bridge,
           perception_pipeline
       ])
   ```

3. **Test Isaac ROS integration**
   ```bash
   # Terminal 1: Launch Isaac Sim
   python -m omni.isaac.examples.simple_world.simple_world

   # Terminal 2: Launch Isaac ROS bridges
   ros2 launch isaac_ros_launch.py
   ```

4. **Verify integration**
   ```bash
   # Check for ROS topics
   ros2 topic list | grep camera
   ros2 topic list | grep perception

   # Echo camera data
   ros2 topic echo /camera/image_rect_color --field data
   ```

### Expected Outcome
Isaac Sim running with Isaac ROS bridges publishing camera and perception data to ROS topics.

## Exercise 5: GPU-Accelerated Perception Pipeline

### Objective
Create a GPU-accelerated perception pipeline using Isaac Sim and Isaac ROS.

### Steps

1. **Create GPU perception node**
   ```python
   # gpu_perception_node.py
   import rclpy
   from rclpy.node import Node
   from sensor_msgs.msg import Image
   from vision_msgs.msg import Detection2DArray
   from geometry_msgs.msg import PointStamped
   from std_msgs.msg import Header
   import numpy as np
   import cv2
   from cv_bridge import CvBridge

   class GPUPerceptionNode(Node):
       """
       GPU-accelerated perception node using Isaac ROS.
       """

       def __init__(self):
           super().__init__('gpu_perception_node')

           # Initialize CV bridge
           self.cv_bridge = CvBridge()

           # Create subscriptions
           self.image_sub = self.create_subscription(
               Image,
               '/camera/image_rect_color',
               self.image_callback,
               10
           )

           # Create publishers
           self.detection_pub = self.create_publisher(
               Detection2DArray,
               '/perception/detections',
               10
           )

           self.point_cloud_pub = self.create_publisher(
               PointStamped,
               '/perception/point_cloud',
               10
           )

           # GPU-accelerated inference model
           self.setup_gpu_inference()

           self.get_logger().info('GPU Perception Node initialized')

       def setup_gpu_inference(self):
           """Setup GPU-accelerated inference."""
           # This would typically use TensorRT or similar
           # For demonstration, we'll use a placeholder
           self.inference_engine = "TensorRT"  # Placeholder
           self.model_loaded = True

       def image_callback(self, msg):
           """Process camera image with GPU acceleration."""
           if not self.model_loaded:
               return

           # Convert ROS image to OpenCV
           cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "rgb8")

           # Perform GPU-accelerated inference
           detections = self.gpu_inference(cv_image)

           # Publish detections
           self.publish_detections(detections, msg.header)

       def gpu_inference(self, image):
           """Perform GPU-accelerated object detection."""
           # Placeholder for actual GPU inference
           # This would use Isaac ROS DNN packages
           # For now, return dummy detections
           return []

       def publish_detections(self, detections, header):
           """Publish object detections."""
           detection_array = Detection2DArray()
           detection_array.header = header
           detection_array.detections = detections

           self.detection_pub.publish(detection_array)

   def main(args=None):
       rclpy.init(args=args)
       node = GPUPerceptionNode()

       try:
           rclpy.spin(node)
       except KeyboardInterrupt:
           pass
       finally:
           node.destroy_node()
           rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

2. **Run GPU perception pipeline**
   ```bash
   # Terminal 1: Launch Isaac Sim with camera
   python camera_scene.py

   # Terminal 2: Run GPU perception node
   ros2 run your_package gpu_perception_node
   ```

3. **Verify GPU acceleration**
   - Monitor GPU utilization during inference
   - Check that detections are published to ROS topics
   - Verify frame rates are acceptable

### Expected Outcome
GPU-accelerated perception pipeline processing camera data and publishing detections to ROS topics.

## Exercise 6: Navigation in Isaac Sim

### Objective
Set up navigation for a humanoid robot in Isaac Sim using Isaac ROS navigation packages.

### Steps

1. **Create navigation scene**
   ```python
   # navigation_scene.py
   import omni
   from omni.isaac.core import World
   from omni.isaac.core.utils.stage import add_reference_to_stage
   from omni.isaac.core.utils.prims import create_prim
   from omni.isaac.core.robots import Robot
   import numpy as np

   class NavigationScene:
       def __init__(self):
           self.world = World(stage_units_in_meters=1.0)

       def setup_navigation_scene(self):
           """Set up navigation scene with obstacles."""
           # Create ground plane
           create_prim(
               prim_path="/World/ground_plane",
               prim_type="Plane",
               position=np.array([0, 0, 0])
           )

           # Add humanoid robot
           add_reference_to_stage(
               usd_path="path/to/humanoid_robot.usd",
               prim_path="/World/HumanoidRobot"
           )

           # Add navigation obstacles
           self.add_navigation_obstacles()

           # Add goal marker
           self.add_goal_marker()

       def add_navigation_obstacles(self):
           """Add obstacles for navigation."""
           # Add walls
           create_prim(
               prim_path="/World/wall_1",
               prim_type="Cube",
               position=np.array([5, 0, 0.5]),
               scale=np.array([0.1, 10, 1])
           )

           create_prim(
               prim_path="/World/wall_2",
               prim_type="Cube",
               position=np.array([-5, 0, 0.5]),
               scale=np.array([0.1, 10, 1])
           )

           # Add random obstacles
           for i in range(10):
               position = np.array([
                   np.random.uniform(-4, 4),
                   np.random.uniform(-4, 4),
                   0.3
               ])
               create_prim(
                   prim_path=f"/World/obstacle_{i}",
                   prim_type="Cylinder",
                   position=position,
                   scale=np.array([0.2, 0.2, 0.6])
               )

       def add_goal_marker(self):
           """Add navigation goal."""
           create_prim(
               prim_path="/World/goal_marker",
               prim_type="Sphere",
               position=np.array([8, 8, 0.5]),
               scale=np.array([0.3, 0.3, 0.3])
           )

       def run_navigation_simulation(self):
           """Run navigation simulation."""
           self.setup_navigation_scene()
           self.world.reset()

           for i in range(5000):  # 5000 steps
               self.world.step(render=True)

               # This would typically interface with ROS navigation
               # For now, just run simulation
               if i % 500 == 0:
                   print(f"Navigation simulation step: {i}")

           self.world.clear()

   if __name__ == "__main__":
       scene = NavigationScene()
       scene.run_navigation_simulation()
   ```

2. **Integrate with Isaac ROS navigation**
   ```python
   # navigation_integration.py
   import rclpy
   from rclpy.node import Node
   from geometry_msgs.msg import PoseStamped
   from nav2_msgs.action import NavigateToPose
   from rclpy.action import ActionClient

   class IsaacNavigationController(Node):
       """
       Navigation controller for Isaac Sim humanoid robot.
       """

       def __init__(self):
           super().__init__('isaac_navigation_controller')

           # Create navigation action client
           self.nav_client = ActionClient(
               self,
               NavigateToPose,
               'navigate_to_pose'
           )

           # Set navigation goal
           self.set_navigation_goal()

       def set_navigation_goal(self):
           """Set navigation goal for humanoid robot."""
           goal_pose = PoseStamped()
           goal_pose.header.frame_id = 'map'
           goal_pose.pose.position.x = 8.0
           goal_pose.pose.position.y = 8.0
           goal_pose.pose.orientation.w = 1.0

           self.navigate_to_pose(goal_pose)

       def navigate_to_pose(self, pose):
           """Send navigation goal."""
           if not self.nav_client.wait_for_server(timeout_sec=1.0):
               self.get_logger().error('Navigation server not available')
               return

           goal_msg = NavigateToPose.Goal()
           goal_msg.pose = pose

           self.nav_client.send_goal_async(goal_msg)

   def main(args=None):
       rclpy.init(args=args)
       node = IsaacNavigationController()

       try:
           rclpy.spin(node)
       except KeyboardInterrupt:
           pass
       finally:
           node.destroy_node()
           rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

3. **Run navigation simulation**
   ```bash
   # Terminal 1: Run Isaac Sim navigation scene
   python navigation_scene.py

   # Terminal 2: Run navigation controller
   ros2 run your_package navigation_integration
   ```

### Expected Outcome
Humanoid robot navigating through obstacles in Isaac Sim with Isaac ROS navigation stack.

## Exercise 7: Complete AI-Powered Humanoid System

### Objective
Combine all elements into a complete AI-powered humanoid system.

### Steps

1. **Create system integration**
   ```python
   # ai_humanoid_system.py
   import rclpy
   from rclpy.node import Node
   from sensor_msgs.msg import Image, LaserScan
   from geometry_msgs.msg import Twist
   from std_msgs.msg import String
   import threading
   import time

   class AIHumanoidSystem(Node):
       """
       Complete AI-powered humanoid system integration.
       """

       def __init__(self):
           super().__init__('ai_humanoid_system')

           # Initialize subsystems
           self.perception_subsystem = self.initialize_perception()
           self.navigation_subsystem = self.initialize_navigation()
           self.control_subsystem = self.initialize_control()

           # Create publishers and subscribers
           self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
           self.status_pub = self.create_publisher(String, '/system_status', 10)

           # Subscribe to sensor data
           self.image_sub = self.create_subscription(
               Image, '/camera/image_rect_color', self.process_camera_data, 10
           )
           self.scan_sub = self.create_subscription(
               LaserScan, '/scan', self.process_lidar_data, 10
           )

           # System state
           self.system_active = True
           self.ai_behavior_thread = None

           self.get_logger().info('AI Humanoid System initialized')

       def initialize_perception(self):
           """Initialize perception subsystem."""
           # This would initialize Isaac ROS perception packages
           return {
               'object_detection': True,
               'depth_estimation': True,
               'segmentation': True
           }

       def initialize_navigation(self):
           """Initialize navigation subsystem."""
           # This would initialize Isaac ROS navigation
           return {
               'path_planning': True,
               'local_planning': True,
               'footstep_planning': True
           }

       def initialize_control(self):
           """Initialize control subsystem."""
           # This would initialize Isaac ROS control
           return {
               'balance_control': True,
               'locomotion': True,
               'manipulation': True
           }

       def start_ai_behavior(self):
           """Start main AI behavior thread."""
           self.ai_behavior_thread = threading.Thread(target=self.ai_behavior_loop)
           self.ai_behavior_thread.daemon = True
           self.ai_behavior_thread.start()

       def ai_behavior_loop(self):
           """Main AI behavior loop."""
           while self.system_active:
               # Process perception data
               self.process_perception()

               # Make decisions based on environment
               self.make_decisions()

               # Execute actions
               self.execute_actions()

               # Sleep for performance
               time.sleep(0.1)

       def process_perception(self):
           """Process perception data."""
           # This would process data from Isaac ROS perception
           # and update internal world model
           pass

       def make_decisions(self):
           """Make high-level decisions."""
           # This would implement AI decision-making
           # based on perception and goals
           pass

       def execute_actions(self):
           """Execute actions based on decisions."""
           # This would send commands to actuators
           # through Isaac ROS control interfaces
           pass

       def process_camera_data(self, msg):
           """Process camera data."""
           # This would feed camera data to perception system
           pass

       def process_lidar_data(self, msg):
           """Process LiDAR data."""
           # This would feed LiDAR data to perception system
           pass

       def publish_status(self, status_msg):
           """Publish system status."""
           status = String()
           status.data = status_msg
           self.status_pub.publish(status)

   def main(args=None):
       rclpy.init(args=args)
       node = AIHumanoidSystem()

       # Start AI behavior
       node.start_ai_behavior()

       try:
           rclpy.spin(node)
       except KeyboardInterrupt:
           node.system_active = False
           if node.ai_behavior_thread:
               node.ai_behavior_thread.join()
       finally:
           node.destroy_node()
           rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

2. **Run complete system**
   ```bash
   # Terminal 1: Launch Isaac Sim with humanoid robot
   python navigation_scene.py

   # Terminal 2: Launch Isaac ROS bridges
   ros2 launch isaac_ros_launch.py

   # Terminal 3: Run AI humanoid system
   ros2 run your_package ai_humanoid_system
   ```

### Expected Outcome
A complete AI-powered humanoid system running in Isaac Sim with integrated perception, navigation, and control systems.

## Assessment

### Evaluation Criteria

Each exercise will be evaluated based on:

1. **Functionality** (40%): Does the system work as expected?
2. **Correctness** (30%): Are the implementations correct and following best practices?
3. **Performance** (20%): Is the system running efficiently?
4. **Documentation** (10%): Are the implementations well-documented?

### Submission Requirements

For each exercise, submit:
- Source code files
- Configuration files
- Screenshots or videos of the system working
- Brief report explaining the implementation

## Learning Outcomes

After completing these practical exercises, students will be able to:
1. Set up and configure Isaac Sim for humanoid robotics
2. Integrate sensors and capture synthetic data
3. Use Isaac ROS packages for perception and navigation
4. Implement GPU-accelerated AI pipelines
5. Create complete AI-powered robotic systems
6. Apply domain randomization techniques

## Next Steps

Continue to [Knowledge Checks](./knowledge-checks.md) to assess understanding of Isaac Sim concepts and practical applications.