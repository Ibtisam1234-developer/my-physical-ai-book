# Navigation Planning

## Introduction to Navigation in Humanoid Robots

Navigation for humanoid robots presents unique challenges compared to wheeled robots. Humanoid robots must consider:

- **Bipedal locomotion**: Walking on two legs with balance requirements
- **Center of Mass**: Maintaining stability during movement
- **Terrain adaptability**: Handling stairs, slopes, and uneven surfaces
- **Human-centric environments**: Navigating spaces designed for humans
- **Dynamic obstacles**: Moving around humans and other robots

## Isaac ROS Navigation Stack

Isaac ROS provides a comprehensive navigation stack with GPU acceleration:

### GPU-Accelerated Components

- **Path Planning**: GPU-accelerated A* and Dijkstra algorithms
- **Trajectory Generation**: Smooth path following with dynamic constraints
- **Obstacle Avoidance**: Real-time collision avoidance with GPU processing
- **SLAM**: Visual and LiDAR SLAM with CUDA acceleration

```python
# Example: Isaac ROS Navigation node
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import MarkerArray
from std_srvs.srv import Empty
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
import tf2_ros
import numpy as np


class IsaacNavigationNode(Node):
    """
    Isaac ROS Navigation node for humanoid robot navigation.
    """

    def __init__(self):
        super().__init__('isaac_navigation_node')

        # Initialize navigation components
        self.initialize_components()

        # Create publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.global_plan_pub = self.create_publisher(Path, '/plan', 10)
        self.local_plan_pub = self.create_publisher(Path, '/local_plan', 10)
        self.footstep_plan_pub = self.create_publisher(Path, '/footstep_plan', 10)

        # Create subscriptions
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Create action client for navigation
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Create service for navigation reset
        self.reset_service = self.create_service(
            Empty,
            '/navigation/reset',
            self.reset_navigation
        )

        # Navigation state
        self.current_pose = None
        self.navigation_goal = None
        self.is_navigating = False

        self.get_logger().info('Isaac Navigation Node initialized')

    def initialize_components(self):
        """Initialize navigation components."""
        # Initialize GPU-accelerated path planner
        self.path_planner = self.initialize_gpu_path_planner()

        # Initialize footstep planner for bipedal locomotion
        self.footstep_planner = self.initialize_footstep_planner()

        # Initialize local planner for obstacle avoidance
        self.local_planner = self.initialize_local_planner()

    def initialize_gpu_path_planner(self):
        """Initialize GPU-accelerated path planner."""
        # This would use Isaac ROS GPU path planning
        # For demonstration, we'll create a simple interface
        return GPUPathPlanner()

    def initialize_footstep_planner(self):
        """Initialize footstep planner for humanoid navigation."""
        # Plan footsteps considering bipedal constraints
        return FootstepPlanner()

    def initialize_local_planner(self):
        """Initialize local planner for obstacle avoidance."""
        # Handle dynamic obstacle avoidance
        return LocalPlanner()

    def odom_callback(self, msg):
        """Update robot pose from odometry."""
        self.current_pose = msg.pose.pose

        if self.is_navigating and self.navigation_goal:
            self.execute_navigation()

    def scan_callback(self, msg):
        """Process laser scan for obstacle detection."""
        # Process scan data for local planning
        self.local_planner.update_scan(msg)

    def navigate_to_pose(self, goal_pose):
        """Navigate to specified pose."""
        if self.nav_client.wait_for_server(timeout_sec=1.0):
            goal_msg = NavigateToPose.Goal()
            goal_msg.pose = goal_pose

            self.nav_client.send_goal_async(goal_msg)
            self.is_navigating = True
            self.navigation_goal = goal_pose
        else:
            self.get_logger().error('Navigation action server not available')

    def execute_navigation(self):
        """Execute navigation behavior."""
        if not self.current_pose or not self.navigation_goal:
            return

        # Plan global path
        global_path = self.path_planner.plan_path(
            self.current_pose,
            self.navigation_goal
        )

        # Plan footstep sequence for bipedal locomotion
        footstep_plan = self.footstep_planner.plan_footsteps(
            global_path,
            self.current_pose
        )

        # Generate local plan with obstacle avoidance
        local_plan = self.local_planner.plan_local_path(
            global_path,
            self.current_pose
        )

        # Execute planned trajectory
        self.follow_trajectory(local_plan, footstep_plan)

    def follow_trajectory(self, local_plan, footstep_plan):
        """Follow the planned trajectory."""
        # Generate velocity commands based on planned path
        cmd_vel = self.compute_velocity_command(local_plan)
        self.cmd_vel_pub.publish(cmd_vel)

        # Publish planned paths for visualization
        self.global_plan_pub.publish(local_plan)
        self.footstep_plan_pub.publish(footstep_plan)


class GPUPathPlanner:
    """
    GPU-accelerated path planner for fast path computation.
    """

    def __init__(self):
        # Initialize GPU path planning resources
        self.gpu_resources = self.initialize_gpu_resources()

    def plan_path(self, start_pose, goal_pose):
        """
        Plan path using GPU acceleration.

        Args:
            start_pose: Starting pose
            goal_pose: Goal pose

        Returns:
            Path: Planned path from start to goal
        """
        # Use Isaac ROS GPU path planning
        # This would leverage CUDA for A* or other algorithms
        path = self.compute_gpu_path(start_pose, goal_pose)
        return path

    def compute_gpu_path(self, start, goal):
        """Compute path using GPU resources."""
        # Implementation would use Isaac ROS GPU acceleration
        # For demonstration, return a simple straight-line path
        path = Path()
        path.poses = [start, goal]  # Simplified
        return path

    def initialize_gpu_resources(self):
        """Initialize GPU resources for path planning."""
        # Initialize CUDA context and GPU memory pools
        return {"initialized": True}
```

## Bipedal Locomotion Planning

### Footstep Planning for Humanoid Robots

```python
# Example: Footstep planning for bipedal navigation
import numpy as np
from scipy.spatial.distance import euclidean


class FootstepPlanner:
    """
    Plans footstep sequences for bipedal humanoid navigation.
    """

    def __init__(self):
        # Humanoid-specific parameters
        self.step_length = 0.3  # meters
        self.step_width = 0.15  # meters (stance width)
        self.step_height = 0.1  # clearance height
        self.max_turn_rate = 0.5  # radians per step

        # Balance constraints
        self.zmp_margin = 0.05  # Zero Moment Point safety margin
        self.com_height = 0.8   # Center of Mass height

    def plan_footsteps(self, global_path, current_pose):
        """
        Plan footstep sequence for following a global path.

        Args:
            global_path: Global path to follow
            current_pose: Current robot pose

        Returns:
            Path: Sequence of footstep poses
        """
        footsteps = []

        # Current foot positions (left and right)
        left_foot = self.get_left_foot_pose(current_pose)
        right_foot = self.get_right_foot_pose(current_pose)

        # Determine which foot to step with first
        supporting_foot = self.determine_supporting_foot(left_foot, right_foot)

        # Plan footsteps along the path
        for i, path_point in enumerate(global_path.poses):
            if i == 0:  # Skip current pose
                continue

            # Calculate desired step location
            step_location = self.calculate_step_location(
                current_pose,
                path_point,
                supporting_foot
            )

            # Check step feasibility
            if self.is_step_feasible(step_location, supporting_foot):
                footsteps.append(step_location)
                supporting_foot = "left" if supporting_foot == "right" else "right"

        return footsteps

    def calculate_step_location(self, current_pose, next_waypoint, supporting_foot):
        """Calculate where to place the next footstep."""
        # Calculate desired step direction
        dx = next_waypoint.pose.position.x - current_pose.position.x
        dy = next_waypoint.pose.position.y - current_pose.position.y
        distance = np.sqrt(dx*dx + dy*dy)

        # Calculate step angle
        step_angle = np.arctan2(dy, dx)

        # Determine step offset based on supporting foot
        if supporting_foot == "left":
            # Step with right foot
            lateral_offset = -self.step_width / 2
        else:
            # Step with left foot
            lateral_offset = self.step_width / 2

        # Calculate new foot position
        new_x = current_pose.position.x + self.step_length * np.cos(step_angle)
        new_y = current_pose.position.y + self.step_length * np.sin(step_angle)
        new_z = current_pose.position.z  # Keep same height

        # Apply lateral offset for stance width
        new_x += lateral_offset * np.sin(step_angle)
        new_y -= lateral_offset * np.cos(step_angle)

        # Create pose for new footstep
        step_pose = PoseStamped()
        step_pose.pose.position.x = new_x
        step_pose.pose.position.y = new_y
        step_pose.pose.position.z = new_z

        # Set orientation
        from geometry_msgs.msg import Quaternion
        import math
        quat = self.angle_to_quaternion(0, 0, step_angle)
        step_pose.pose.orientation = quat

        return step_pose

    def is_step_feasible(self, step_location, supporting_foot):
        """Check if the step is feasible given constraints."""
        # Check for obstacles in step region
        # Check for step height constraints
        # Check for balance constraints

        # For now, assume feasible
        return True

    def get_left_foot_pose(self, robot_pose):
        """Get left foot pose relative to robot."""
        # Calculate left foot position
        left_foot = PoseStamped()
        left_foot.pose.position.x = robot_pose.position.x
        left_foot.pose.position.y = robot_pose.position.y + self.step_width / 2
        left_foot.pose.position.z = robot_pose.position.z
        return left_foot

    def get_right_foot_pose(self, robot_pose):
        """Get right foot pose relative to robot."""
        # Calculate right foot position
        right_foot = PoseStamped()
        right_foot.pose.position.x = robot_pose.position.x
        right_foot.pose.position.y = robot_pose.position.y - self.step_width / 2
        right_foot.pose.position.z = robot_pose.position.z
        return right_foot

    def determine_supporting_foot(self, left_foot, right_foot):
        """Determine which foot is currently supporting the robot."""
        # For simplicity, alternate feet
        # In practice, this would consider current balance state
        return "left"  # or "right" based on current gait phase

    def angle_to_quaternion(self, roll, pitch, yaw):
        """Convert Euler angles to quaternion."""
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        quat = Quaternion()
        quat.w = w
        quat.x = x
        quat.y = y
        quat.z = z
        return quat
```

## Isaac ROS Navigation Integration

### Nav2 with Isaac ROS

```python
# Example: Isaac ROS integration with Nav2
import rclpy
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import math


class IsaacNav2Integration(Node):
    """
    Integrates Isaac ROS navigation capabilities with Nav2.
    """

    def __init__(self):
        super().__init__('isaac_nav2_integration')

        # Initialize TF2 for coordinate transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Create action client for Nav2
        self.nav_to_pose_client = ActionClient(
            self,
            NavigateToPose,
            'navigate_to_pose'
        )

        # Navigation parameters
        self.declare_parameter('planner_frequency', 1.0)
        self.declare_parameter('controller_frequency', 20.0)
        self.declare_parameter('max_linear_speed', 0.5)
        self.declare_parameter('max_angular_speed', 1.0)

        # Timer for navigation updates
        self.nav_timer = self.create_timer(
            1.0 / self.get_parameter('planner_frequency').value,
            self.navigation_callback
        )

        self.get_logger().info('Isaac Nav2 Integration initialized')

    def navigate_to_pose(self, pose, frame_id='map'):
        """
        Navigate to a pose using Isaac-enhanced Nav2.

        Args:
            pose: Target pose (PoseStamped or Pose)
            frame_id: Frame of reference for the pose
        """
        if not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error('Navigation server not available')
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = frame_id
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        if isinstance(pose, PoseStamped):
            goal_msg.pose.pose = pose.pose
        else:
            goal_msg.pose.pose = pose

        # Send goal with Isaac-specific options
        goal_msg.behavior_tree_id = 'IsaacDefaultBT'  # Custom BT for Isaac

        future = self.nav_to_pose_client.send_goal_async(goal_msg)
        return future

    def navigation_callback(self):
        """Periodic navigation updates."""
        # Check navigation status
        # Update local costmap with Isaac sensors
        # Handle navigation recovery behaviors
        pass

    def transform_pose(self, target_pose, target_frame):
        """
        Transform pose to target frame.

        Args:
            target_pose: Pose to transform
            target_frame: Target coordinate frame

        Returns:
            Transformed pose
        """
        try:
            # Get transform
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                target_pose.header.frame_id,
                rclpy.time.Time()
            )

            # Apply transform
            transformed_pose = PoseStamped()
            # Implementation of transformation
            return transformed_pose

        except TransformException as ex:
            self.get_logger().error(f'Could not transform pose: {ex}')
            return None

    def check_navigation_progress(self):
        """Check if navigation is progressing."""
        # Monitor navigation progress
        # Detect navigation failures
        # Trigger recovery behaviors if needed
        pass
```

## GPU-Accelerated Path Planning

### CUDA-Accelerated Path Finding

```python
# Example: GPU-accelerated path planning concepts
import numpy as np
from numba import cuda
import math


class GPUPathPlanner:
    """
    GPU-accelerated path planner using CUDA kernels.
    """

    def __init__(self):
        # Initialize GPU memory for grid map
        self.grid_map = None
        self.grid_width = 0
        self.grid_height = 0

    def initialize_grid(self, width, height, resolution=0.05):
        """
        Initialize occupancy grid for path planning.

        Args:
            width: Grid width in meters
            height: Grid height in meters
            resolution: Cell resolution in meters
        """
        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)
        self.resolution = resolution

        # Initialize grid on CPU
        self.cpu_grid = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)

        # Copy to GPU
        self.gpu_grid = cuda.to_device(self.cpu_grid)

    def plan_path_gpu(self, start, goal):
        """
        Plan path using GPU-accelerated A* algorithm.

        Args:
            start: Start coordinates (x, y) in grid indices
            goal: Goal coordinates (x, y) in grid indices

        Returns:
            List of waypoints forming the path
        """
        # Convert world coordinates to grid coordinates
        start_idx = self.world_to_grid(start)
        goal_idx = self.world_to_grid(goal)

        # Allocate GPU memory for pathfinding
        start_gpu = cuda.to_device(np.array([start_idx[0], start_idx[1]], dtype=np.int32))
        goal_gpu = cuda.to_device(np.array([goal_idx[0], goal_idx[1]], dtype=np.int32))

        # Launch GPU kernel for path planning
        threads_per_block = (16, 16)
        blocks_per_grid_x = math.ceil(self.grid_width / threads_per_block[0])
        blocks_per_grid_y = math.ceil(self.grid_height / threads_per_block[1])
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        # Execute GPU path planning kernel
        self.gpu_astar_kernel[blocks_per_grid, threads_per_block](
            self.gpu_grid,
            start_gpu,
            goal_gpu,
            self.grid_width,
            self.grid_height
        )

        # Copy result back to CPU
        result_path = self.extract_path_from_gpu()
        return result_path

    @cuda.jit
    def gpu_astar_kernel(grid, start, goal, width, height):
        """
        GPU kernel for A* pathfinding algorithm.
        """
        row, col = cuda.grid(2)

        if row < height and col < width:
            # A* algorithm implementation on GPU
            # This is a simplified representation
            # Actual implementation would be more complex
            pass

    def world_to_grid(self, world_coords):
        """Convert world coordinates to grid indices."""
        x_idx = int(world_coords[0] / self.resolution)
        y_idx = int(world_coords[1] / self.resolution)
        return (x_idx, y_idx)

    def grid_to_world(self, grid_indices):
        """Convert grid indices to world coordinates."""
        x_world = grid_indices[0] * self.resolution
        y_world = grid_indices[1] * self.resolution
        return (x_world, y_world)

    def extract_path_from_gpu(self):
        """Extract planned path from GPU memory."""
        # Implementation to extract path from GPU results
        return []  # Placeholder
```

## Local Planning and Obstacle Avoidance

### Dynamic Window Approach for Humanoids

```python
# Example: Local planner for humanoid robots
import numpy as np
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
import math


class HumanoidLocalPlanner:
    """
    Local planner for humanoid robot obstacle avoidance.
    """

    def __init__(self):
        # Robot-specific parameters
        self.max_linear_speed = 0.5  # m/s
        self.max_angular_speed = 1.0  # rad/s
        self.min_linear_speed = 0.1   # m/s
        self.min_angular_speed = 0.1  # rad/s

        # Safety margins
        self.safety_margin = 0.3      # meters
        self.horizon_time = 2.0       # seconds
        self.simulation_resolution = 0.1  # seconds

        # Footprint for collision checking
        self.robot_footprint = self.create_robot_footprint()

    def create_robot_footprint(self):
        """
        Create robot footprint polygon for collision checking.
        For humanoid, this represents the support polygon.
        """
        # Simplified rectangular footprint
        footprint = np.array([
            [-0.3, -0.15],  # rear left
            [0.3, -0.15],   # front left
            [0.3, 0.15],    # front right
            [-0.3, 0.15]    # rear right
        ])
        return footprint

    def plan_local_path(self, global_path, current_pose, laser_scan):
        """
        Plan local path with obstacle avoidance.

        Args:
            global_path: Global path to follow
            current_pose: Current robot pose
            laser_scan: Laser scan data for obstacle detection

        Returns:
            Twist: Velocity command for navigation
        """
        # Get current velocity
        current_vel = self.get_current_velocity()

        # Generate velocity candidates
        velocity_candidates = self.generate_velocity_candidates(current_vel)

        # Evaluate each candidate
        best_velocity = self.evaluate_candidates(
            velocity_candidates,
            current_pose,
            laser_scan,
            global_path
        )

        return best_velocity

    def generate_velocity_candidates(self, current_vel):
        """
        Generate velocity candidates for evaluation.

        Args:
            current_vel: Current velocity

        Returns:
            List of Twist velocities to evaluate
        """
        candidates = []

        # Linear velocity range
        linear_velocities = np.linspace(
            self.min_linear_speed,
            self.max_linear_speed,
            num=5
        )

        # Angular velocity range
        angular_velocities = np.linspace(
            -self.max_angular_speed,
            self.max_angular_speed,
            num=7
        )

        for lin_vel in linear_velocities:
            for ang_vel in angular_velocities:
                vel_cmd = Twist()
                vel_cmd.linear.x = lin_vel
                vel_cmd.angular.z = ang_vel
                candidates.append(vel_cmd)

        return candidates

    def evaluate_candidates(self, candidates, current_pose, laser_scan, global_path):
        """
        Evaluate velocity candidates based on multiple criteria.

        Args:
            candidates: List of velocity candidates
            current_pose: Current robot pose
            laser_scan: Laser scan data
            global_path: Global path to follow

        Returns:
            Twist: Best velocity command
        """
        best_score = float('-inf')
        best_candidate = Twist()

        for candidate in candidates:
            # Simulate trajectory
            trajectory = self.simulate_trajectory(
                current_pose,
                candidate,
                self.horizon_time
            )

            # Evaluate trajectory
            score = self.evaluate_trajectory(
                trajectory,
                laser_scan,
                global_path
            )

            if score > best_score:
                best_score = score
                best_candidate = candidate

        return best_candidate

    def simulate_trajectory(self, start_pose, velocity_cmd, duration):
        """
        Simulate robot trajectory for given velocity command.

        Args:
            start_pose: Starting pose
            velocity_cmd: Velocity command to execute
            duration: Duration to simulate

        Returns:
            List of poses representing the trajectory
        """
        trajectory = []
        current_pose = start_pose
        time_step = self.simulation_resolution

        for t in np.arange(0, duration, time_step):
            # Update pose based on velocity
            new_pose = self.update_pose(
                current_pose,
                velocity_cmd,
                time_step
            )
            trajectory.append(new_pose)
            current_pose = new_pose

        return trajectory

    def update_pose(self, current_pose, velocity_cmd, dt):
        """
        Update robot pose based on velocity command.

        Args:
            current_pose: Current pose
            velocity_cmd: Velocity command
            dt: Time step

        Returns:
            Updated pose
        """
        # Create copy of current pose
        new_pose = PoseStamped()
        new_pose.pose = current_pose

        # Apply motion model for differential drive
        # For humanoid, this would be more complex
        linear_dist = velocity_cmd.linear.x * dt
        angular_change = velocity_cmd.angular.z * dt

        # Update position
        new_pose.pose.position.x += linear_dist * math.cos(angular_change)
        new_pose.pose.position.y += linear_dist * math.sin(angular_change)

        # Update orientation
        current_yaw = self.quaternion_to_yaw(new_pose.pose.orientation)
        new_yaw = current_yaw + angular_change
        new_pose.pose.orientation = self.yaw_to_quaternion(new_yaw)

        return new_pose

    def evaluate_trajectory(self, trajectory, laser_scan, global_path):
        """
        Evaluate trajectory based on multiple criteria.

        Args:
            trajectory: Simulated trajectory
            laser_scan: Laser scan data
            global_path: Global path to follow

        Returns:
            float: Evaluation score
        """
        # Collision score (penalty for collisions)
        collision_penalty = self.check_collision(trajectory, laser_scan)

        # Goal progress score (reward for moving toward goal)
        goal_reward = self.calculate_goal_progress(trajectory, global_path)

        # Smoothness score (reward for smooth trajectories)
        smoothness_reward = self.calculate_smoothness(trajectory)

        # Combined score
        total_score = goal_reward + smoothness_reward - collision_penalty
        return total_score

    def check_collision(self, trajectory, laser_scan):
        """
        Check if trajectory collides with obstacles.

        Args:
            trajectory: Trajectory to check
            laser_scan: Laser scan data

        Returns:
            float: Collision penalty score
        """
        collision_penalty = 0.0

        for pose in trajectory:
            # Check if pose is in collision
            if self.is_pose_in_collision(pose, laser_scan):
                collision_penalty += 1000.0  # Heavy penalty for collisions

        return collision_penalty

    def is_pose_in_collision(self, pose, laser_scan):
        """
        Check if a pose is in collision with obstacles.

        Args:
            pose: Pose to check
            laser_scan: Laser scan data

        Returns:
            bool: True if in collision, False otherwise
        """
        # Check if robot footprint intersects with obstacles
        # This would involve checking the robot's support polygon
        # against obstacle positions from laser scan
        return False  # Simplified

    def calculate_goal_progress(self, trajectory, global_path):
        """
        Calculate reward based on progress toward goal.

        Args:
            trajectory: Trajectory to evaluate
            global_path: Global path

        Returns:
            float: Goal progress reward
        """
        if not trajectory or not global_path.poses:
            return 0.0

        # Calculate distance to goal for final pose
        final_pose = trajectory[-1]
        goal_pose = global_path.poses[-1]

        distance_to_goal = self.calculate_distance(
            final_pose.pose.position,
            goal_pose.pose.position
        )

        # Reward is inversely proportional to distance
        reward = 1.0 / (1.0 + distance_to_goal)
        return reward

    def calculate_smoothness(self, trajectory):
        """
        Calculate reward based on trajectory smoothness.

        Args:
            trajectory: Trajectory to evaluate

        Returns:
            float: Smoothness reward
        """
        if len(trajectory) < 2:
            return 0.0

        # Calculate curvature of trajectory
        total_curvature = 0.0

        for i in range(1, len(trajectory) - 1):
            p1 = trajectory[i-1].pose.position
            p2 = trajectory[i].pose.position
            p3 = trajectory[i+1].pose.position

            # Calculate angle between segments
            angle = self.calculate_angle(p1, p2, p3)
            total_curvature += abs(angle)

        # Reward for low curvature (smooth paths)
        smoothness_reward = 1.0 / (1.0 + total_curvature)
        return smoothness_reward

    def calculate_distance(self, pos1, pos2):
        """Calculate Euclidean distance between two positions."""
        dx = pos2.x - pos1.x
        dy = pos2.y - pos1.y
        return math.sqrt(dx*dx + dy*dy)

    def calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points."""
        v1 = (p1.x - p2.x, p1.y - p2.y)
        v2 = (p3.x - p2.x, p3.y - p2.y)

        dot_product = v1[0]*v2[0] + v1[1]*v2[1]
        magnitude1 = math.sqrt(v1[0]**2 + v1[1]**2)
        magnitude2 = math.sqrt(v2[0]**2 + v2[1]**2)

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        cos_angle = dot_product / (magnitude1 * magnitude2)
        cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to [-1, 1]
        angle = math.acos(cos_angle)

        return angle

    def quaternion_to_yaw(self, quaternion):
        """Convert quaternion to yaw angle."""
        siny_cosp = 2 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
        cosy_cosp = 1 - 2 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def yaw_to_quaternion(self, yaw):
        """Convert yaw angle to quaternion."""
        from geometry_msgs.msg import Quaternion
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cq = math.cos(0.0 * 0.5)
        sq = math.sin(0.0 * 0.5)

        quat = Quaternion()
        quat.w = cy * cq + sy * sq
        quat.x = cy * sq - sy * cq
        quat.y = sy * cq + cy * sq
        quat.z = sy * sq - cy * cq
        return quat
```

## Isaac Navigation Configuration

### Navigation Parameters for Humanoids

```yaml
# Isaac Navigation Configuration for Humanoid Robot
# File: config/humanoid_navigation.yaml

amcl:
  ros__parameters:
    use_sim_time: False
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha_slow_filter: 0.001
    alpha_fast_filter: 0.1
    base_frame_id: "base_link"
    beam_skip_distance: 0.5
    beam_skip_error_threshold: 0.9
    beam_skip_threshold: 0.3
    do_beamskip: false
    global_frame_id: "map"
    lambda_short: 0.1
    likelihood_max_dist: 2.0
    max_beams: 60
    max_particles: 2000
    min_particles: 500
    odom_frame_id: "odom"
    pf_err: 0.05
    pf_z: 0.5
    recovery_alpha_fast: 0.0
    recovery_alpha_slow: 0.0
    resample_interval: 1
    robot_model_type: "nav2_amcl::DifferentialMotionModel"
    save_pose_rate: 0.5
    sigma_hit: 0.2
    tf_broadcast: true
    transform_tolerance: 1.0
    update_min_a: 0.2
    update_min_d: 0.25
    z_hit: 0.5
    z_max: 0.05
    z_rand: 0.5
    z_short: 0.05

bt_navigator:
  ros__parameters:
    use_sim_time: False
    global_frame: "map"
    robot_base_frame: "base_link"
    odom_topic: "odom"
    bt_loop_duration: 10
    default_server_timeout: 20
    enable_groot_monitoring: True
    groot_zmq_publisher_port: 1666
    groot_zmq_server_port: 1667
    # Use Isaac-specific behavior tree
    default_bt_xml_filename: "config/isaac_default_bt.xml"
    plugin_lib_names:
      - nav2_compute_path_to_pose_action_bt_node
      - nav2_follow_path_action_bt_node
      - nav2_back_up_action_bt_node
      - nav2_spin_action_bt_node
      - nav2_wait_action_bt_node
      - nav2_clear_costmap_service_bt_node
      - nav2_is_stuck_condition_bt_node
      - nav2_goal_reached_condition_bt_node
      - nav2_goal_updated_condition_bt_node
      - nav2_initial_pose_received_condition_bt_node
      - nav2_reinitialize_global_localization_service_bt_node
      - nav2_rate_controller_bt_node
      - nav2_distance_controller_bt_node
      - nav2_speed_controller_bt_node
      - nav2_truncate_path_action_bt_node
      - nav2_goal_updater_node_bt_node
      - nav2_recovery_node_bt_node
      - nav2_pipeline_sequence_bt_node
      - nav2_round_robin_node_bt_node
      - nav2_transform_available_condition_bt_node
      - nav2_time_expired_condition_bt_node
      - nav2_path_expiring_timer_condition
      - nav2_distance_traveled_condition_bt_node
      - nav2_single_trigger_bt_node
      - nav2_is_battery_low_condition_bt_node
      - nav2_navigate_through_poses_action_bt_node
      - nav2_navigate_to_pose_action_bt_node
      - nav2_remove_passed_goals_action_bt_node
      - nav2_planner_selector_bt_node
      - nav2_controller_selector_bt_node
      - nav2_goal_checker_selector_bt_node

controller_server:
  ros__parameters:
    use_sim_time: False
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    # Use Isaac-specific controller for bipedal locomotion
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]

    # DWB Controller for Isaac
    FollowPath:
      plugin: "nav2_mppi_controller::MPPICtrl"
      # Humanoid-specific parameters
      time_steps: 20
      dt: 0.1
      # Footstep planning integration
      use_footstep_planning: True
      max_step_length: 0.3
      step_width: 0.15
      step_height: 0.1

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: "odom"
      robot_base_frame: "base_link"
      use_sim_time: False
      rolling_window: true
      width: 6
      height: 6
      resolution: 0.05
      robot_radius: 0.3  # Larger for humanoid safety
      plugins: ["voxel_layer", "inflation_layer"]
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      voxel_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        enabled: True
        publish_voxel_map: True
        origin_z: 0.0
        z_resolution: 0.2
        z_voxels: 10
        max_obstacle_height: 2.0
        mark_threshold: 0
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0

global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 0.5
      global_frame: "map"
      robot_base_frame: "base_link"
      use_sim_time: False
      robot_radius: 0.3
      resolution: 0.05
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: True
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: True
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55

planner_server:
  ros__parameters:
    expected_planner_frequency: 2.0
    use_sim_time: False
    planner_plugins: ["GridBased"]
    GridBased:
      # Use Isaac GPU-accelerated planner
      plugin: "nav2_navfn_planner/NavfnPlanner"
      tolerance: 0.5
      use_astar: false
      allow_unknown: true
```

## Exercise: Implement Humanoid Navigation System

Create a complete navigation system for a humanoid robot that includes:
1. GPU-accelerated path planning
2. Footstep planning for bipedal locomotion
3. Local obstacle avoidance with humanoid-specific constraints
4. Integration with Isaac ROS navigation stack
5. Configuration for real-world humanoid navigation

## Learning Outcomes

After completing this section, students will be able to:
1. Implement GPU-accelerated path planning for humanoid robots
2. Plan footsteps for bipedal locomotion navigation
3. Use Isaac ROS navigation stack with Nav2
4. Configure navigation parameters for humanoid robots
5. Handle dynamic obstacle avoidance for bipeds
6. Integrate perception and navigation systems

## Next Steps

Continue to [Isaac Sim Practical Exercises](./isaac-sim-practical-exercises.md) to apply navigation concepts in Isaac Sim environments.