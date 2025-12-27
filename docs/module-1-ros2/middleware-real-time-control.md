# Middleware for Real-Time Control

## Quality of Service (QoS) in ROS 2

Quality of Service (QoS) settings allow you to configure how messages are delivered between publishers and subscribers. This is crucial for real-time control applications where timing and reliability are essential.

### QoS Policies

ROS 2 provides several QoS policies that can be configured:

#### History Policy
Controls how many messages are stored:

```python
from rclpy.qos import QoSProfile, HistoryPolicy

# Keep only the last N messages
qos_profile = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10  # Keep last 10 messages
)

# Keep all messages (be careful with memory!)
qos_keep_all = QoSProfile(
    history=HistoryPolicy.KEEP_ALL
)
```

#### Reliability Policy
Controls message delivery guarantees:

```python
from rclpy.qos import ReliabilityPolicy

# Reliable delivery (like TCP)
reliable_qos = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE
)

# Best effort (like UDP, faster but may lose messages)
best_effort_qos = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
    reliability=ReliabilityPolicy.BEST_EFFORT
)
```

#### Durability Policy
Controls message persistence:

```python
from rclpy.qos import DurabilityPolicy

# Volatile - new subscribers won't receive old messages
volatile_qos = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE
)

# Transient local - new subscribers receive latest message
transient_qos = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL
)
```

### QoS for Different Use Cases

#### Sensor Data (High Volume, Can Lose Samples)
```python
sensor_qos = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)

# Use for camera feeds, LiDAR scans, IMU data
camera_pub = self.create_publisher(Image, 'camera/image_raw', sensor_qos)
```

#### Control Commands (Must Be Reliable)
```python
control_qos = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)

# Use for velocity commands, joint positions
cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', control_qos)
```

#### Configuration Data (Persistent)
```python
config_qos = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)

# Use for robot configuration, calibration data
config_pub = self.create_publisher(String, 'robot_configuration', config_qos)
```

## Real-Time Considerations

### Timing Requirements

Humanoid robots have strict timing requirements:

- **Balance Control**: 200-1000 Hz (2-5 ms)
- **Motor Control**: 100-500 Hz (2-10 ms)
- **Perception**: 30-60 Hz (16-33 ms)
- **Planning**: 10-20 Hz (50-100 ms)

### Publisher-Subscriber Timing

```python
#!/usr/bin/env python3
"""
Example: Real-time publisher with precise timing.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
import time


class RealTimeController(Node):
    """
    Demonstrates real-time control with proper QoS and timing.
    """

    def __init__(self):
        super().__init__('real_time_controller')

        # High-frequency control publisher
        control_qos = QoSProfile(
            history=ReliabilityPolicy.RELIABLE,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=QosDurabilityPolicy.VOLATILE
        )

        self.joint_pub = self.create_publisher(
            JointState,
            'joint_commands',
            control_qos
        )

        # Timer for 200Hz control loop (5ms period)
        self.control_timer = self.create_timer(
            0.005,  # 5ms = 200Hz
            self.control_callback
        )

        # Track timing
        self.last_time = self.get_clock().now()
        self.get_logger().info('Real-time controller started at 200Hz')

    def control_callback(self):
        """Real-time control loop with timing analysis."""
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time

        # Check timing performance
        expected_dt = 0.005  # 5ms
        timing_error = abs(dt - expected_dt)

        if timing_error > 0.001:  # 1ms tolerance
            self.get_logger().warn(f'Timing violation: {dt*1000:.1f}ms (expected {expected_dt*1000:.1f}ms)')

        # Generate control commands
        cmd = self.generate_control_commands()
        self.joint_pub.publish(cmd)

    def generate_control_commands(self):
        """Generate joint commands for humanoid control."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['left_hip_pitch', 'left_knee', 'right_hip_pitch', 'right_knee']

        # Example PD controller for balance
        import math
        t = self.get_clock().now().nanoseconds / 1e9

        positions = [
            0.1 * math.sin(2 * math.pi * 0.5 * t),  # Left hip
            -0.2 * math.sin(2 * math.pi * 0.5 * t), # Left knee
            0.1 * math.sin(2 * math.pi * 0.5 * t),  # Right hip
            -0.2 * math.sin(2 * math.pi * 0.5 * t)  # Right knee
        ]

        msg.position = positions
        msg.velocity = [0.0] * len(positions)
        msg.effort = [0.0] * len(positions)

        return msg
```

### Threading and Concurrency

```python
#!/usr/bin/env python3
"""
Example: Multi-threaded node for real-time performance.
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import Twist


class MultiThreadedRobotController(Node):
    """
    Demonstrates multi-threaded ROS 2 node for real-time performance.
    """

    def __init__(self):
        super().__init__('multithreaded_controller')

        # Create callback groups for different priorities
        self.high_priority_group = MutuallyExclusiveCallbackGroup()
        self.low_priority_group = MutuallyExclusiveCallbackGroup()

        # High-priority: Balance control (IMU data)
        self.imu_sub = self.create_subscription(
            Imu,
            'imu/data',
            self.imu_callback,
            10,
            callback_group=self.high_priority_group
        )

        # Low-priority: Navigation planning
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10,
            callback_group=self.low_priority_group
        )

        # Publishers
        self.joint_pub = self.create_publisher(
            JointState,
            'joint_commands',
            1
        )

        # High-frequency control loop
        self.control_timer = self.create_timer(
            0.005,  # 200Hz
            self.balance_control,
            callback_group=self.high_priority_group
        )

        self.get_logger().info('Multi-threaded controller initialized')

    def imu_callback(self, msg):
        """High-priority IMU callback for balance."""
        # Process IMU data for balance control
        self.current_orientation = msg.orientation
        self.angular_velocity = msg.angular_velocity
        self.linear_acceleration = msg.linear_acceleration

    def cmd_vel_callback(self, msg):
        """Low-priority navigation command."""
        # Store navigation command for later use
        self.desired_velocity = msg

    def balance_control(self):
        """High-priority balance control loop."""
        # Implement PD controller for balance
        # This runs at 200Hz regardless of other callbacks
        joint_commands = self.compute_balance_commands()
        self.joint_pub.publish(joint_commands)


def main(args=None):
    rclpy.init(args=args)

    # Use multi-threaded executor
    executor = MultiThreadedExecutor(num_threads=4)

    node = MultiThreadedRobotController()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
```

## Performance Optimization

### Efficient Message Handling

```python
#!/usr/bin/env python3
"""
Example: Efficient message handling for real-time performance.
"""

import rclpy
from rclpy.node import Node
import numpy as np


class EfficientMessageHandler(Node):
    """
    Demonstrates efficient message handling techniques.
    """

    def __init__(self):
        super().__init__('efficient_handler')

        # Pre-allocate message objects to avoid allocation overhead
        self.preallocated_joint_msg = JointState()
        self.joint_buffer = [0.0] * 18  # For 18 DOF humanoid
        self.position_array = np.zeros(18, dtype=np.float64)

        # Subscribe to high-frequency sensor data
        self.lidar_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            1  # Minimal queue depth for real-time
        )

        self.get_logger().info('Efficient message handler started')

    def scan_callback(self, msg):
        """Efficiently process laser scan data."""
        # Use numpy for efficient array operations
        ranges = np.array(msg.ranges, dtype=np.float32)

        # Filter out invalid ranges
        valid_ranges = ranges[np.isfinite(ranges)]

        # Compute obstacle distances efficiently
        if len(valid_ranges) > 0:
            min_distance = np.min(valid_ranges)

            # Publish control command based on obstacle detection
            if min_distance < 0.5:  # 50cm threshold
                self.apply_emergency_stop()

    def apply_emergency_stop(self):
        """Apply emergency stop with minimal delay."""
        # Zero out joint commands immediately
        self.joint_buffer[:] = [0.0] * len(self.joint_buffer)

        # Publish emergency stop command
        self.publish_joint_commands(self.joint_buffer)
```

### Memory Management

```python
#!/usr/bin/env python3
"""
Example: Memory-efficient real-time control.
"""

import rclpy
from rclpy.node import Node
from collections import deque
import gc


class MemoryEfficientController(Node):
    """
    Demonstrates memory-efficient real-time control.
    """

    def __init__(self):
        super().__init__('memory_efficient_controller')

        # Use deques for efficient append/pop operations
        self.command_history = deque(maxlen=100)  # Fixed-size history
        self.sensor_history = deque(maxlen=50)

        # Pre-allocate reusable objects
        self.joint_msg_template = JointState()
        self.tf_msg_template = TransformStamped()

        # Real-time timer
        self.rt_timer = self.create_timer(0.005, self.rt_callback)

        # Memory monitoring
        self.memory_check_timer = self.create_timer(1.0, self.check_memory)

    def rt_callback(self):
        """Real-time callback with minimal allocations."""
        # Process control logic
        commands = self.compute_control_output()

        # Reuse pre-allocated message
        self.joint_msg_template.header.stamp = self.get_clock().now().to_msg()
        self.joint_msg_template.position = commands

        # Publish without creating new objects
        self.joint_pub.publish(self.joint_msg_template)

        # Add to history for debugging
        self.command_history.append(commands)

    def check_memory(self):
        """Monitor memory usage."""
        if len(self.command_history) > 90:  # Near capacity
            self.get_logger().warn('Command history near capacity')
```

## Advanced QoS Patterns

### Deadline and Lifespan

```python
#!/usr/bin/env python3
"""
Example: Advanced QoS patterns with deadline and lifespan.
"""

from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy
from rclpy.duration import Duration


class AdvancedQoSNode(Node):
    """
    Demonstrates advanced QoS patterns.
    """

    def __init__(self):
        super().__init__('advanced_qos_node')

        # Deadline QoS: Messages must be delivered within 100ms
        deadline_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            deadline=Duration(seconds=0, nanoseconds=100000000)  # 100ms
        )

        # Lifespan QoS: Messages expire after 500ms
        lifespan_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            lifespan=Duration(seconds=0, nanoseconds=500000000)  # 500ms
        )

        self.deadline_pub = self.create_publisher(
            Twist,
            'deadline_cmd_vel',
            deadline_qos
        )

        self.lifespan_pub = self.create_publisher(
            JointState,
            'lifespan_joint_commands',
            lifespan_qos
        )
```

## Exercise: Implement Real-Time Humanoid Controller

Create a real-time controller node that:
1. Subscribes to IMU data at 500Hz with reliable QoS
2. Publishes joint commands at 200Hz with reliable QoS
3. Uses multi-threading for different control priorities
4. Implements efficient memory management
5. Applies proper timing analysis

## Learning Outcomes

After completing this section, students will be able to:
1. Configure Quality of Service settings for different use cases
2. Implement real-time control loops with proper timing
3. Use multi-threading for performance optimization
4. Apply memory-efficient message handling techniques
5. Design QoS policies for safety-critical systems

## Next Steps

Complete Module 1 by reviewing all concepts and practicing with the exercises. Then continue to [Module 2: The Digital Twin (Gazebo & Unity)](../module-2-simulation/intro.md) to learn about simulation environments.