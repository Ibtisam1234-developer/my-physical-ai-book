# Nodes, Topics, Services

## Understanding ROS 2 Nodes

A ROS 2 node is a fundamental software component that performs a single, modular task. Nodes communicate with each other to accomplish complex robot behaviors.

### Node Characteristics

- **Modularity**: Each node has a single responsibility
- **Reusability**: Nodes can be reused across different robots
- **Composability**: Multiple nodes combine to form complete systems
- **Distribution**: Nodes can run on different computers

### Anatomy of a ROS 2 Node

```python
#!/usr/bin/env python3
"""
Example: A simple ROS 2 node that publishes temperature readings.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
import random


class TemperatureSensorNode(Node):
    """
    A node that simulates a temperature sensor and publishes readings.
    Demonstrates basic node structure and publishing.
    """

    def __init__(self):
        # Call parent class constructor with node name
        super().__init__('temperature_sensor')

        # Create a publisher for temperature readings
        self.publisher_ = self.create_publisher(
            Float64,
            'temperature',
            10  # QoS depth
        )

        # Create a timer for periodic publishing
        self.timer = self.create_timer(1.0, self.publish_temperature)

        # Log node initialization
        self.get_logger().info('Temperature Sensor Node has started')

        # Track publication count
        self.publication_count = 0

    def publish_temperature(self):
        """Callback function to publish temperature readings."""
        # Simulate temperature reading (in Celsius)
        temperature = 20.0 + random.gauss(0, 0.5)

        # Create message
        msg = Float64()
        msg.data = temperature

        # Publish message
        self.publisher_.publish(msg)
        self.publication_count += 1

        # Log the publication
        self.get_logger().debug(f'Published temperature: {temperature:.2f}°C')


def main(args=None):
    """Entry point for the temperature sensor node."""
    # Initialize ROS 2 client library
    rclpy.init(args=args)

    # Create the node
    node = TemperatureSensorNode()

    # Keep the node alive (spin)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Topics: Publisher-Subscriber Communication

Topics implement a publish-subscribe pattern where:
- **Publishers**: Send messages to a topic without knowing subscribers
- **Subscribers**: Receive messages from a topic without knowing publishers
- **Decoupling**: Publishers and subscribers are completely independent

### Publishing to Topics

```python
#!/usr/bin/env python3
"""
Example: Publisher node for robot velocity commands.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3
import math


class VelocityPublisher(Node):
    """
    Publishes velocity commands for robot motion control.
    Demonstrates Topic publishing with standard message types.
    """

    def __init__(self):
        super().__init__('velocity_publisher')

        # Create publisher for velocity commands
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            'cmd_vel',
            10
        )

        # Create timer for periodic velocity commands
        self.timer = self.create_timer(0.1, self.publish_velocity)

        # Track time for sinusoidal motion
        self.start_time = self.get_clock().now()

        self.get_logger().info('Velocity Publisher started')

    def publish_velocity(self):
        """Publish velocity commands for circular motion."""
        # Calculate time
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9

        # Create velocity message
        cmd_vel = Twist()

        # Circular motion parameters
        linear_speed = 0.2  # m/s
        angular_speed = 0.5  # rad/s

        # Sinusoidal velocity profile
        cmd_vel.linear = Vector3(
            x=linear_speed * math.sin(elapsed),
            y=0.0,
            z=0.0
        )
        cmd_vel.angular = Vector3(
            x=0.0,
            y=0.0,
            z=angular_speed * math.cos(elapsed)
        )

        # Publish
        self.cmd_vel_publisher.publish(cmd_vel)

        # Log every second
        if int(elapsed) > int(elapsed - 0.1):
            self.get_logger().debug(f'Publishing: linear.x={cmd_vel.linear.x:.2f}, angular.z={cmd_vel.angular.z:.2f}')


class TwistSubscriber(Node):
    """
    Subscribes to velocity commands and logs them.
    Demonstrates Topic subscription with callbacks.
    """

    def __init__(self):
        super().__init__('twist_subscriber')

        # Create subscription
        self.subscription = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10
        )

        self.get_logger().info('Twist Subscriber started')

    def cmd_vel_callback(self, msg: Twist):
        """Callback function for velocity commands."""
        self.get_logger().info(
            f'Received velocity: linear=({msg.linear.x:.2f}, {msg.linear.y:.2f}, {msg.linear.z:.2f}), '
            f'angular=({msg.angular.x:.2f}, {msg.angular.y:.2f}, {msg.angular.z:.2f})'
        )
```

## Services: Request-Response Communication

Services implement a request-response pattern for synchronous operations:
- **Service Server**: Handles requests and returns responses
- **Service Client**: Sends requests and waits for responses
- **Use Cases**: Getting robot state, triggering actions, configuration

### Creating a Service Server

```python
#!/usr/bin/env python3
"""
Example: Service server for robot control.
"""

from enum import IntEnum
import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
from std_srvs.srv import Trigger


class RobotController(Node):
    """
    Service server for robot control operations.
    Demonstrates both standard and custom service types.
    """

    class RobotMode(IntEnum):
        IDLE = 0
        MOVING = 1
        CALIBRATING = 2
        ERROR = 3

    def __init__(self):
        super().__init__('robot_controller')

        # Initialize robot state
        self.is_enabled = False
        self.current_mode = self.RobotMode.IDLE

        # Create service servers
        self.enable_srv = self.create_service(
            SetBool,
            'robot/enable',
            self.enable_callback
        )

        self.home_srv = self.create_service(
            Trigger,
            'robot/home',
            self.home_callback
        )

        self.get_logger().info('Robot Controller services available')

    def enable_callback(self, request: SetBool.Request, response: SetBool.Response):
        """Enable or disable the robot."""
        self.is_enabled = request.data

        if self.is_enabled:
            self.current_mode = self.RobotMode.IDLE
            response.success = True
            response.message = 'Robot enabled'
            self.get_logger().info('Robot enabled')
        else:
            self.current_mode = self.RobotMode.ERROR
            response.success = True
            response.message = 'Robot disabled'
            self.get_logger().info('Robot disabled')

        return response

    def home_callback(self, request: Trigger.Request, response: Trigger.Response):
        """Move robot to home position."""
        if not self.is_enabled:
            response.success = False
            response.message = 'Robot not enabled'
            return response

        self.get_logger().info('Homing robot...')

        # Simulate homing operation
        import time
        time.sleep(1.0)

        self.current_mode = self.RobotMode.IDLE
        response.success = True
        response.message = 'Robot homed successfully'

        return response
```

## Actions: Long-Running Tasks with Feedback

Actions are designed for long-running tasks:
- **Goal**: The task to accomplish
- **Feedback**: Progress updates during execution
- **Result**: Final outcome when task completes
- **Preemption**: Cancel a running goal

### Creating an Action Server

```python
#!/usr/bin/env python3
"""
Example: Action server for robot navigation.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.action import CancelPolicy
from rclpy.action import GoalResponse
from geometry_msgs.msg import Pose
from nav2_msgs.action import NavigateToPose


class NavigationActionServer(Node):
    """
    Action server for robot navigation using NavigateToPose action.
    Demonstrates action server implementation with feedback and preemption.
    """

    def __init__(self):
        super().__init__('navigation_action_server')

        # Create action server
        self._action_server = ActionServer(
            self,
            NavigateToPose,
            'navigate_to_pose',
            self.execute_callback,
            cancel_policy=CancelPolicy.CancelGoal,
            goal_callback=self.goal_callback,
        )

        self.current_goal = None
        self.cancel_requested = False

        self.get_logger().info('Navigation Action Server started')

    def goal_callback(self, goal_request):
        """Accept or reject goals based on robot state."""
        if self.current_goal is not None:
            self.get_logger().warn('Already executing a goal, rejecting new goal')
            return GoalResponse.REJECT

        self.get_logger().info(f'Received goal: {goal_request.pose.header.frame_id}')
        return GoalResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """Execute the navigation goal with feedback."""
        self.current_goal = goal_handle
        self.cancel_requested = False

        goal_pose = goal_handle.request.pose
        feedback_msg = NavigateToPose.Feedback()
        result_msg = NavigateToPose.Result()

        self.get_logger().info(f'Navigating to: ({goal_pose.pose.position.x:.2f}, {goal_pose.pose.position.y:.2f})')

        # Simulate navigation with periodic feedback
        start_position = Pose()
        start_position.position.x = 0.0
        start_position.position.y = 0.0

        target_x = goal_pose.pose.position.x
        target_y = goal_pose.pose.position.y
        distance_total = ((target_x - start_position.position.x)**2 +
                         (target_y - start_position.position.y)**2)**0.5

        # Simulate movement in steps
        num_steps = 20
        for i in range(num_steps):
            # Check for cancellation
            if self.cancel_requested:
                result_msg.pose.header.frame_id = 'cancelled'
                goal_handle.canceled()
                self.get_logger().info('Goal cancelled')
                self.current_goal = None
                self.cancel_requested = False
                return result_msg

            # Check if goal was aborted
            if goal_handle.status == 'ABORTED':
                self.get_logger().error('Goal aborted')
                result_msg.pose.header.frame_id = 'aborted'
                return result_msg

            # Update progress
            progress = (i + 1) / num_steps
            current_x = start_position.position.x + progress * (target_x - start_position.position.x)
            current_y = start_position.position.y + progress * (target_y - start_position.position.y)

            feedback_msg.current_pose.header.frame_id = 'map'
            feedback_msg.current_pose.pose.position.x = current_x
            feedback_msg.current_pose.pose.position.y = current_y
            feedback_msg.distance_remaining = distance_total * (1 - progress)

            # Publish feedback
            goal_handle.publish_feedback(feedback_msg)

            self.get_logger().debug(f'Progress: {progress*100:.1f}%, Distance remaining: {feedback_msg.distance_remaining:.2f}m')

            # Simulate computation time
            import time
            time.sleep(0.5)

        # Goal completed
        goal_handle.succeed()
        result_msg.pose = goal_pose
        self.get_logger().info('Navigation completed successfully')

        self.current_goal = None
        return result_msg
```

## Quality of Service (QoS) Settings

QoS policies control how messages are delivered:

### Available QoS Policies

| Policy | Options | Use Case |
|--------|---------|----------|
| History | Keep last, Keep all | Memory vs. completeness |
| Depth | 1-N (for Keep last) | Buffer size |
| Reliability | Reliable, Best effort | Guaranteed vs. speed |
| Durability | Transient local, Volatile | Persistence |
| Deadline | Duration or none | Timing guarantees |
| Lifespan | Duration or none | Message expiration |

## Exercise: Create a Complete Robot Interface

Create a complete robot control interface with:
1. Node for joint state publishing
2. Service server for enabling/disabling the robot
3. Publisher for velocity commands
4. Action server for navigation

## Learning Outcomes

After completing this section, students will be able to:
1. Create and configure ROS 2 nodes
2. Implement publisher-subscriber communication
3. Create service servers and clients
4. Use actions for long-running tasks
5. Configure Quality of Service settings

## Next Steps

Continue to [Python-ROS Integration](./python-ros-integration.md) to learn how to use rclpy for Python-based ROS nodes.