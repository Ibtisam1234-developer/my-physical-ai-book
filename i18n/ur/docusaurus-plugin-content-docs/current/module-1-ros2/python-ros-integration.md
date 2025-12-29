---
slug: /module-1-ros2/python-ros-integration
title: "Python اور ROS Integration"
hide_table_of_contents: false
---

# Python اور ROS Integration

rclpy استعمال کرتے ہوئے ROS 2 کو Python کے ساتھ integrate کرنا سیکھیں۔

## rclpy کی بنیادی باتیں

rclpy ROS 2 کا Python client library ہے۔ یہ ROS 2 concepts کو Pythonic interface میں wrap کرتا ہے۔

### Environment Setup

```bash
# ROS 2 environment set کریں
source /opt/ros/humble/setup.bash

# Python environment create کریں
python3 -m venv ros2_env
source ros2_env/bin/activate

# ROS 2 Python packages install کریں
pip install rclpy
```

### Basic rclpy Node

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

class MinimalNode(Node):
    def __init__(self):
        super().__init__('minimal_node')
        self.get_logger().info('Minimal ROS 2 Node Started')

def main(args=None):
    rclpy.init(args=args)
    node = MinimalNode()
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

## Custom Messages

Python میں custom message types create کرنا۔

### Message Definition

```python
# custom_message.msg
float32[] positions
float32[] velocities
float32[] efforts
string joint_name
timestamp
```

### Custom Message Use کرنا

```python
from your_package.msg import CustomMessage

class MessageNode(Node):
    def __init__(self):
        super().__init__('message_node')
        self.publisher = self.create_publisher(CustomMessage, 'custom_topic', 10)

    def publish_message(self):
        msg = CustomMessage()
        msg.positions = [0.0, 0.0, 0.0]
        msg.velocities = [0.0, 0.0, 0.0]
        msg.joint_name = 'arm_joint'
        self.publisher.publish(msg)
```

## Action Clients اور Servers

Long-running tasks کے لیے actions use کریں۔

### Action Server

```python
from rclpy.action import ActionServer
from your_package.action import Trajectory

class TrajectoryActionServer(Node):
    def __init__(self):
        super().__init__('trajectory_action_server')
        self._action_server = ActionServer(
            self,
            Trajectory,
            'execute_trajectory',
            self.execute_callback
        )

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing trajectory...')
        # Execute trajectory steps
        for i in range(10):
            feedback = Trajectory.Feedback()
            feedback.progress = float(i) / 10.0
            goal_handle.publish_feedback(feedback)

        result = Trajectory.Result()
        result.success = True
        goal_handle.succeed(result)
        return result
```

### Action Client

```python
from rclpy.action import ActionClient
from your_package.action import Trajectory

class TrajectoryActionClient(Node):
    def __init__(self):
        super().__init__('trajectory_action_client')
        self._action_client = ActionClient(self, Trajectory, 'execute_trajectory')

    def send_goal(self, trajectory_points):
        goal_msg = Trajectory.Goal()
        goal_msg.points = trajectory_points
        self._action_client.wait_for_server()
        self._future = self._action_client.send_goal_async(goal_msg)
```

## Parameter Handling

ROS 2 parameters کے ساتھ کام کرنا۔

### Parameter Declaration اور Use

```python
class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters
        self.declare_parameter('robot_name', 'default_robot')
        self.declare_parameter('max_velocity', 1.0)

        # Get parameter values
        robot_name = self.get_parameter('robot_name').get_parameter_value()
        max_velocity = self.get_parameter('max_velocity').get_parameter_value()

        self.get_logger().info(f'Robot: {robot_name}, Max Velocity: {max_velocity}')
```

## اگلے steps

[urdf-humanoid-robots.md](./urdf-humanoid-robots.md) پڑھیں تاکہ URDF اور humanoid robots کے بارے میں جان سکیں۔
