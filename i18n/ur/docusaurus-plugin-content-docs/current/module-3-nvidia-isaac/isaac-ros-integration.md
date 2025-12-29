---
slug: /module-3-nvidia-isaac/isaac-ros-integration
title: "Isaac ROS Integration"
hide_table_of_contents: false
---

# Isaac ROS Integration

Isaac ROS کو ROS 2 کے ساتھ integrate کرنا سیکھیں۔

## Isaac ROS Packages

```bash
# Install Isaac ROS packages
sudo apt-get install ros-humble-isaac-ros-dev
```

### Common Packages

- `isaac_ros_image_pipeline`: GPU-accelerated image processing
- `isaac_ros_navigation`: Navigation stack
- `isaac_ros_manipulation`: Manipulation capabilities
- `isaac_ros_object_detection`: Object detection

## Bridge Setup

### Isaac-ROS Bridge

```python
from isaac_ros_bridge import Bridge

# Create bridge
bridge = Bridge(
    isaac_topic="/isaac/camera",
    ros_topic="/camera/image_raw",
    message_type="Image"
)
```

### Message Conversion

```python
from isaac_ros_messages.msg import Image

def convert_ros_to_isaac(ros_msg):
    isaac_msg = Image()
    isaac_msg.data = ros_msg.data
    return isaac_msg
```

## Navigation Integration

```python
import rclpy
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose

class NavigationNode(Node):
    def __init__(self):
        super().__init__('navigation_node')
        self.client = ActionClient(
            self,
            NavigateToPose,
            '/navigate_to_pose'
        )

    def send_goal(self, x, y, theta):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        self.client.wait_for_server()
        self.client.send_goal_async(goal_msg)
```

## اگلے steps

[synthetic-data-perception.md](./synthetic-data-perception.md) پڑھیں۔
