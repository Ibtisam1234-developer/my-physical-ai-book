---
slug: /module-1-ros2/middleware-real-time-control
title: "Middleware اور Real-time Control"
hide_table_of_contents: false
---

# Middleware اور Real-time Control

ROS 2 کا middleware architecture اور real-time control capabilities کو سمجھنا important ہے۔

## ROS 2 Middleware (DDS)

ROS 2 DDS (Data Distribution Service) پر based ہے۔ یہ communication کے لیے underlying transport provide کرتا ہے۔

### DDS کے فوائد

- **Loose coupling**: Publishers اور subscribers کو ایک دوسرے کے بارے میں جاننے کی ضرورت نہیں
- **Scalability**: Large-scale systems کو handle کر سکتا ہے
- **Quality of Service (QoS)**: Configurable reliability اور latency settings
- **Security**: Built-in security features

### QoS Policies

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

qos_profile = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)
```

## Real-time Control

Humanoid robots کے لیے real-time control essential ہے۔ ROS 2 کئی features provide کرتا ہے۔

### Executor Types

```python
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor

# Single-threaded - simpler but slower
executor = SingleThreadedExecutor()

# Multi-threaded - better for complex systems
executor = MultiThreadedExecutor(num_threads=4)
```

### Timer-based Control

```python
class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node')
        self.control_period = 0.01  # 10ms
        self.timer = self.create_timer(self.control_period, self.control_loop)

    def control_loop(self):
        # Real-time control logic
        self.get_logger().info('Control step executed')
```

## Lifecycle Management

ROS 2 nodes کے لیے lifecycle management provide کرتا ہے۔

### Lifecycle Node

```python
from rclpy_lifecycle import LifecycleNode
from lifecycle_msgs.msg import Transition

class LifecycleControlNode(LifecycleNode):
    def __init__(self):
        super().__init__('lifecycle_control_node')

    def on_configure(self, state):
        self.get_logger().info('Configuring...')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self.get_logger().info('Activating...')
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        self.get_logger().info('Deactivating...')
        return TransitionCallbackReturn.SUCCESS
```

## اگلے steps

[python-ros-integration.md](./python-ros-integration.md) پڑھیں تاکہ Python کے ساتھ ROS 2 integration کے بارے میں جان سکیں۔