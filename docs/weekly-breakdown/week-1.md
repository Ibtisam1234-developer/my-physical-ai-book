# Week 1: Introduction to Physical AI & ROS 2 Fundamentals

## Overview
This week introduces the fundamental concepts of Physical AI and establishes the ROS 2 development environment. Students will learn the basics of robot software development using ROS 2 Humble Hawksbill.

## Learning Objectives
By the end of this week, students will be able to:
- Explain the principles of Physical AI and embodied intelligence
- Set up and configure a ROS 2 development environment
- Create and run basic ROS 2 nodes using Python
- Understand the publish-subscribe communication pattern
- Implement simple robot control commands

## Day 1: Physical AI Concepts
### Topics Covered
- Introduction to Physical AI vs. Digital AI
- Embodied cognition principles
- Humanoid robotics applications
- Course overview and expectations

### Activities
- Watch introductory videos on Physical AI
- Read foundational papers on embodied intelligence
- Set up development environment checklist

### Resources
- [What is Physical AI?](../intro/what-is-physical-ai.md)
- [Embodied Intelligence Principles](../intro/embodied-intelligence.md)
- [Course Syllabus](../intro/course-overview.md)

## Day 2: ROS 2 Installation and Setup
### Topics Covered
- Ubuntu 22.04 LTS installation/setup
- ROS 2 Humble Hawksbill installation
- Workspace creation and management
- Basic ROS 2 commands and tools

### Hands-on Activities
- Install ROS 2 Humble
- Create catkin workspace
- Run basic ROS 2 tutorials
- Verify installation with talker/listener example

### Code Tasks
```bash
# Install ROS 2 Humble
sudo apt update
sudo apt install ros-humble-desktop
sudo apt install python3-rosdep2 python3-rosinstall python3-rosinstall-generator python3-wstool build-essential

# Initialize rosdep
sudo rosdep init
rosdep update

# Source ROS 2
source /opt/ros/humble/setup.bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

## Day 3: ROS 2 Nodes and Topics
### Topics Covered
- Node architecture and lifecycle
- Publisher-subscriber pattern
- Message types and custom messages
- Quality of Service (QoS) settings

### Hands-on Activities
- Create first ROS 2 node in Python
- Implement publisher and subscriber
- Test communication between nodes
- Use ROS 2 command-line tools (ros2 topic, ros2 node)

### Code Tasks
```python
# Create a simple publisher node
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Day 4: Services and Actions
### Topics Covered
- Service-based communication (request-response)
- Action-based communication (long-running tasks with feedback)
- When to use topics vs services vs actions
- Error handling and timeouts

### Hands-on Activities
- Implement ROS 2 service server and client
- Create action server and client
- Test error handling scenarios
- Compare communication patterns

## Day 5: Robot Control Basics
### Topics Covered
- Robot Operating System concepts
- Joint control and motor commands
- Basic robot movement commands
- Safety considerations in robot control

### Hands-on Activities
- Control simulated joints
- Implement basic movement commands
- Test safety limits and boundaries
- Create simple robot dance routine

## Assessment
- Complete ROS 2 installation checklist
- Successfully run publisher-subscriber example
- Create custom service for robot control
- Document environment setup process

## Next Week Preview
Week 2 will focus on simulation environments, introducing Gazebo and Unity for robot simulation. Students will learn to create virtual environments for testing robot behaviors safely.