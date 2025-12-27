# Introduction to ROS 2

## ROS 2: The Robotic Nervous System

ROS 2 (Robot Operating System 2) is the industry-standard framework for robot software development. It provides the middleware that enables communication between different software components in a robot system, acting as the "nervous system" that allows different parts of the robot to coordinate their actions.

### Key Concepts

- **Nodes**: Individual software modules that perform specific tasks
- **Topics**: Publish-subscribe communication for continuous data streams
- **Services**: Request-response communication for discrete operations
- **Actions**: Long-running tasks with feedback and preemption

### Why ROS 2 for Humanoids?

- **Real-time capabilities**: Deterministic timing for control systems
- **Security**: DDS security framework for safe operation
- **Lifecycle management**: Controlled node startup/shutdown
- **Quality of Service**: Configurable message delivery guarantees
- **Multi-robot support**: Native support for multiple robots

## Installation and Setup

```bash
# Set locale
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

# Setup sources
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2 Humble
sudo apt update
sudo apt install ros-humble-desktop

# Setup environment
source /opt/ros/humble/setup.bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

## Learning Outcomes

After completing this module, students will be able to:
1. Understand ROS 2 architecture and core concepts
2. Create and configure ROS 2 nodes using rclpy
3. Implement publisher-subscriber communication
4. Create service servers and clients
5. Use actions for long-running tasks
6. Build ROS 2 packages with proper structure

## Prerequisites

- Basic Python programming experience
- Understanding of basic robotics concepts
- Familiarity with command-line interfaces

## Estimated Time

3 weeks (Weeks 3-5 of the course)

## Next Steps

Continue to [Nodes, Topics, Services](./nodes-topics-services.md) to learn about the fundamental communication patterns in ROS 2.