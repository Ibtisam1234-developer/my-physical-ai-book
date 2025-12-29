---
slug: /weekly-breakdown/week-1
title: "Week 1: Physical AI اور ROS 2 Fundamentals کا تعارف"
hide_table_of_contents: false
---

# Week 1: Physical AI اور ROS 2 Fundamentals کا تعارف (ہفتہ 1)

## جائزہ (Overview)
اس ہفتے میں Physical AI کے بنیادی تصورات متعارف کروائے جائیں گے اور ROS 2 development environment قائم کیا جائے گا۔ Students ROS 2 Humble Hawksbill use کرتے ہوئے robot software development کی بنیادیں سیکھیں گے۔

## سیکھنے کے اہداف (Learning Objectives)
اس ہفتے کے آخر تک، students یہ کر سکیں گے:
- Physical AI کے اصولوں اور embodied intelligence کو explain کریں
- ROS 2 development environment setup اور configure کریں
- Python use کرتے ہوئے basic ROS 2 nodes create اور run کریں
- Publish-subscribe communication pattern سمجھیں
- Simple robot control commands implement کریں

## Day 1: Physical AI کے تصورات (Physical AI Concepts)
### مضامین کا احاطہ (Topics Covered)
- Physical AI کا تعارف اور Digital AI سے فرق
- Embodied cognition کے اصول
- Humanoid robotics کے ایپلیکیشنز
- Course کا جائزہ اور توقعات

### سرگرمیاں (Activities)
- Physical AI پر introductory videos دیکھیں
- Embodied intelligence پر foundational papers پڑھیں
- Development environment checklist setup کریں

### وسائل (Resources)
- [Physical AI کیا ہے؟](../intro/what-is-physical-ai.md)
- [Embodied Intelligence کے اصول](../intro/embodied-intelligence.md)
- [Course Syllabus](../intro/course-overview.md)

## Day 2: ROS 2 Installation اور Setup
### مضامین کا احاطہ (Topics Covered)
- Ubuntu 22.04 LTS installation/setup
- ROS 2 Humble Hawksbill installation
- Workspace creation اور management
- Basic ROS 2 commands اور tools

### Hands-on Activities
- ROS 2 Humble install کریں
- Catkin workspace create کریں
- Basic ROS 2 tutorials run کریں
- Talker/listener example سے installation verify کریں

### Code Tasks
```bash
# ROS 2 Humble install کریں
sudo apt update
sudo apt install ros-humble-desktop
sudo apt install python3-rosdep2 python3-rosinstall python3-rosinstall-generator python3-wstool build-essential

# Rosdep initialize کریں
sudo rosdep init
rosdep update

# ROS 2 source کریں
source /opt/ros/humble/setup.bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

## Day 3: ROS 2 Nodes اور Topics
### مضامین کا احاطہ (Topics Covered)
- Node architecture اور lifecycle
- Publisher-subscriber pattern
- Message types اور custom messages
- Quality of Service (QoS) settings

### Hands-on Activities
- Python میں first ROS 2 node create کریں
- Publisher اور subscriber implement کریں
- Nodes کے درمی communication test کریں
- ROS 2 command-line tools (ros2 topic, ros2 node) use کریں

### Code Tasks
```python
# Simple publisher node create کریں
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

## Day 4: Services اور Actions
### مضامین کا احاطہ (Topics Covered)
- Service-based communication (request-response)
- Action-based communication (long-running tasks with feedback)
- Topics vs services vs actions کب use کریں
- Error handling اور timeouts

### Hands-on Activities
- ROS 2 service server اور client implement کریں
- Action server اور client create کریں
- Error handling scenarios test کریں
- Communication patterns compare کریں

## Day 5: Robot Control Basics
### مضامین کا احاطہ (Topics Covered)
- Robot Operating System کے تصورات
- Joint control اور motor commands
- Basic robot movement commands
- Robot control میں safety considerations

### Hands-on Activities
- Simulated joints control کریں
- Basic movement commands implement کریں
- Safety limits اور boundaries test کریں
- Simple robot dance routine create کریں

## Assessment (تقييم)
- ROS 2 installation checklist complete کریں
- Publisher-subscriber example successfully run کریں
- Robot control کے لیے custom service create کریں
- Environment setup process document کریں

## اگلے ہفتے کا پیش نظر (Next Week Preview)
Week 2 میں simulation environments پر focus ہوگا، Gazebo اور Unity کو robot simulation کے لیے introduce کرتے ہوئے۔ Students virtual environments create کرنا سیکھیں گے جہاں robot behaviors کو safely test کیا جا سکے۔
