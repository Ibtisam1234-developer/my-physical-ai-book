---
slug: /module-1-ros2/intro
title: "ROS 2 کا تعارف"
hide_table_of_contents: false
---

# ROS 2 کا تعارف

## ROS 2: روبوٹکس کا nervous system

ROS 2 (Robot Operating System 2) روبوٹکس software development کے لیے industry-standard framework ہے۔ یہ middleware فراہم کرتا ہے جو robot system میں different software components کے درمی communication enable کرتا ہے۔

### اہم تصورات

- **Nodes**: Individual software modules جو specific tasks perform کرتے ہیں
- **Topics**: Publish-subscribe communication continuous data streams کے لیے
- **Services**: Request-response communication discrete operations کے لیے
- **Actions**: Long-running tasks feedback اور preemption کے ساتھ

### انسان نما روبوٹکس کے لیے ROS 2

- **Real-time capabilities**: Control systems کے لیے deterministic timing
- **Security**: Safe operation کے لیے DDS security framework
- **Lifecycle management**: Controlled node startup/shutdown
- **Quality of Service**: Configurable message delivery guarantees
- **Multi-robot support**: Multiple robots کے لیے native support

## تنصیب اور سیٹ اپ

```bash
# Locale set کریں
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

# Sources setup کریں
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# ROS 2 Humble install کریں
sudo apt update
sudo apt install ros-humble-desktop

# Environment setup کریں
source /opt/ros/humble/setup.bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

## سیکھنے کے نتائج

اس ماڈیول کو مکمل کرنے کے بعد، طالب علم یہ کر سکیں گے:
1. ROS 2 architecture اور core concepts سمجھ سکیں گے
2. rclpy استعمال کرتے ہوئے ROS 2 nodes create اور configure کر سکیں گے
3. Publisher-subscriber communication implement کر سکیں گے
4. Service servers اور clients create کر سکیں گے
5. Actions use کر سکیں گے
6. Proper structure کے ساتھ ROS 2 packages build کر سکیں گے

## شرائط

- بنیادی Python programming experience
- بنیادی robotics concepts کی سمجھ
- Command-line interfaces کا familiarity

## تخمینی وقت

3 ہفتے (کورس کے ہفتے 3-5)

## اگلے steps

[Nodes, Topics, Services](./nodes-topics-services.md) پڑھیں تاکہ ROS 2 میں fundamental communication patterns سیکھ سکیں۔
