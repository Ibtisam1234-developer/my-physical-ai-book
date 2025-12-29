---
slug: /module-2-simulation/sensor-simulation
title: "Sensor Simulation"
hide_table_of_contents: false
---

# Sensor Simulation

Virtual sensors real-world sensor data generate کرتے ہیں جو robot perception کے لیے use ہوتا ہے۔

## Camera Simulation

### RGB Camera

```xml
<sensor name="camera" type="camera">
  <camera name="rgb_camera">
    <horizontal_fov>1.396</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>100</far>
    </clip>
  </camera>
  <always_on>true</always_on>
  <update_rate>30</update_rate>
</sensor>
```

### Depth Camera

```xml
<sensor name="depth_camera" type="depth">
  <camera name="depth">
    <horizontal_fov>1.396</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
    </image>
    <clip>
      <near>0.1</near>
      <far>10</far>
    </clip>
  </camera>
</sensor>
```

## LIDAR Simulation

### 2D LIDAR

```xml
<sensor name="lidar" type="ray">
  <ray>
    <scan>
      <horizontal>
        <samples>360</samples>
        <resolution>1</resolution>
        <min_angle>0</min_angle>
        <max_angle>6.28318</max_angle>
      </horizontal>
    </scan>
    <range>
      <min>0.1</min>
      <max>10</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
</sensor>
```

### 3D LIDAR

```xml
<sensor name="lidar3d" type="ray">
  <ray>
    <scan>
      <horizontal>
        <samples>360</samples>
        <resolution>1</resolution>
      </horizontal>
      <vertical>
        <samples>32</samples>
        <resolution>1</resolution>
        <min_angle>-0.5</min_angle>
        <max_angle>0.5</max_angle>
      </vertical>
    </scan>
    <range>
      <min>0.1</min>
      <max>100</max>
    </range>
  </ray>
</sensor>
```

## IMU Simulation

```xml
<sensor name="imu" type="imu">
  <imu>
    <linear_acceleration>
      <noise type="gaussian">
        <mean>0.0</mean>
        <stddev>0.01</stddev>
      </noise>
    </linear_acceleration>
    <angular_velocity>
      <noise type="gaussian">
        <mean>0.0</mean>
        <stddev>0.001</stddev>
      </noise>
    </angular_velocity>
  </imu>
</sensor>
```

## Sensor Plugins

### Custom Sensor Plugin

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

class CustomSensor(Node):
    def __init__(self):
        super().__init__('custom_sensor')
        self.publisher = self.create_publisher(Image, 'sensor_data', 10)
        self.timer = self.create_timer(0.033, self.publish_data)

    def publish_data(self):
        msg = Image()
        # Generate sensor data
        self.publisher.publish(msg)
```

## اگلے steps

[unity-rendering.md](./unity-rendering.md) پڑھیں تاکہ Unity rendering کے بارے میں جان سکیں۔
