---
slug: /module-1-ros2/urdf-humanoid-robots
title: "URDF اور Humanoid Robots"
hide_table_of_contents: false
---

# URDF اور Humanoid Robots

URDF (Unified Robot Description Format) humanoid robots کو describe کرنے کے لیے use ہوتا ہے۔

## URDF کی بنیادی باتیں

URDF XML-based format ہے robot کی physical description کے لیے۔

### Basic URDF Structure

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot">
  <!-- Links -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.3 0.3 0.5"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.3 0.5"/>
      </geometry>
    </collision>
  </link>

  <!-- Joint -->
  <joint name="neck_joint" type="revolute">
    <parent link="base_link"/>
    <child link="head_link"/>
    <origin xyz="0 0 0.5"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57"/>
  </joint>
</robot>
```

## Humanoid Robot URDF

Humanoid robots کے لیے complex URDF structure کی ضرورت ہوتا ہے۔

### Multi-body Structure

```xml
<!-- Humanoid Robot URDF Example -->
<robot name="humanoid">
  <!-- Torso -->
  <link name="torso"/>
  <joint name="torso_to_neck" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
  </joint>

  <!-- Arms -->
  <link name="left_upper_arm"/>
  <link name="left_lower_arm"/>
  <joint name="left_shoulder" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_arm"/>
  </joint>

  <!-- Legs -->
  <link name="left_upper_leg"/>
  <link name="left_lower_leg"/>
  <joint name="left_hip" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_leg"/>
  </joint>
</robot>
```

### Physical Properties

```xml
<link name="torso">
  <inertial>
    <mass value="10.0"/>
    <origin xyz="0 0 0.3"/>
    <inertia ixx="0.5" iyy="0.5" izz="0.5"/>
  </inertial>
</link>
```

## Xacro Use کرنا

Complex URDF files کے لیے xacro use کریں۔

### Xacro Example

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://wiki.ros.org/xacro">
  <xacro:property name="mass" value="5.0"/>

  <link name="${name}_link">
    <visual>
      <geometry>
        <box size="${size}"/>
      </geometry>
    </visual>
  </link>
</robot>
```

### Xacro Command

```bash
xacro model.xacro > model.urdf
```

## Gazebo Integration

URDF کو Gazebo simulation میں use کرنے کے لیے Gazebo-specific tags add کریں۔

```xml
<robot name="humanoid">
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so"/>
  </gazebo>

  <link name="torso">
    <gazebo reference="torso">
      <material>Gazebo/Blue</material>
    </gazebo>
  </link>
</robot>
```

## اگلے steps

Module 1 کو مکمل کرنے کے بعد، [Module 2: Simulation Environments](../module-2-simulation/intro) پڑھیں۔
