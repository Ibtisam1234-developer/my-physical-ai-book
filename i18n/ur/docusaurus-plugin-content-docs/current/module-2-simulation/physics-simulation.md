---
slug: /module-2-simulation/physics-simulation
title: "Physics Simulation"
hide_table_of_contents: false
---

# Physics Simulation

Physics simulation robot dynamics اور interactions کو accurately model کرتا ہے۔

## Physics Engines

### ODE (Open Dynamics Engine)
ODE basic physics simulation کے لیے use ہوتا ہے۔

### Bullet Physics
Bullet real-time physics simulation کے لیے use ہوتا ہے۔

### Dart Physics
DART (Dynamic Animation and Robotics Toolkit) complex robot dynamics کے لیے use ہوتا ہے۔

## Physics in Gazebo

### World File Example

```xml
<sdf version="1.6">
  <world name="physics_world">
    <physics type="ode">
      <ode>
        <solver>
          <type>quick</type>
          <iters>100</iters>
          <sor>1.0</sor>
        </solver>
        <constraints>
          <cfm>0.0</cfm>
          <erp>0.2</erp>
        </constraints>
      </ode>
    </physics>
  </world>
</sdf>
```

### Gravity Settings

```xml
<gravity>0 0 -9.8</gravity>
```

### Ground Plane

```xml
<model name="ground">
  <static>true</static>
  <link name="link">
    <collision name="collision">
      <geometry>
        <plane/>
      </geometry>
    </collision>
  </link>
</model>
```

## Robot Dynamics

### Joint Dynamics

```xml
<joint name="arm_joint" type="revolute">
  <pose>0 0 0 0 0 0</pose>
  <parent link="torso"/>
  <child link="upper_arm"/>
  <axis>
    <xyz>0 0 1</xyz>
    <dynamics>
      <damping>0.1</damping>
      <friction>0.0</friction>
    </dynamics>
  </axis>
  <limit>
    <lower>-3.14</lower>
    <upper>3.14</upper>
    <effort>100</effort>
    <velocity>5.0</velocity>
  </limit>
</joint>
```

## Collision Detection

### Basic Collision

```xml
<link name="robot_link">
  <collision name="collision">
    <geometry>
      <box size="0.5 0.5 1.0"/>
    </geometry>
    <surface>
      <friction>
        <ode>
          <mu>0.9</mu>
          <mu2>0.9</mu2>
        </ode>
      </friction>
    </surface>
  </collision>
</link>
```

### Collision Filtering

```xml
<gazebo>
  <plugin name="collision_filter" filename="libcollision_filter.so">
    <collision>link1 link2</collision>
  </plugin>
</gazebo>
```

## اگلے steps

[sensor-simulation.md](./sensor-simulation.md) پڑھیں تاکہ virtual sensors کے بارے میں جان سکیں۔
