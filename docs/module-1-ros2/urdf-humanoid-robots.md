# URDF for Humanoid Robots

## Introduction to URDF

URDF (Unified Robot Description Format) is an XML format for representing robot models in ROS. It describes the physical and visual properties of a robot, including its links, joints, and other components.

### URDF Components

- **Links**: Rigid bodies with inertial, visual, and collision properties
- **Joints**: Connections between links with specified degrees of freedom
- **Materials**: Visual appearance definitions
- **Transmission**: Actuator configurations for ROS control

## Basic URDF Structure

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot">
    <!-- MATERIALS -->
    <material name="white">
        <color rgba="1 1 1 1"/>
    </material>
    <material name="black">
        <color rgba="0.1 0.1 0.1 1"/>
    </material>
    <material name="metal">
        <color rgba="0.8 0.8 0.8 1"/>
    </material>

    <!-- BASE LINK -->
    <link name="base_link">
        <visual>
            <geometry>
                <box size="0.3 0.3 0.5"/>
            </geometry>
            <material name="white"/>
        </visual>
        <collision>
            <geometry>
                <box size="0.3 0.3 0.5"/>
            </geometry>
        </collision>
        <inertial>
            <mass value="5.0"/>
            <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
        </inertial>
    </link>
</robot>
```

## Creating a Humanoid Robot URDF

### Joint Definitions

```xml
<!-- REVOLUTE JOINT: Hip Pitch -->
<joint name="left_hip_pitch" type="revolute">
    <parent link="pelvis"/>
    <child link="left_upper_leg"/>
    <origin xyz="0 -0.1 0"/>
    <axis xyz="1 0 0"/>  <!-- Rotate around X axis -->
    <limit lower="-1.57" upper="1.57" effort="100" velocity="5"/>
    <dynamics damping="0.5"/>
</joint>

<!-- CONTINUOUS JOINT: Hip Yaw -->
<joint name="left_hip_yaw" type="continuous">
    <parent link="pelvis"/>
    <child link="left_upper_leg_yaw"/>
    <origin xyz="0 -0.1 0"/>
    <axis xyz="0 0 1"/>  <!-- Rotate around Z axis -->
    <dynamics damping="0.5"/>
</joint>

<!-- PRISMATIC JOINT: Linear Actuator -->
<joint name="waist_prism" type="prismatic">
    <parent link="base"/>
    <child link="waist"/>
    <origin xyz="0 0 0"/>
    <axis xyz="0 0 1"/>  <!-- Linear motion along Z -->
    <limit lower="0" upper="0.3" effort="500" velocity="0.1"/>
</joint>

<!-- FIXED JOINT: Non-moving connections -->
<joint name="sensor_mount" type="fixed">
    <parent link="head"/>
    <child link="camera_mount"/>
    <origin xyz="0.1 0 0.05"/>
</joint>
```

## Complete Humanoid URDF Example

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">

    <!-- ==================== MATERIALS ==================== -->
    <material name="body_color">
        <color rgba="0.8 0.8 0.8 1"/>
    </material>
    <material name="joint_color">
        <color rgba="0.2 0.2 0.2 1"/>
    </material>
    <material name="sensor_color">
        <color rgba="0.1 0.1 0.4 1"/>
    </material>

    <!-- ==================== BASE LINK ==================== -->
    <link name="base_link">
        <inertial>
            <mass value="2.0"/>
            <origin xyz="0 0 0"/>
            <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
        </inertial>
        <visual>
            <geometry>
                <cylinder length="0.1" radius="0.15"/>
            </geometry>
            <material name="joint_color"/>
        </visual>
        <collision>
            <geometry>
                <cylinder length="0.1" radius="0.15"/>
            </geometry>
        </collision>
    </link>

    <!-- ==================== PELVIS ==================== -->
    <link name="pelvis">
        <inertial>
            <mass value="5.0"/>
            <origin xyz="0 0 0.05"/>
            <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05"/>
        </inertial>
        <visual>
            <geometry>
                <box size="0.2 0.18 0.08"/>
            </geometry>
            <material name="body_color"/>
        </visual>
        <collision>
            <geometry>
                <box size="0.2 0.18 0.08"/>
            </geometry>
        </collision>
    </link>

    <joint name="base_to_pelvis" type="fixed">
        <parent link="base_link"/>
        <child link="pelvis"/>
        <origin xyz="0 0 0.05"/>
    </joint>

    <!-- ==================== LEFT LEG ==================== -->
    <!-- Left Upper Leg -->
    <link name="left_upper_leg">
        <inertial>
            <mass value="2.5"/>
            <origin xyz="0 -0.05 -0.2"/>
            <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.01"/>
        </inertial>
        <visual>
            <geometry>
                <cylinder length="0.35" radius="0.06"/>
            </geometry>
            <material name="body_color"/>
        </visual>
        <collision>
            <geometry>
                <cylinder length="0.35" radius="0.06"/>
            </geometry>
        </collision>
    </link>

    <!-- Left Hip Joint (Pitch) -->
    <joint name="left_hip_pitch" type="revolute">
        <parent link="pelvis"/>
        <child link="left_upper_leg"/>
        <origin xyz="0 -0.1 0"/>
        <axis xyz="1 0 0"/>
        <limit lower="-1.57" upper="1.57" effort="100" velocity="5"/>
        <dynamics damping="0.5"/>
    </joint>

    <!-- Left Lower Leg -->
    <link name="left_lower_leg">
        <inertial>
            <mass value="2.0"/>
            <origin xyz="0 -0.02 -0.18"/>
            <inertia ixx="0.015" ixy="0" ixz="0" iyy="0.015" iyz="0" izz="0.005"/>
        </inertial>
        <visual>
            <geometry>
                <cylinder length="0.32" radius="0.05"/>
            </geometry>
            <material name="body_color"/>
        </visual>
        <collision>
            <geometry>
                <cylinder length="0.32" radius="0.05"/>
            </geometry>
        </collision>
    </link>

    <!-- Left Knee Joint -->
    <joint name="left_knee" type="revolute">
        <parent link="left_upper_leg"/>
        <child link="left_lower_leg"/>
        <origin xyz="0 -0.1 -0.35"/>
        <axis xyz="1 0 0"/>
        <limit lower="-2.0" upper="0" effort="80" velocity="8"/>
        <dynamics damping="0.3"/>
    </joint>

    <!-- Left Foot -->
    <link name="left_foot">
        <inertial>
            <mass value="0.5"/>
            <origin xyz="0 0.02 0"/>
            <inertia ixx="0.002" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.003"/>
        </inertial>
        <visual>
            <geometry>
                <box size="0.12 0.04 0.2"/>
            </geometry>
            <material name="joint_color"/>
        </visual>
        <collision>
            <geometry>
                <box size="0.12 0.04 0.2"/>
            </geometry>
        </collision>
    </link>

    <!-- Left Ankle Joint -->
    <joint name="left_ankle" type="revolute">
        <parent link="left_lower_leg"/>
        <child link="left_foot"/>
        <origin xyz="0 -0.1 -0.32"/>
        <axis xyz="1 0 0"/>
        <limit lower="-0.5" upper="0.5" effort="50" velocity="10"/>
        <dynamics damping="0.2"/>
    </joint>

    <!-- ==================== RIGHT LEG (Mirror of Left) ==================== -->
    <link name="right_upper_leg">
        <inertial>
            <mass value="2.5"/>
            <origin xyz="0 -0.05 -0.2"/>
            <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.01"/>
        </inertial>
        <visual>
            <geometry>
                <cylinder length="0.35" radius="0.06"/>
            </geometry>
            <material name="body_color"/>
        </visual>
        <collision>
            <geometry>
                <cylinder length="0.35" radius="0.06"/>
            </geometry>
        </collision>
    </link>

    <joint name="right_hip_pitch" type="revolute">
        <parent link="pelvis"/>
        <child link="right_upper_leg"/>
        <origin xyz="0 0.1 0"/>
        <axis xyz="1 0 0"/>
        <limit lower="-1.57" upper="1.57" effort="100" velocity="5"/>
        <dynamics damping="0.5"/>
    </joint>

    <!-- (Similar joints for right knee and ankle...) -->

    <!-- ==================== HEAD ==================== -->
    <link name="head">
        <inertial>
            <mass value="1.5"/>
            <origin xyz="0 0 0.1"/>
            <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
        </inertial>
        <visual>
            <geometry>
                <sphere radius="0.12"/>
            </geometry>
            <material name="body_color"/>
        </visual>
        <collision>
            <geometry>
                <sphere radius="0.12"/>
            </geometry>
        </collision>
    </link>

    <joint name="neck" type="revolute">
        <parent link="torso"/>
        <child link="head"/>
        <origin xyz="0 0 0.25"/>
        <axis xyz="0 1 0"/>
        <limit lower="-1.0" upper="1.0" effort="20" velocity="3"/>
        <dynamics damping="0.3"/>
    </joint>

    <!-- ==================== SENSOR (LiDAR) ==================== -->
    <link name="lidar_link">
        <inertial>
            <mass value="0.1"/>
            <origin xyz="0 0 0"/>
            <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
        </inertial>
        <visual>
            <geometry>
                <cylinder length="0.05" radius="0.03"/>
            </geometry>
            <material name="sensor_color"/>
        </visual>
        <collision>
            <geometry>
                <cylinder length="0.05" radius="0.03"/>
            </geometry>
        </collision>
    </link>

    <joint name="lidar_mount" type="fixed">
        <parent link="head"/>
        <child link="lidar_link"/>
        <origin xyz="0 0 0.15"/>
    </joint>

    <!-- ==================== TRANSMISSIONS ==================== -->
    <transmission name="left_hip_pitch_trans">
        <type>transmission_interface/SimpleTransmission</type>
        <actuator name="left_hip_pitch_motor">
            <hardwareInterface>PositionJointInterface</hardwareInterface>
        </actuator>
        <joint name="left_hip_pitch">
            <mechanicalReduction>1</mechanicalReduction>
        </joint>
    </transmission>

    <transmission name="left_knee_trans">
        <type>transmission_interface/SimpleTransmission</type>
        <actuator name="left_knee_motor">
            <hardwareInterface>PositionJointInterface</hardwareInterface>
        </actuator>
        <joint name="left_knee">
            <mechanicalReduction>1</mechanicalReduction>
        </joint>
    </transmission>

</robot>
```

## Using Xacro for Modular Descriptions

Xacro (XML Macro) enables parameterized and modular robot descriptions:

```xml
<?xml version="1.0"?>
<robot name="xacro_humanoid" xmlns:xacro="http://ros.org/wiki/xacro">

    <!-- Include macros file -->
    <xacro:include filename="leg_macro.xacro"/>
    <xacro:include filename="arm_macro.xacro"/>

    <!-- Define parameters -->
    <xacro:property name="body_color" value="0.8 0.8 0.8 1"/>
    <xacro:property name="leg_length" value="0.35"/>

    <!-- Use macros to generate robot -->
    <xacro:leg prefix="left_" side="-1" color="${body_color}"/>
    <xacro:leg prefix="right_" side="1" color="${body_color}"/>
    <xacro:arm prefix="left_" side="-1"/>
    <xacro:arm prefix="right_" side="1"/>

</robot>
```

### Leg Macro Example

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://ros.org/wiki/xacro">

    <xacro:macro name="leg" params="prefix side color">
        <!-- Upper Leg -->
        <link name="${prefix}upper_leg">
            <visual>
                <geometry>
                    <cylinder length="${leg_length}" radius="0.06"/>
                </geometry>
                <material name="${color}"/>
            </visual>
        </link>

        <!-- Hip Joint -->
        <joint name="${prefix}hip" type="revolute">
            <origin xyz="0 ${side*0.1} 0"/>
            <axis xyz="1 0 0"/>
            <limit lower="-1.57" upper="1.57"/>
        </joint>

        <!-- Lower Leg -->
        <link name="${prefix}lower_leg">
            <visual>
                <geometry>
                    <cylinder length="${leg_length}" radius="0.05"/>
                </geometry>
                <material name="${color}"/>
            </visual>
        </link>

        <!-- Knee Joint -->
        <joint name="${prefix}knee" type="revolute">
            <origin xyz="0 0 -${leg_length}"/>
            <axis xyz="1 0 0"/>
            <limit lower="-2.0" upper="0"/>
        </joint>
    </xacro:macro>

</robot>
```

## URDF Validation and Testing

### Validating URDF Files

```bash
# Install URDF validation tools
sudo apt install liburdfdom-tools

# Validate URDF file
check_urdf my_robot.urdf

# Convert URDF to text description
urdf_to_graphiz my_robot.urdf
```

### Testing URDF in RViz

```xml
<!-- Launch file to visualize URDF -->
<launch>
    <param name="robot_description" command="xacro $(find my_robot_description)/urdf/my_robot.urdf.xacro"/>

    <node name="robot_state_publisher" pkg="robot_state_publisher" type="robot_state_publisher"/>

    <node name="rviz" pkg="rviz2" type="rviz2" args="-d $(find my_robot_description)/rviz/robot_description.rviz"/>
</launch>
```

## Best Practices for URDF Creation

### Inertial Properties

Accurate inertial properties are crucial for simulation:

```xml
<!-- Properly calculated inertial properties -->
<link name="upper_arm">
    <inertial>
        <mass value="2.0"/>
        <origin xyz="0.0 -0.05 0.15"/>
        <inertia ixx="0.005" ixy="0.0" ixz="0.0"
                 iyy="0.004" iyz="0.0" izz="0.002"/>
    </inertial>
</link>
```

### Joint Limits and Safety

Always define appropriate joint limits:

```xml
<joint name="shoulder_pitch" type="revolute">
    <parent link="torso"/>
    <child link="upper_arm"/>
    <origin xyz="0 0.2 0.3"/>
    <axis xyz="1 0 0"/>
    <!-- Safe limits for humanoid robot -->
    <limit lower="-2.0" upper="2.0" effort="50" velocity="2.0"/>
    <dynamics damping="0.5" friction="0.1"/>
</joint>
```

## Exercise: Create a Complete Humanoid URDF

Create a URDF file for a humanoid robot with:
1. 18+ degrees of freedom (9 per leg, plus arms)
2. Include pelvis, torso, head, arms, and legs
3. Add visual and collision geometries
4. Include inertial properties
5. Add sensor mount for camera
6. Use transmission elements for all joints

## Learning Outcomes

After completing this section, students will be able to:
1. Create URDF files for humanoid robots
2. Define links, joints, and their properties
3. Use Xacro for modular robot descriptions
4. Validate URDF files
5. Test URDF in visualization tools

## Next Steps

Continue to [Middleware for Real-Time Control](./middleware-real-time-control.md) to learn about Quality of Service and real-time considerations in ROS 2.