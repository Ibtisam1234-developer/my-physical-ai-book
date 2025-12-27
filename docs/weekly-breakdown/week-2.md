# Week 2: Simulation Environments - Gazebo and Unity

## Overview
This week focuses on creating and working with simulation environments for humanoid robots. Students will learn to build realistic simulation worlds using both Gazebo and Unity, with emphasis on physics accuracy and visual fidelity.

## Learning Objectives
By the end of this week, students will be able to:
- Create Gazebo simulation environments for humanoid robots
- Configure physics properties for realistic robot behavior
- Set up Unity for high-fidelity robot visualization
- Integrate simulation with ROS 2 for hardware-in-the-loop testing
- Generate synthetic training data from simulations

## Day 1: Gazebo Fundamentals
### Topics Covered
- Gazebo architecture and components
- World file creation and physics configuration
- Robot model integration with URDF/SDF
- Sensor simulation (LiDAR, cameras, IMU)

### Hands-on Activities
- Install and configure Gazebo Garden
- Create basic world file with obstacles
- Import robot model into Gazebo
- Test basic physics simulation

### Code Tasks
```xml
<!-- Example world file structure -->
<?xml version="1.0"?>
<sdf version="1.8">
  <world name="humanoid_lab">
    <!-- Physics engine configuration -->
    <physics name="ode_physics" type="ode">
      <gravity>0 0 -9.81</gravity>
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

    <!-- Ground plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

## Day 2: Advanced Gazebo Simulation
### Topics Covered
- Complex environment creation
- Dynamic object simulation
- Contact sensors and force feedback
- Performance optimization techniques

### Hands-on Activities
- Create multi-room environment
- Add dynamic obstacles
- Configure contact sensors
- Optimize simulation performance

## Day 3: Unity ML-Agents Setup
### Topics Covered
- Unity installation and ML-Agents toolkit
- Robot model import and configuration
- Physics setup for humanoid locomotion
- Camera and sensor configuration

### Hands-on Activities
- Install Unity 2022.3 LTS
- Set up ML-Agents environment
- Import humanoid robot model
- Configure physics and sensors

### Code Tasks
```csharp
// Unity C# script for robot control
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class HumanoidAgent : Agent
{
    [SerializeField] private Transform target;
    [SerializeField] private float moveSpeed = 5f;

    private Rigidbody rb;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
    }

    public override void OnEpisodeBegin()
    {
        // Reset robot position and target
        rb.velocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
        transform.position = new Vector3(Random.Range(-3f, 3f), 0.5f, Random.Range(-3f, 3f));
        target.position = new Vector3(Random.Range(-3f, 3f), 0.5f, Random.Range(-3f, 3f));
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Add observations for the agent
        sensor.AddObservation(transform.position);
        sensor.AddObservation(target.position);
        sensor.AddObservation(rb.velocity);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        // Process actions to move the robot
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = actions.ContinuousActions[0];
        controlSignal.z = actions.ContinuousActions[1];

        rb.AddForce(controlSignal * moveSpeed);

        // Reward system
        float distanceToTarget = Vector3.Distance(transform.position, target.position);
        SetReward(-distanceToTarget * 0.01f);

        if (distanceToTarget < 1.0f)
        {
            SetReward(1.0f);
            EndEpisode();
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // For human testing
        var continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = Input.GetAxis("Horizontal");
        continuousActionsOut[1] = Input.GetAxis("Vertical");
    }
}
```

## Day 4: Isaac Sim Introduction
### Topics Covered
- NVIDIA Isaac Sim architecture
- USD (Universal Scene Description) format
- GPU-accelerated physics simulation
- Integration with ROS 2

### Hands-on Activities
- Install Isaac Sim
- Create first USD scene
- Import robot model into Isaac Sim
- Test GPU-accelerated physics

## Day 5: Simulation Integration and Testing
### Topics Covered
- Connecting simulation to ROS 2
- Hardware-in-the-loop testing
- Synthetic data generation
- Performance benchmarking

### Hands-on Activities
- Set up ROS 2 bridges for simulation
- Test robot control in simulation
- Generate synthetic training data
- Compare simulation vs real robot performance

## Assessment
- Create complete Gazebo environment with humanoid robot
- Set up Unity scene with robot model
- Implement basic robot control in simulation
- Generate synthetic data for AI training

## Next Week Preview
Week 3 will dive into the NVIDIA Isaac platform, focusing on Isaac Sim and Isaac ROS integration for advanced robotics applications. Students will learn to leverage GPU acceleration for perception and navigation.