---
slug: /weekly-breakdown/week-2
title: "Week 2: Simulation Environments - Gazebo اور Unity"
hide_table_of_contents: false
---

# Week 2: Simulation Environments - Gazebo اور Unity (ہفتہ 2)

## جائزہ (Overview)
اس ہفتے میں humanoid robots کے لیے simulation environments create اور work کرنے پر focus ہے۔ Students Gazebo اور Unity use کرتے ہوئے realistic simulation worlds build کرنا سیکھیں گے، physics accuracy اور visual fidelity پر emphasis کے ساتھ۔

## سیکھنے کے اہداف (Learning Objectives)
اس ہفتے کے آخر تک، students یہ کر سکیں گے:
- Humanoid robots کے لیے Gazebo simulation environments create کریں
- Realistic robot behavior کے لیے physics properties configure کریں
- High-fidelity robot visualization کے لیے Unity setup کریں
- Hardware-in-the-loop testing کے لیے simulation کو ROS 2 کے ساتھ integrate کریں
- Simulations سے synthetic training data generate کریں

## Day 1: Gazebo Fundamentals
### مضامین کا احاطہ (Topics Covered)
- Gazebo architecture اور components
- World file creation اور physics configuration
- URDF/SDF کے ساتھ robot model integration
- Sensor simulation (LiDAR, cameras, IMU)

### Hands-on Activities
- Gazebo Garden install اور configure کریں
- Obstacles کے ساتھ basic world file create کریں
- Robot model کو Gazebo میں import کریں
- Basic physics simulation test کریں

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
### مضامین کا احاطہ (Topics Covered)
- Complex environment creation
- Dynamic object simulation
- Contact sensors اور force feedback
- Performance optimization techniques

### Hands-on Activities
- Multi-room environment create کریں
- Dynamic obstacles add کریں
- Contact sensors configure کریں
- Simulation performance optimize کریں

## Day 3: Unity ML-Agents Setup
### مضامین کا احاطہ (Topics Covered)
- Unity installation اور ML-Agents toolkit
- Robot model import اور configuration
- Humanoid locomotion کے لیے physics setup
- Camera اور sensor configuration

### Hands-on Activities
- Unity 2022.3 LTS install کریں
- ML-Agents environment setup کریں
- Humanoid robot model import کریں
- Physics اور sensors configure کریں

### Code Tasks
```csharp
// Robot control کے لیے Unity C# script
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
        // Robot position اور target reset کریں
        rb.velocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
        transform.position = new Vector3(Random.Range(-3f, 3f), 0.5f, Random.Range(-3f, 3f));
        target.position = new Vector3(Random.Range(-3f, 3f), 0.5f, Random.Range(-3f, 3f));
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Agent کے لیے observations add کریں
        sensor.AddObservation(transform.position);
        sensor.AddObservation(target.position);
        sensor.AddObservation(rb.velocity);
    }

    public override void OnActionReceived(ActionBuffers actions)
        // Robot move کرنے کے لیے actions process کریں
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
        // Human testing کے لیے
        var continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = Input.GetAxis("Horizontal");
        continuousActionsOut[1] = Input.GetAxis("Vertical");
    }
}
```

## Day 4: Isaac Sim Introduction
### مضامین کا احاطہ (Topics Covered)
- NVIDIA Isaac Sim architecture
- USD (Universal Scene Description) format
- GPU-accelerated physics simulation
- ROS 2 کے ساتھ integration

### Hands-on Activities
- Isaac Sim install کریں
- First USD scene create کریں
- Robot model کو Isaac Sim میں import کریں
- GPU-accelerated physics test کریں

## Day 5: Simulation Integration اور Testing
### مضامین کا احاطہ (Topics Covered)
- Simulation کو ROS 2 سے connect کرنا
- Hardware-in-the-loop testing
- Synthetic data generation
- Performance benchmarking

### Hands-on Activities
- Simulation کے لیے ROS 2 bridges setup کریں
- Simulation میں robot control test کریں
- AI training کے لیے synthetic data generate کریں
- Simulation vs real robot performance compare کریں

## Assessment (تقييم)
- Humanoid robot کے ساتھ complete Gazebo environment create کریں
- Robot model کے ساتھ Unity scene setup کریں
- Simulation میں basic robot control implement کریں
- AI training کے لیے synthetic data generate کریں

## اگلے ہفتے کا پیش نظر (Next Week Preview)
Week 3 میں NVIDIA Isaac platform پر deeper dive ہوگا، Isaac Sim اور Isaac ROS integration پر focus کرتے ہوئے advanced robotics applications کے لیے۔ Students perception اور navigation کے لیے GPU acceleration سیکھیں گے۔
