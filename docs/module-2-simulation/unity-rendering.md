# Unity Rendering for Human-Robot Interaction

## Introduction to Unity for Robotics

Unity is a powerful 3D development platform that provides high-fidelity rendering capabilities for robotics applications. For Human-Robot Interaction (HRI), Unity offers:

- **Photorealistic Graphics**: High-quality rendering for immersive experiences
- **Real-time Performance**: Smooth interaction at 60+ FPS
- **Cross-platform Support**: Deploy to desktop, mobile, and AR/VR platforms
- **Asset Integration**: Import 3D models, animations, and materials
- **Scripting**: C# programming for complex robot behaviors

## Unity ML-Agents Overview

Unity ML-Agents is a toolkit for training intelligent agents using reinforcement learning:

```python
# Python API for Unity ML-Agents
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel
)
from mlagents_envs.side_channel.manager_side_channel import (
    ManagerSideChannel
)

# Initialize Unity environment
engine_config = EngineConfigurationChannel()
env = UnityEnvironment(
    file_name="humanoid_sim",  # Path to Unity build
    side_channels=[engine_config],
    worker_id=0
)

# Configure engine
engine_config.set_configuration_parameters(
    width=1280,
    height=720,
    quality_level=5,      # High quality
    time_scale=1.0        # Normal speed
)

# Reset environment
env.reset()

# Get behavior specifications
behavior_name = list(env.behavior_specs.keys())[0]
spec = env.behavior_specs[behavior_name]

print(f"Observation shapes: {[obs.shape for obs in spec.observation_specs]}")
print(f"Action shape: {spec.action_spec}")
```

## Setting Up Unity for Robotics

### Unity Project Configuration

```csharp
// Unity Editor configuration for robotics
using UnityEditor;

public class RobotProjectSetup : MonoBehaviour
{
    [MenuItem("Robotics/Setup Project")]
    public static void SetupRobotProject()
    {
        // Configure player settings
        PlayerSettings.companyName = "RoboticsLab";
        PlayerSettings.productName = "HumanoidRobotSim";
        PlayerSettings.bundleVersion = "1.0.0";

        // Configure quality settings
        QualitySettings.vSyncCount = 0;  // Disable V-Sync for consistent frame rates
        QualitySettings.maxQueuedFrames = 1;  // Minimize input lag
        QualitySettings.antiAliasing = 4;  // Moderate AA for quality

        // Configure XR settings if needed
        // XRSettings.enabled = true;

        Debug.Log("Robotics project configured!");
    }
}
```

### High-Fidelity Rendering Settings

```csharp
// Humanoid rendering configuration
using UnityEngine;
using UnityEngine.Rendering;

public class HumanoidRenderer : MonoBehaviour
{
    public Material humanoidMaterial;
    public Light mainLight;
    public Light fillLight;
    public Light rimLight;

    void Start()
    {
        // Configure high-quality rendering
        ConfigureRendering();
        ConfigureLighting();
        ConfigurePostProcessing();
    }

    void ConfigureRendering()
    {
        // Quality settings
        QualitySettings.shadows = ShadowQuality.All;
        QualitySettings.shadowResolution = ShadowResolution.High;
        QualitySettings.shadowProjection = ShadowProjection.StableFit;
        QualitySettings.shadowCascades = 2;
        QualitySettings.shadowDistance = 50.0f;
        QualitySettings.shadowNearPlaneOffset = 2.0f;
        QualitySettings.shadowCascade2Split = 0.33f;
        QualitySettings.shadowCascade4Split = new Vector3(0.067f, 0.2f, 0.467f);

        // Texture quality
        QualitySettings.masterTextureLimit = 0;  // Full resolution
        QualitySettings.anisotropicFiltering = AnisotropicFiltering.Enable;

        // Anti-aliasing
        QualitySettings.antiAliasing = 4;  // 4x MSAA

        // Real-time reflection probes for metallic surfaces
        QualitySettings.realtimeReflectionProbes = true;
        QualitySettings.billboardsFaceCameraPosition = true;
    }

    void ConfigureLighting()
    {
        // Main directional light (sun)
        mainLight.type = LightType.Directional;
        mainLight.color = Color.white;
        mainLight.intensity = 1.0f;
        mainLight.shadows = LightShadows.Soft;
        mainLight.shadowStrength = 0.8f;
        mainLight.shadowResolution = LightShadowResolution.Medium;
        mainLight.shadowBias = 0.05f;
        mainLight.shadowNormalBias = 0.4f;
        mainLight.shadowNearPlane = 0.2f;

        // Fill light for softer shadows
        fillLight.type = LightType.Directional;
        fillLight.color = Color.gray;
        fillLight.intensity = 0.3f;
        fillLight.shadows = LightShadows.None;

        // Rim light for highlighting edges
        rimLight.type = LightType.Directional;
        rimLight.color = Color.blue;
        rimLight.intensity = 0.1f;
        rimLight.shadows = LightShadows.None;
    }

    void ConfigurePostProcessing()
    {
        // If using Post-Processing Stack
        var volume = GetComponent<UnityEngine.Rendering.Volume>();
        if (volume != null)
        {
            // Configure color grading
            var colorGrading = volume.profile.components
                .Find<UnityEngine.Rendering.ColorGrading>();

            if (colorGrading != null)
            {
                colorGrading.contrast.value = 10f;
                colorGrading.saturation.value = 5f;
                colorGrading.temperature.value = 10f;
            }
        }
    }
}
```

## Creating Humanoid Robot Models

### Robot Armature and Animation

```csharp
// Humanoid robot controller with inverse kinematics
using UnityEngine;
using UnityEngine.Animations.Rigging;

public class HumanoidController : MonoBehaviour
{
    [Header("Joint Transforms")]
    public Transform pelvis;
    public Transform[] leftArmJoints;   // shoulder, elbow, wrist
    public Transform[] rightArmJoints;  // shoulder, elbow, wrist
    public Transform[] leftLegJoints;   // hip, knee, ankle
    public Transform[] rightLegJoints;  // hip, knee, ankle

    [Header("IK Targets")]
    public Transform leftHandTarget;
    public Transform rightHandTarget;
    public Transform leftFootTarget;
    public Transform rightFootTarget;

    [Header("Physical Properties")]
    public float bodyMass = 60f;
    public float jointStiffness = 1000f;

    void Start()
    {
        ConfigurePhysicalCharacteristics();
    }

    void ConfigurePhysicalCharacteristics()
    {
        // Configure mass distribution
        foreach (Transform joint in leftLegJoints)
        {
            var rb = joint.GetComponent<Rigidbody>();
            if (rb != null)
            {
                rb.mass = bodyMass * 0.1f; // Legs ~10% of body mass each
            }
        }

        foreach (Transform joint in rightLegJoints)
        {
            var rb = joint.GetComponent<Rigidbody>();
            if (rb != null)
            {
                rb.mass = bodyMass * 0.1f;
            }
        }
    }

    void LateUpdate()
    {
        // Apply inverse kinematics for natural movement
        ApplyInverseKinematics();
    }

    void ApplyInverseKinematics()
    {
        // Simple IK for hands and feet
        if (leftHandTarget != null)
            MoveToTarget(leftArmJoints[leftArmJoints.Length - 1], leftHandTarget);

        if (rightHandTarget != null)
            MoveToTarget(rightArmJoints[rightArmJoints.Length - 1], rightHandTarget);

        if (leftFootTarget != null)
            MoveToTarget(leftLegJoints[leftLegJoints.Length - 1], leftFootTarget);

        if (rightFootTarget != null)
            MoveToTarget(rightLegJoints[rightLegJoints.Length - 1], rightFootTarget);
    }

    void MoveToTarget(Transform joint, Transform target)
    {
        // Simple proportional control for IK
        Vector3 direction = (target.position - joint.position).normalized;
        joint.position += direction * Time.deltaTime * jointStiffness * 0.01f;
    }
}
```

## ROS-Unity Integration

### ROS TCP Connector Setup

```csharp
// Unity ROS connection for robot control
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Std;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;

public class UnityRobotController : MonoBehaviour
{
    ROSConnection ros;
    public string rosIPAddress = "127.0.0.1";
    public int rosPort = 10000;

    [Header("Robot Configuration")]
    public ArticulationBody[] joints;
    public float[] targetPositions;

    void Start()
    {
        // Initialize ROS connection
        ros = ROSConnection.GetOrCreateInstance();
        ros.Initialize(rosIPAddress, rosPort);

        // Subscribe to ROS topics
        ros.Subscribe<Float64MultiArray>("/humanoid/joint_commands", JointCommandCallback);
        ros.Subscribe<Twist>("/humanoid/cmd_vel", VelocityCommandCallback);

        // Get all articulation bodies (joints)
        joints = GetComponentsInChildren<ArticulationBody>();
        targetPositions = new float[joints.Length];

        Debug.Log($"Connected to ROS at {rosIPAddress}:{rosPort}");
    }

    void FixedUpdate()
    {
        // Apply joint commands
        for (int i = 0; i < joints.Length && i < targetPositions.Length; i++)
        {
            var drive = joints[i].xDrive;
            drive.target = targetPositions[i];
            joints[i].xDrive = drive;
        }
    }

    void JointCommandCallback(Float64MultiArray msg)
    {
        // Update target positions from ROS
        for (int i = 0; i < Mathf.Min(msg.data.Length, targetPositions.Length); i++)
        {
            targetPositions[i] = (float)msg.data[i];
        }
    }

    void VelocityCommandCallback(Twist msg)
    {
        // Convert velocity command to base movement
        Vector3 linearVel = new Vector3((float)msg.linear.x, (float)msg.linear.y, (float)msg.linear.z);
        Vector3 angularVel = new Vector3((float)msg.angular.x, (float)msg.angular.y, (float)msg.angular.z);

        // Apply to base of robot
        Rigidbody baseRb = GetComponent<Rigidbody>();
        if (baseRb != null)
        {
            baseRb.AddForce(linearVel, ForceMode.VelocityChange);
            baseRb.AddTorque(angularVel, ForceMode.VelocityChange);
        }
    }

    void OnDestroy()
    {
        // Clean up ROS connection
        if (ros != null)
        {
            ros.Close();
        }
    }
}
```

## Human-Robot Interaction (HRI) Features

### Interactive Elements

```csharp
// Interactive object for HRI
using UnityEngine;

public class InteractiveObject : MonoBehaviour
{
    public Material defaultMaterial;
    public Material highlightMaterial;
    public Renderer objectRenderer;
    public bool isGrabbable = true;
    public bool isTouchable = true;

    private bool isHighlighted = false;
    private bool isGrabbed = false;

    void Start()
    {
        if (objectRenderer == null)
            objectRenderer = GetComponent<Renderer>();
    }

    void OnMouseEnter()
    {
        if (!isHighlighted)
        {
            Highlight();
        }
    }

    void OnMouseExit()
    {
        if (isHighlighted)
        {
            Unhighlight();
        }
    }

    void OnMouseDown()
    {
        if (isGrabbable)
        {
            Grab();
        }
    }

    void Highlight()
    {
        if (objectRenderer != null && highlightMaterial != null)
        {
            objectRenderer.material = highlightMaterial;
            isHighlighted = true;
        }
    }

    void Unhighlight()
    {
        if (objectRenderer != null && defaultMaterial != null)
        {
            objectRenderer.material = defaultMaterial;
            isHighlighted = false;
        }
    }

    void Grab()
    {
        isGrabbed = true;
        // Add grab logic here
        Debug.Log($"Grabbed object: {gameObject.name}");

        // Send ROS message about interaction
        // ros.Publish("/interaction_events", new StringMessage($"grabbed_{gameObject.name}"));
    }

    void OnMouseUp()
    {
        if (isGrabbed)
        {
            Release();
        }
    }

    void Release()
    {
        isGrabbed = false;
        Unhighlight();
        Debug.Log($"Released object: {gameObject.name}");
    }
}
```

### Gesture Recognition

```csharp
// Simple gesture recognition for HRI
using UnityEngine;
using System.Collections.Generic;

public class GestureRecognizer : MonoBehaviour
{
    public List<Vector3> gesturePath = new List<Vector3>();
    public float gestureThreshold = 0.1f;
    public float maxGestureTime = 2.0f;

    private float gestureStartTime;
    private bool isRecording = false;

    void Update()
    {
        if (isRecording)
        {
            // Record mouse/finger position
            Vector3 currentPos = GetCursorPosition();
            if (gesturePath.Count == 0 ||
                Vector3.Distance(gesturePath[gesturePath.Count - 1], currentPos) > gestureThreshold)
            {
                gesturePath.Add(currentPos);

                // Check if gesture took too long
                if (Time.time - gestureStartTime > maxGestureTime)
                {
                    StopRecording();
                    ProcessGesture();
                }
            }
        }
    }

    Vector3 GetCursorPosition()
    {
        // Get cursor position in world coordinates
        Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
        RaycastHit hit;

        if (Physics.Raycast(ray, out hit))
        {
            return hit.point;
        }

        return Vector3.zero;
    }

    public void StartRecording()
    {
        gesturePath.Clear();
        gestureStartTime = Time.time;
        isRecording = true;
    }

    public void StopRecording()
    {
        isRecording = false;
    }

    void ProcessGesture()
    {
        // Simple gesture classification
        if (gesturePath.Count >= 3)
        {
            string gesture = ClassifyGesture();
            Debug.Log($"Recognized gesture: {gesture}");

            // Send gesture to ROS
            // ros.Publish("/gesture_recognition", new StringMessage(gesture));
        }
    }

    string ClassifyGesture()
    {
        // Simple gesture classification based on path
        Vector3 startPoint = gesturePath[0];
        Vector3 endPoint = gesturePath[gesturePath.Count - 1];
        Vector3 direction = (endPoint - startPoint).normalized;

        // Classify based on direction and path characteristics
        if (Mathf.Abs(direction.x) > Mathf.Abs(direction.y))
        {
            return direction.x > 0 ? "swipe_right" : "swipe_left";
        }
        else
        {
            return direction.y > 0 ? "swipe_up" : "swipe_down";
        }
    }
}
```

## Performance Optimization

### LOD (Level of Detail) System

```csharp
// Level of Detail for humanoid robot
using UnityEngine;

[RequireComponent(typeof(Renderer))]
public class HumanoidLOD : MonoBehaviour
{
    public float lodDistance = 10.0f;
    public GameObject highDetailModel;
    public GameObject lowDetailModel;
    public Renderer detailRenderer;

    private Camera mainCamera;

    void Start()
    {
        mainCamera = Camera.main;
        if (detailRenderer == null)
            detailRenderer = GetComponent<Renderer>();
    }

    void Update()
    {
        if (mainCamera != null)
        {
            float distance = Vector3.Distance(transform.position, mainCamera.transform.position);

            if (distance < lodDistance)
            {
                if (highDetailModel != null) highDetailModel.SetActive(true);
                if (lowDetailModel != null) lowDetailModel.SetActive(false);
            }
            else
            {
                if (highDetailModel != null) highDetailModel.SetActive(false);
                if (lowDetailModel != null) lowDetailModel.SetActive(true);
            }
        }
    }
}
```

## Exercise: Create Interactive HRI Scene

Create a Unity scene with:
1. Humanoid robot model with articulated joints
2. Interactive objects for manipulation
3. Gesture recognition system
4. ROS connection for remote control
5. High-fidelity rendering settings

## Learning Outcomes

After completing this section, students will be able to:
1. Set up Unity projects for robotics applications
2. Configure high-fidelity rendering for HRI
3. Implement ROS-Unity integration
4. Create interactive elements for HRI
5. Optimize Unity scenes for performance

## Next Steps

Continue to [Sensor Simulation](./sensor-simulation.md) to learn about simulating LiDAR, depth cameras, and IMUs in Unity.