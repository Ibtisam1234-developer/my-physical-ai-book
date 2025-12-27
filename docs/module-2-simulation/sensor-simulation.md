# Sensor Simulation

## Introduction to Sensor Simulation

Robotic systems rely heavily on sensors to perceive their environment and themselves. In simulation, we must accurately model these sensors to ensure that algorithms developed in simulation will transfer to real robots.

### Types of Sensors in Robotics

- **Proprioceptive Sensors**: Sense the robot's own state (joint encoders, IMU)
- **Exteroceptive Sensors**: Sense the environment (cameras, LiDAR, sonar)
- **Interoceptive Sensors**: Sense internal state (temperature, power)

## LiDAR Simulation

LiDAR (Light Detection and Ranging) sensors emit laser beams and measure the time it takes for them to return after reflecting off objects.

### Gazebo LiDAR Configuration

```xml
<!-- LiDAR sensor configuration -->
<link name="lidar_link">
    <visual>
        <geometry>
            <cylinder length="0.05" radius="0.03"/>
        </geometry>
        <material name="sensor_material"/>
    </visual>
    <collision>
        <geometry>
            <cylinder length="0.05" radius="0.03"/>
        </geometry>
    </collision>
    <inertial>
        <mass value="0.1"/>
        <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
</link>

<joint name="lidar_mount" type="fixed">
    <parent link="head"/>
    <child link="lidar_link"/>
    <origin xyz="0 0 0.15"/>
</joint>

<!-- LiDAR sensor plugin -->
<gazebo reference="lidar_link">
    <sensor name="lidar" type="ray">
        <always_on>true</always_on>
        <update_rate>10</update_rate>
        <visualize>true</visualize>
        <ray>
            <scan>
                <horizontal>
                    <samples>360</samples>
                    <resolution>1.0</resolution>
                    <min_angle>-3.14159</min_angle> <!-- -π -->
                    <max_angle>3.14159</max_angle>   <!-- π -->
                </horizontal>
                <vertical>
                    <samples>1</samples>
                    <resolution>1.0</resolution>
                    <min_angle>0</min_angle>
                    <max_angle>0</max_angle>
                </vertical>
            </scan>
            <range>
                <min>0.1</min>
                <max>30.0</max>
                <resolution>0.01</resolution>
            </range>
        </ray>
        <plugin name="lidar_plugin" filename="libgazebo_ros_ray_sensor.so">
            <ros>
                <namespace>/humanoid1</namespace>
                <remapping>~/out:=scan</remapping>
            </ros>
            <output_type>sensor_msgs/LaserScan</output_type>
        </plugin>
    </sensor>
</gazebo>
```

### Advanced LiDAR Configuration

```xml
<!-- 3D LiDAR with vertical scanning -->
<sensor name="velodyne_lidar" type="ray">
    <always_on>true</always_on>
    <update_rate>10</update_rate>
    <visualize>true</visualize>
    <ray>
        <scan>
            <horizontal>
                <samples>800</samples>
                <resolution>1</resolution>
                <min_angle>-3.14159</min_angle>
                <max_angle>3.14159</max_angle>
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
            <max>100.0</max>
            <resolution>0.01</resolution>
        </range>
        <noise>
            <type>gaussian</type>
            <mean>0.0</mean>
            <stddev>0.01</stddev>
        </noise>
    </ray>
</sensor>
```

## Camera Simulation

Cameras provide visual information about the environment. We need to configure both RGB and depth cameras.

### RGB Camera Configuration

```xml
<!-- RGB Camera configuration -->
<link name="camera_link">
    <visual>
        <geometry>
            <box size="0.05 0.05 0.05"/>
        </geometry>
        <material name="sensor_material"/>
    </visual>
    <collision>
        <geometry>
            <box size="0.05 0.05 0.05"/>
        </geometry>
    </collision>
    <inertial>
        <mass value="0.05"/>
        <inertia ixx="0.00001" ixy="0" ixz="0" iyy="0.00001" iyz="0" izz="0.00001"/>
    </inertial>
</link>

<joint name="camera_mount" type="fixed">
    <parent link="head"/>
    <child link="camera_link"/>
    <origin xyz="0.1 0 0.1"/>
</joint>

<gazebo reference="camera_link">
    <sensor name="camera" type="camera">
        <always_on>true</always_on>
        <update_rate>30</update_rate>
        <visualize>true</visualize>
        <camera>
            <horizontal_fov>1.396</horizontal_fov> <!-- 80 degrees -->
            <image>
                <width>640</width>
                <height>480</height>
                <format>R8G8B8</format>
            </image>
            <clip>
                <near>0.1</near>
                <far>30</far>
            </clip>
            <noise>
                <type>gaussian</type>
                <mean>0.0</mean>
                <stddev>0.007</stddev>
            </noise>
        </camera>
        <plugin name="camera_plugin" filename="libgazebo_ros_camera.so">
            <ros>
                <namespace>/humanoid1</namespace>
                <remapping>image_raw:=camera/image_raw</remapping>
                <remapping>camera_info:=camera/camera_info</remapping>
            </ros>
        </plugin>
    </sensor>
</gazebo>
```

### Depth Camera Configuration

```xml
<!-- Depth camera configuration -->
<link name="depth_camera_link">
    <visual>
        <geometry>
            <box size="0.05 0.05 0.05"/>
        </geometry>
        <material name="sensor_material"/>
    </visual>
    <collision>
        <geometry>
            <box size="0.05 0.05 0.05"/>
        </geometry>
    </collision>
    <inertial>
        <mass value="0.05"/>
        <inertia ixx="0.00001" ixy="0" ixz="0" iyy="0.00001" iyz="0" izz="0.00001"/>
    </inertial>
</link>

<joint name="depth_camera_mount" type="fixed">
    <parent link="head"/>
    <child link="depth_camera_link"/>
    <origin xyz="0.1 0 0.1"/>
</joint>

<gazebo reference="depth_camera_link">
    <sensor name="depth_camera" type="depth">
        <always_on>true</always_on>
        <update_rate>30</update_rate>
        <visualize>true</visualize>
        <camera>
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
        <plugin name="depth_camera_plugin" filename="libgazebo_ros_openni_kinect.so">
            <ros>
                <namespace>/humanoid1</namespace>
                <remapping>depth/image_raw:=camera/depth/image_raw</remapping>
                <remapping>depth/camera_info:=camera/depth/camera_info</remapping>
                <remapping>rgb/image_raw:=camera/rgb/image_raw</remapping>
                <remapping>rgb/camera_info:=camera/rgb/camera_info</remapping>
            </ros>
        </plugin>
    </sensor>
</gazebo>
```

## IMU (Inertial Measurement Unit) Simulation

IMUs measure acceleration and angular velocity, essential for balance control and navigation.

### IMU Configuration

```xml
<!-- IMU sensor configuration -->
<link name="imu_link">
    <inertial>
        <mass value="0.01"/>
        <inertia ixx="0.000001" ixy="0" ixz="0" iyy="0.000001" iyz="0" izz="0.000001"/>
    </inertial>
    <visual>
        <geometry>
            <box size="0.01 0.01 0.01"/>
        </geometry>
        <material name="sensor_material"/>
    </visual>
</link>

<joint name="imu_mount" type="fixed">
    <parent link="torso"/>
    <child link="imu_link"/>
    <origin xyz="0 0 0.1"/>
</joint>

<gazebo reference="imu_link">
    <sensor name="imu_sensor" type="imu">
        <always_on>true</always_on>
        <update_rate>100</update_rate>
        <visualize>false</visualize>
        <imu>
            <angular_velocity>
                <x>
                    <noise type="gaussian">
                        <mean>0.0</mean>
                        <stddev>0.01</stddev>
                        <bias_mean>0.0</bias_mean>
                        <bias_stddev>0.001</bias_stddev>
                    </noise>
                </x>
                <y>
                    <noise type="gaussian">
                        <mean>0.0</mean>
                        <stddev>0.01</stddev>
                        <bias_mean>0.0</bias_mean>
                        <bias_stddev>0.001</bias_stddev>
                    </noise>
                </y>
                <z>
                    <noise type="gaussian">
                        <mean>0.0</mean>
                        <stddev>0.01</stddev>
                        <bias_mean>0.0</bias_mean>
                        <bias_stddev>0.001</bias_stddev>
                    </noise>
                </z>
            </angular_velocity>
            <linear_acceleration>
                <x>
                    <noise type="gaussian">
                        <mean>0.0</mean>
                        <stddev>0.017</stddev>
                        <bias_mean>0.0</bias_mean>
                        <bias_stddev>0.001</bias_stddev>
                    </noise>
                </x>
                <y>
                    <noise type="gaussian">
                        <mean>0.0</mean>
                        <stddev>0.017</stddev>
                        <bias_mean>0.0</bias_mean>
                        <bias_stddev>0.001</bias_stddev>
                    </noise>
                </y>
                <z>
                    <noise type="gaussian">
                        <mean>0.0</mean>
                        <stddev>0.017</stddev>
                        <bias_mean>0.0</bias_mean>
                        <bias_stddev>0.001</bias_stddev>
                    </noise>
                </z>
            </linear_acceleration>
        </imu>
        <plugin name="imu_plugin" filename="libgazebo_ros_imu_sensor.so">
            <ros>
                <namespace>/humanoid1</namespace>
                <remapping>~/out:=imu/data</remapping>
            </ros>
        </plugin>
    </sensor>
</gazebo>
```

## Force/Torque Sensor Simulation

Force/torque sensors measure forces and moments applied to robot joints, crucial for manipulation.

### Force/Torque Sensor Configuration

```xml
<!-- Force/Torque sensor configuration -->
<gazebo>
    <plugin name="ft_sensor_plugin" filename="libgazebo_ros_ft_sensor.so">
        <update_rate>100</update_rate>
        <topic>/humanoid1/left_foot/force_torque</topic>
        <joint_name>left_ankle</joint_name>
        <frame_id>left_foot</frame_id>
        <always_on>true</always_on>
        <body_name>left_foot</body_name>
        <gaussian_noise>0.01</gaussian_noise>
    </plugin>
</gazebo>
```

## Unity Sensor Simulation

### Camera Simulation in Unity

```csharp
// Unity camera sensor simulation
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;

public class UnityCameraSensor : MonoBehaviour
{
    public Camera unityCamera;
    public string rosTopic = "/humanoid1/camera/image_raw";
    public int imageWidth = 640;
    public int imageHeight = 480;
    public float updateRate = 30.0f;

    private RenderTexture renderTexture;
    private Texture2D texture2D;
    private ROSConnection ros;
    private float updateInterval;
    private float lastUpdateTime;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();

        // Create render texture for camera
        renderTexture = new RenderTexture(imageWidth, imageHeight, 24);
        unityCamera.targetTexture = renderTexture;

        // Create texture2D for ROS publishing
        texture2D = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);

        updateInterval = 1.0f / updateRate;
        lastUpdateTime = Time.time;
    }

    void Update()
    {
        if (Time.time - lastUpdateTime >= updateInterval)
        {
            PublishCameraImage();
            lastUpdateTime = Time.time;
        }
    }

    void PublishCameraImage()
    {
        // Copy render texture to texture2D
        RenderTexture.active = renderTexture;
        texture2D.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        texture2D.Apply();

        // Convert to byte array for ROS message
        byte[] imageData = texture2D.EncodeToPNG();

        // Create and publish ROS message
        var imageMsg = new ImageMsg
        {
            header = new StdMsgs.HeaderMsg
            {
                stamp = new builtin_interfaces.TimeMsg
                {
                    sec = (int)Time.time,
                    nanosec = (uint)((Time.time % 1) * 1e9)
                },
                frame_id = "camera_frame"
            },
            height = (uint)imageHeight,
            width = (uint)imageWidth,
            encoding = "rgb8",
            is_bigendian = 0,
            step = (uint)(imageWidth * 3), // 3 bytes per pixel (RGB)
            data = imageData
        };

        ros.Publish(rosTopic, imageMsg);
    }
}
```

### IMU Simulation in Unity

```csharp
// Unity IMU sensor simulation
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;

public class UnityIMUSensor : MonoBehaviour
{
    public string rosTopic = "/humanoid1/imu/data";
    public float updateRate = 100.0f;
    public float noiseLevel = 0.01f;

    private ROSConnection ros;
    private float updateInterval;
    private float lastUpdateTime;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        updateInterval = 1.0f / updateRate;
        lastUpdateTime = Time.time;
    }

    void Update()
    {
        if (Time.time - lastUpdateTime >= updateInterval)
        {
            PublishIMUData();
            lastUpdateTime = Time.time;
        }
    }

    void PublishIMUData()
    {
        // Get linear acceleration from rigidbody
        Vector3 linearAcceleration = GetLinearAcceleration();

        // Get angular velocity from rigidbody
        Vector3 angularVelocity = GetAngularVelocity();

        // Add noise to measurements
        linearAcceleration += Random.insideUnitSphere * noiseLevel;
        angularVelocity += Random.insideUnitSphere * noiseLevel * 0.1f;

        // Create IMU message
        var imuMsg = new ImuMsg
        {
            header = new StdMsgs.HeaderMsg
            {
                stamp = new builtin_interfaces.TimeMsg
                {
                    sec = (int)Time.time,
                    nanosec = (uint)((Time.time % 1) * 1e9)
                },
                frame_id = "imu_frame"
            },
            orientation = new GeometryMsgs.QuaternionMsg
            {
                x = transform.rotation.x,
                y = transform.rotation.y,
                z = transform.rotation.z,
                w = transform.rotation.w
            },
            angular_velocity = new GeometryMsgs.Vector3Msg
            {
                x = angularVelocity.x,
                y = angularVelocity.y,
                z = angularVelocity.z
            },
            linear_acceleration = new GeometryMsgs.Vector3Msg
            {
                x = linearAcceleration.x,
                y = linearAcceleration.y,
                z = linearAcceleration.z
            }
        };

        ros.Publish(rosTopic, imuMsg);
    }

    Vector3 GetLinearAcceleration()
    {
        // Calculate linear acceleration from change in velocity
        Rigidbody rb = GetComponent<Rigidbody>();
        if (rb != null)
        {
            return rb.velocity - rb.GetPointVelocity(Vector3.zero);
        }
        return Vector3.zero;
    }

    Vector3 GetAngularVelocity()
    {
        Rigidbody rb = GetComponent<Rigidbody>();
        if (rb != null)
        {
            return rb.angularVelocity;
        }
        return Vector3.zero;
    }
}
```

## Sensor Fusion and Calibration

### Sensor Calibration

```python
#!/usr/bin/env python3
"""
Example: Sensor calibration for simulation.
"""

import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import Imu, CameraInfo
from geometry_msgs.msg import TransformStamped
import tf2_ros


class SensorCalibrator(Node):
    """
    Calibrates simulated sensors to match real-world characteristics.
    """

    def __init__(self):
        super().__init__('sensor_calibrator')

        # IMU calibration parameters
        self.imu_bias = np.array([0.01, -0.02, 0.005])  # Bias in m/s²
        self.imu_scale = np.array([0.99, 1.01, 1.005])  # Scale factors
        self.imu_noise_std = 0.01  # Noise standard deviation

        # Camera calibration parameters
        self.camera_matrix = np.array([
            [500, 0, 320],  # fx, 0, cx
            [0, 500, 240],  # 0, fy, cy
            [0, 0, 1]       # 0, 0, 1
        ])

        # Create TF broadcaster for sensor transforms
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Subscribe to raw sensor data
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/raw',
            self.imu_calibration_callback,
            10
        )

        self.get_logger().info('Sensor calibrator initialized')

    def add_calibration_to_imu(self, raw_imu_msg):
        """
        Apply calibration parameters to IMU data.
        """
        calibrated_imu = Imu()
        calibrated_imu.header = raw_imu_msg.header

        # Apply bias and scale to linear acceleration
        raw_acc = np.array([
            raw_imu_msg.linear_acceleration.x,
            raw_imu_msg.linear_acceleration.y,
            raw_imu_msg.linear_acceleration.z
        ])

        calibrated_acc = (raw_acc + self.imu_bias) * self.imu_scale

        # Add noise
        noise = np.random.normal(0, self.imu_noise_std, 3)
        calibrated_acc += noise

        calibrated_imu.linear_acceleration.x = calibrated_acc[0]
        calibrated_imu.linear_acceleration.y = calibrated_acc[1]
        calibrated_imu.linear_acceleration.z = calibrated_acc[2]

        # Apply similar calibration to angular velocity
        raw_ang_vel = np.array([
            raw_imu_msg.angular_velocity.x,
            raw_imu_msg.angular_velocity.y,
            raw_imu_msg.angular_velocity.z
        ])

        calibrated_ang_vel = raw_ang_vel * 1.01  # Scale factor
        calibrated_ang_vel += np.random.normal(0, 0.001, 3)  # Noise

        calibrated_imu.angular_velocity.x = calibrated_ang_vel[0]
        calibrated_imu.angular_velocity.y = calibrated_ang_vel[1]
        calibrated_imu.angular_velocity.z = calibrated_ang_vel[2]

        # Copy orientation (assuming perfect estimation for simulation)
        calibrated_imu.orientation = raw_imu_msg.orientation

        return calibrated_imu
```

## Realistic Sensor Noise Models

### Noise Characteristics

Different sensors have different noise characteristics:

```python
import numpy as np


class SensorNoiseModels:
    """
    Various sensor noise models for realistic simulation.
    """

    @staticmethod
    def gaussian_noise(mean=0.0, std=1.0, size=1):
        """Standard Gaussian noise model."""
        return np.random.normal(mean, std, size)

    @staticmethod
    def quantization_noise(resolution):
        """Quantization noise for discrete sensors."""
        return np.random.uniform(-resolution/2, resolution/2)

    @staticmethod
    def drift_noise(initial_bias, drift_rate, time_elapsed):
        """Drift noise that accumulates over time."""
        drift = drift_rate * time_elapsed
        return initial_bias + np.random.normal(0, 0.1) + drift

    @staticmethod
    def bias_instability_noise(time_constant, gain):
        """Bias instability following first-order Gauss-Markov process."""
        # Implementation of bias instability model
        pass

    @staticmethod
    def camera_noise(image, photon_noise_std=0.01, readout_noise_std=0.005):
        """Add realistic noise to camera images."""
        # Photon noise (proportional to signal)
        photon_noise = np.random.normal(0, photon_noise_std * np.sqrt(np.abs(image)))

        # Readout noise (constant)
        readout_noise = np.random.normal(0, readout_noise_std, image.shape)

        noisy_image = image + photon_noise + readout_noise
        return np.clip(noisy_image, 0, 1)  # Clamp to valid range
```

## Exercise: Complete Sensor Suite for Humanoid Robot

Create a complete sensor configuration for a humanoid robot that includes:
1. LiDAR on the head with 360° horizontal scan
2. Stereo camera pair for depth perception
3. IMU in the torso for balance control
4. Force/torque sensors in the feet
5. Joint position sensors for all DOFs

## Learning Outcomes

After completing this section, students will be able to:
1. Configure LiDAR sensors with proper ray tracing parameters
2. Set up camera and depth camera simulation with noise models
3. Implement IMU simulation with realistic noise characteristics
4. Add force/torque sensors for manipulation
5. Apply sensor calibration to match real-world characteristics
6. Implement sensor fusion techniques

## Next Steps

Complete Module 2 by reviewing all simulation concepts and practicing with the exercises. Then continue to [Module 3: The AI-Robot Brain (NVIDIA Isaac)](../module-3-nvidia-isaac/intro.md) to learn about AI-powered robotics with NVIDIA Isaac platform.