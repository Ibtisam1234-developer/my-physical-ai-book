# Physics Simulation

## Understanding Physics in Simulation

Physics simulation is the foundation of realistic robot simulation. It encompasses:

- **Collision Detection**: Determining when objects intersect
- **Gravity**: Simulating gravitational forces
- **Friction**: Modeling contact forces between surfaces
- **Dynamics**: Computing motion based on forces and torques
- **Contact Models**: Handling interactions between bodies

## Gazebo Physics Engine

Gazebo uses ODE (Open Dynamics Engine) as its default physics engine. The physics configuration controls how objects behave in the simulation.

### Physics Configuration

```xml
<?xml version="1.0"?>
<sdf version="1.8">
    <world name="humanoid_lab">
        <!-- Physics Plugin -->
        <physics name="physics" type="ode">
            <ode>
                <solver>
                    <type>quick</type>
                    <iters>100</iters>        <!-- Iterations for constraint solving -->
                    <sor>1.0</sor>            <!-- Successive Over Relaxation parameter -->
                    <friction_model>cone_model</friction_model>
                </solver>
                <constraints>
                    <cfm>0.0</cfm>            <!-- Constraint Force Mixing -->
                    <erp>0.2</erp>            <!-- Error Reduction Parameter -->
                    <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
                    <contact_surface_layer>0.001</contact_surface_layer>
                </constraints>
            </ode>
            <max_step_size>0.001</max_step_size>      <!-- Simulation timestep -->
            <real_time_factor>1.0</real_time_factor>  <!-- Speed relative to real time -->
            <real_time_update_rate>1000</real_time_update_rate>
        </physics>

        <!-- Gravity -->
        <gravity>0 0 -9.81</gravity>

        <!-- Ground Plane -->
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
                    <surface>
                        <friction>
                            <ode>
                                <mu>0.8</mu>      <!-- Primary friction coefficient -->
                                <mu2>0.8</mu2>    <!-- Secondary friction coefficient -->
                            </ode>
                        </friction>
                    </surface>
                </collision>
                <visual name="visual">
                    <geometry>
                        <plane>
                            <normal>0 0 1</normal>
                            <size>100 100</size>
                        </plane>
                    </geometry>
                    <material>
                        <script>
                            <uri>file://media/materials/scripts/gazebo.material</uri>
                            <name>Gazebo/Gray</name>
                        </script>
                    </material>
                </visual>
            </link>
        </model>
    </world>
</sdf>
```

## Collision Detection

Collision detection determines when two objects intersect. Gazebo supports various collision shapes:

### Collision Shapes

```xml
<!-- Box collision -->
<collision name="box_collision">
    <geometry>
        <box>
            <size>0.5 0.3 0.2</size>
        </box>
    </geometry>
</collision>

<!-- Cylinder collision -->
<collision name="cylinder_collision">
    <geometry>
        <cylinder>
            <radius>0.1</radius>
            <length>0.5</length>
        </cylinder>
    </geometry>
</collision>

<!-- Sphere collision -->
<collision name="sphere_collision">
    <geometry>
        <sphere>
            <radius>0.1</radius>
        </sphere>
    </geometry>
</collision>

<!-- Mesh collision -->
<collision name="mesh_collision">
    <geometry>
        <mesh>
            <uri>model://humanoid/meshes/complex_shape.stl</uri>
        </mesh>
    </geometry>
</collision>
```

### Surface Properties

Surface properties control how objects interact during contact:

```xml
<collision name="collision">
    <geometry>
        <box size="0.2 0.2 0.2"/>
    </geometry>
    <surface>
        <!-- Friction properties -->
        <friction>
            <ode>
                <mu>0.8</mu>              <!-- Primary friction coefficient -->
                <mu2>0.8</mu2>            <!-- Secondary friction coefficient -->
                <fdir1>0 1 0</fdir1>      <!-- Friction direction -->
                <slip1>0</slip1>          <!-- Primary slip coefficient -->
                <slip2>0</slip2>          <!-- Secondary slip coefficient -->
            </ode>
        </friction>

        <!-- Bounce properties -->
        <bounce>
            <restitution>0.0</restitution>  <!-- Bounciness (0 = no bounce) -->
            <threshold>100</threshold>      <!-- Velocity threshold for bouncing -->
        </bounce>

        <!-- Contact properties -->
        <contact>
            <collide_without_contact>false</collide_without_contact>
            <collide_without_contact_bitmask>1</collide_without_contact_bitmask>
            <collide_bitmask>1</collide_bitmask>
            <ode>
                <soft_cfm>0.01</soft_cfm>      <!-- Soft Constraint Force Mixing -->
                <soft_erp>0.2</soft_erp>       <!-- Soft Error Reduction Parameter -->
                <kp>1e8</kp>                  <!-- Spring stiffness -->
                <kd>1e6</kd>                  <!-- Damping coefficient -->
                <max_vel>0.01</max_vel>        <!-- Maximum contact correction velocity -->
                <min_depth>0.001</min_depth>   <!-- Minimum contact depth -->
            </ode>
        </contact>
    </surface>
</collision>
```

## Gravity and Forces

Gravity is a fundamental force in physics simulation:

```xml
<!-- Standard Earth gravity -->
<gravity>0 0 -9.81</gravity>

<!-- Moon gravity (about 1/6 of Earth) -->
<gravity>0 0 -1.62</gravity>

<!-- Zero gravity (for space robotics) -->
<gravity>0 0 0</gravity>

<!-- Custom gravity direction (for tilted surfaces) -->
<gravity>-1.0 0 -9.81</gravity>
```

## Humanoid-Specific Physics Considerations

For humanoid robots, physics simulation has special requirements:

### Balance and Stability

```xml
<!-- High-friction feet for stable walking -->
<link name="left_foot">
    <collision name="collision">
        <geometry>
            <box size="0.12 0.04 0.2"/>
        </geometry>
        <surface>
            <friction>
                <ode>
                    <mu>1.0</mu>      <!-- High friction for walking stability -->
                    <mu2>1.0</mu2>
                </ode>
            </friction>
        </surface>
    </collision>
</link>
```

### Joint Limits and Dynamics

```xml
<!-- Hip joint with appropriate limits for humanoid movement -->
<joint name="left_hip_pitch" type="revolute">
    <parent link="pelvis"/>
    <child link="left_upper_leg"/>
    <origin xyz="0 -0.1 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="5"/>
    <dynamics damping="0.5" friction="0.1"/>
</joint>
```

## Tuning Physics Parameters

### Timestep Considerations

```xml
<!-- Physics for high-precision control -->
<physics name="high_precision" type="ode">
    <max_step_size>0.0005</max_step_size>      <!-- 2kHz simulation -->
    <real_time_update_rate>2000</real_time_update_rate>
</physics>

<!-- Physics for fast simulation -->
<physics name="fast_simulation" type="ode">
    <max_step_size>0.01</max_step_size>        <!-- 100Hz simulation -->
    <real_time_update_rate>100</real_time_update_rate>
</physics>
```

### Solver Parameters

```xml
<!-- Accurate solver (slower but more stable) -->
<physics name="accurate_solver" type="ode">
    <ode>
        <solver>
            <type>quick</type>
            <iters>200</iters>      <!-- More iterations for accuracy -->
            <sor>1.0</sor>
        </solver>
        <constraints>
            <cfm>0.0</cfm>
            <erp>0.1</erp>          <!-- Lower ERP for more constraint accuracy -->
        </constraints>
    </ode>
</physics>

<!-- Fast solver (less accurate but faster) -->
<physics name="fast_solver" type="ode">
    <ode>
        <solver>
            <type>quick</type>
            <iters>50</iters>       <!-- Fewer iterations for speed -->
            <sor>1.3</sor>
        </solver>
        <constraints>
            <cfm>0.01</cfm>         <!-- Higher CFM for stability -->
            <erp>0.3</erp>
        </constraints>
    </ode>
</physics>
```

## Physics Debugging and Visualization

### Checking Physics Performance

```python
#!/usr/bin/env python3
"""
Example: Physics performance monitoring.
"""

import rclpy
from rclpy.node import Node
import time


class PhysicsMonitor(Node):
    """
    Monitors physics simulation performance.
    """

    def __init__(self):
        super().__init__('physics_monitor')

        # Performance tracking
        self.simulation_times = []
        self.real_times = []
        self.timing_errors = []

        # Timer for monitoring
        self.monitor_timer = self.create_timer(1.0, self.check_performance)

    def check_performance(self):
        """Check physics simulation performance."""
        if len(self.simulation_times) > 10:
            avg_real_time = sum(self.real_times[-10:]) / len(self.real_times[-10:])
            avg_sim_time = sum(self.simulation_times[-10:]) / len(self.simulation_times[-10:])

            real_time_factor = avg_sim_time / avg_real_time if avg_real_time > 0 else 0

            if real_time_factor < 0.8:
                self.get_logger().warn(f'Physics running slow: RTF={real_time_factor:.2f}')
            elif real_time_factor > 1.2:
                self.get_logger().info(f'Physics running fast: RTF={real_time_factor:.2f}')
            else:
                self.get_logger().info(f'Physics performance OK: RTF={real_time_factor:.2f}')
```

## Exercise: Configure Physics for Humanoid Robot

Create a Gazebo world file with:
1. Appropriate physics parameters for humanoid simulation
2. High-friction ground plane for stable walking
3. Proper joint limits and dynamics for humanoid joints
4. Collision properties optimized for humanoid contact

## Learning Outcomes

After completing this section, students will be able to:
1. Configure physics parameters for different simulation requirements
2. Set up collision detection for humanoid robots
3. Tune friction and contact properties for realistic behavior
4. Optimize physics simulation for performance
5. Monitor and debug physics simulation performance

## Next Steps

Continue to [Unity Rendering for HRI](./unity-rendering.md) to learn about high-fidelity rendering and Human-Robot Interaction in Unity.