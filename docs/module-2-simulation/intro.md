# Introduction to Simulation Environments

## Why Simulate Before Deploying?

Robot simulation is a critical component of modern robotics development. It allows us to:

- **Test Safely**: Evaluate dangerous scenarios without hardware risk
- **Iterate Rapidly**: Run experiments faster than real-time
- **Reduce Costs**: Minimize wear on physical robots during development
- **Ensure Reproducibility**: Create identical test conditions
- **Enable Remote Development**: Work without physical robot access

## The Digital Twin Concept

A digital twin is a virtual replica of a physical robot that mirrors its geometry, dynamics, and behavior. This enables:

- **Parallel Development**: Software development while hardware is being built
- **Validation**: Test algorithms in simulation before physical deployment
- **Training**: Train machine learning models in safe virtual environments
- **Troubleshooting**: Debug issues in simulation before physical testing

```python
class DigitalTwin:
    """
    Digital twin that mirrors a physical humanoid robot.
    """

    def __init__(self, physical_robot):
        self.physical_robot = physical_robot
        self.simulation = SimulationEnvironment()
        self.state_estimator = StateEstimator()

    def sync_from_physical(self):
        """Synchronize simulation state from physical robot."""
        sensor_data = self.physical_robot.get_sensor_data()
        self.simulation.set_state(sensor_data)

    def run_simulation(self, duration):
        """Run simulation for specified duration."""
        self.simulation.step(duration)
        return self.simulation.get_state()

    def validate_software(self):
        """Validate software on simulation before physical deployment."""
        self.sync_from_physical()
        results = self.run_software_stack()
        return self.software_passes_validation(results)
```

## Simulation-to-Real Transfer

The gap between simulation and reality is called the "sim-to-real gap." Key challenges include:

### Physics Discrepancies
- Friction models don't perfectly match reality
- Contact dynamics are hard to simulate accurately
- Motor responses differ from simulation

### Sensor Noise
- Real sensors have noise that simulation may not capture
- Lighting conditions vary from simulation
- Sensor calibration differs

### Domain Randomization

One effective technique is domain randomization:

```python
def randomize_domain(env):
    """Apply domain randomization to simulation."""
    # Randomize textures
    env.randomize_textures()

    # Randomize lighting
    env.randomize_lighting(intensity_range=(0.5, 1.5))

    # Randomize physics parameters
    env.randomize_friction(mu_range=(0.3, 0.8))
    env.randomize_mass(mass_range=(0.9, 1.1))

    # Randomize sensor noise
    env.randomize_sensor_noise(noise_level_range=(0.0, 0.1))
```

## Simulation Platforms Comparison

| Platform | Strengths | Weaknesses | Use Case |
|----------|-----------|------------|----------|
| Gazebo | Physics accuracy, ROS integration | Graphics quality | Control, navigation |
| Unity | High-fidelity graphics, ML integration | Physics less accurate | Perception, HRI |
| Isaac Sim | Photorealistic, synthetic data | Complex setup | Perception, AI |
| PyBullet | Lightweight, fast | Less realistic | Prototyping, learning |

## Learning Outcomes

After completing this module, students will be able to:
1. Understand the importance of simulation in robotics
2. Set up and configure Gazebo simulation environments
3. Create URDF/SDF models for humanoid robots in simulation
4. Simulate sensors including LiDAR, cameras, and IMUs
5. Use Unity for high-fidelity robot visualization
6. Apply simulation-to-real transfer techniques

## Prerequisites

- Completion of Module 1 (ROS 2 Fundamentals)
- Understanding of URDF robot descriptions
- Basic physics concepts

## Estimated Time

2 weeks (Weeks 6-7 of the course)

## Next Steps

Continue to [Physics Simulation](./physics-simulation.md) to learn about collision detection, gravity, and friction in simulation environments.