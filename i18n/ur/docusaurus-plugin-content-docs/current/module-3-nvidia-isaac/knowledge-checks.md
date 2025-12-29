# Knowledge Checks (اردو میں)

## Isaac Sim Fundamentals

### سوال 1: USD اور Scene Structure
**Difficulty: Beginner**

Isaac Sim میں USD (Universal Scene Description) کے بارے میں مندرجہ ذیل میں سے کون سا statement TRUE ہے؟

A) USD صرف robot kinematic chains store کرنے کے لیے use ہوتا ہے
B) USD references اور payloads کے ساتھ 3D scenes کی layered composition support کرتا ہے
C) USD files میں صرف geometric data ہو سکتی ہے، material properties نہیں
D) USD Isaac Sim کے physics engine کے ساتھ incompatible ہے

**Answer:** B) USD references اور payloads کے ساتھ 3D scenes کی layered composition support کرتا ہے

**Explanation:** USD (Universal Scene Description) Isaac Sim scenes کا underlying format ہے۔ یہ support کرتا ہے:
- Layered composition جس میں multiple scene components combine ہو سکتے ہیں
- References اور payloads efficient asset management کے لیے
- Geometric data, materials, animations, اور physics properties
- Complex scenes کا scalable representation

### سوال 2: Physics Configuration
**Difficulty: Intermediate**

Isaac Sim میں realistic humanoid robot simulation achieve کرنے کے لیے کون سے parameters MOST critical ہیں؟

A) Render resolution اور anti-aliasing settings
B) Collision mesh complexity اور contact material properties
C) Lighting models اور texture resolution
D) Audio properties اور sound effects

**Answer:** B) Collision mesh complexity اور contact material properties

**Explanation:** Realistic humanoid robot simulation کے لیے physics parameters crucial ہیں:
- Collision meshes determine کرتی ہیں کہ robot environment کے ساتھ کیسے interact کرتا ہے
- Contact material properties (friction, restitution) walking stability کو affect کرتی ہیں
- Proper contact modeling bipedal locomotion کے لیے essential ہے
- Physics parameters directly sim-to-real transfer کو impact کرتی ہیں

### سوال 3: GPU Acceleration Benefits
**Difficulty: Advanced**

Isaac Sim میں کون سی features GPU acceleration سے MOST benefit حاصل کرتی ہیں؟

A) صرف rendering اور visual effects
B) Physics simulation, rendering, اور AI inference
C) صرف sensor data processing
D) صرف user interface اور controls

**Answer:** B) Physics simulation, rendering, اور AI inference

**Explanation:** Isaac Sim GPU acceleration multiple domains میں leverage کرتا ہے:
- Rendering: High-fidelity photorealistic graphics
- Physics: Collision detection اور response کے لیے PhysX engine
- AI Inference: Neural network acceleration کے لیے TensorRT integration
- Sensor Simulation: Depth اور image processing کے لیے GPU-accelerated

## Isaac ROS Integration

### سوال 4: Isaac ROS Packages
**Difficulty: Intermediate**

Visual SLAM (Simultaneous Localization and Mapping) کے لیے کون سا Isaac ROS package BEST ہے؟

A) Isaac ROS AprilTag
B) Isaac ROS Stereo DNN
C) Isaac ROS Visual SLAM
D) Isaac ROS Image Pipeline

**Answer:** C) Isaac ROS Visual SLAM

**Explanation:** Isaac ROS Visual SLAM specifically designed ہے:
- Cameras اور IMU data use کرتے ے visual-inertial SLAM کے لیے
- Real-time performance کے لیے GPU-accelerated processing
- Accurate localization اور mapping
- ROS navigation stack کے ساتھ integration

### سوال 5: Sensor Simulation
**Difficulty: Intermediate**

Isaac Sim sensor noise modeling کیسے handle کرتا ہے؟

A) Noise post-processing میں user manually add کرتا ہے
B) Isaac Sim automatically realistic noise models apply کرتا ہے based on sensor specifications
C) Isaac Sim sensor noise modeling support نہیں کرتا
D) Noise modeling کے لیے external simulation tools required ہیں

**Answer:** B) Isaac Sim automatically realistic noise models apply کرتا ہے based on sensor specifications

**Explanation:** Isaac Sim provide کرتا ہے:
- Various sensors کے لیے built-in noise models (cameras, LiDAR, IMU)
- Real sensor specifications based configurable noise parameters
- Simulation کے دوران noise کا automatic application
- Gaussian noise, bias, اور drift modeling support

### سوال 6: ROS Bridge Performance
**Difficulty: Advanced**

Traditional ROS sensor plugins کے compared، Isaac ROS bridges کا PRIMARY advantage کیا ہے؟

A) Lower memory consumption
B) GPU-accelerated processing اور optimized data pipelines
C) Simpler installation process
D) More sensor types support

**Answer:** B) GPU-accelerated processing اور optimized data pipelines

**Explanation:** Isaac ROS bridges offer کرتے ہیں:
- AI workloads کے لیے GPU-accelerated processing
- CPU overhead reduce کرتی optimized data pipelines
- Isaac Sim کے rendering اور physics کے ساتھ direct integration
- Perception algorithms کے لیے hardware acceleration

## Synthetic Data Generation

### سوال 7: Domain Randomization
**Difficulty: Intermediate**

Synthetic data generation میں domain randomization کا MAIN purpose کیا ہے؟

A) Synthetic data کو visually appealing بنانے کے لیے
B) AI models کی sim-to-real transfer improve کرنے کے لیے
C) Simulation کی computational cost reduce کرنے کے لیے
D) Real-world data کی ضرورت eliminate کرنے کے لیے

**Answer:** B) AI models کی sim-to-real transfer improve کرنے کے لیے

**Explanation:** Domain randomization:
- Environmental parameters randomize کرتا ہے (lighting, textures, physics)
- AI models کو invariant features سیکھنے پر مجبور کرتا ہے
- Sim اور real کے درمیان domain shift کے لیے robustness improve کرتا ہے
- Real-world data پر better performance enable کرتا ہے

### سوال 8: Annotation Generation
**Difficulty: Advanced**

کون سی types کی ground truth annotations Isaac Sim automatically generate کر سکتا ہے؟

A) صرف RGB images
B) RGB, depth, segmentation, اور 3D bounding boxes
C) صرف depth اور RGB data
D) صرف semantic segmentation masks

**Answer:** B) RGB, depth, segmentation, اور 3D bounding boxes

**Explanation:** Isaac Sim automatically generate کر سکता ہے:
- Virtual cameras سے RGB images
- Depth sensors سے depth maps
- Instance segmentation masks
- Semantic segmentation masks
- 2D اور 3D bounding boxes
- Objects کے لیے pose annotations

### سوال 9: Synthetic Data Pipeline
**Difficulty: Advanced**

Synthetic data pipeline create کرنے کا CORRECT sequence کیا ہے؟

A) Randomize → Capture → Label → Train
B) Design → Implement → Deploy → Monitor
C) Model → Simulate → Test → Validate
D) Plan → Build → Integrate → Optimize

**Answer:** A) Randomize → Capture → Label → Train

**Explanation:** Synthetic data pipeline follow کرتا ہے:
1. Randomize: Domain randomization set up کریں
2. Capture: Images اور sensor data generate کریں
3. Label: Ground truth annotations automatically create کریں
4. Train: AI models train کرنے کے لیے data use کریں

## Navigation and Planning

### سوال 10: Bipedal Navigation
**Difficulty: Advanced**

Wheeled robots کے compared، humanoid robot navigation میں PRIMARY challenge کیا ہے؟

A) Higher computational requirements
B) Moving اور turning کے دوران balance maintain کرنا
C) Limited sensor options
D) Shorter operational time

**Answer:** B) Moving اور turning کے دوران balance maintain کرنا

**Explanation:** Humanoid navigation challenges include:
- Locomotion کے دوران balance maintenance
- Turns کے دوران center of mass control
- Stable gait کے لیے footstep planning
- Stability maintain کرتے ے dynamic obstacle avoidance

### سوال 11: GPU Path Planning
**Difficulty: Intermediate**

GPU-accelerated path planning humanoid robots کو کیسے benefit دیتا ہے؟

A) Dynamic environments کے لیے faster path computation
B) Path visualization کا better rendering
C) Improved sensor data processing
D) Enhanced robot kinematics

**Answer:** A) Dynamic environments کے لیے faster path computation

**Explanation:** GPU-accelerated path planning provide کرتा ہے:
- Complex paths کا faster computation
- Dynamic environments میں real-time replanning
- Complex terrain navigation support
- Improved obstacle avoidance

### سوال 12: Isaac Navigation Stack
**Difficulty: Advanced**

Humanoid robots کے ساتھ work کرنے کے لیے Isaac ROS navigation میں کون سا component ESSENTIAL ہے؟

A) Wheel odometry publisher
B) Bipedal locomotion کے لیے footstep planner
C) Positioning کے لیے GPS sensor
D) External motion capture system

**Answer:** B) Bipedal locomotion کے لیے footstep planner

**Explanation:** Humanoid navigation کے لیے:
- Footstep planner stable stepping sequences generate کرتا ہے
- Bipedal locomotion planning کے لیے essential ہے
- Balance control systems کے ساتھ coordinate کرتا ہے
- Safe navigation کے لیے path planning کے ساتھ integrate ہوتا ہے

## Perception Systems

### سوال 13: Isaac Perception Pipelines
**Difficulty: Intermediate**

Real-time object detection کے لیے کون سا Isaac ROS package BEST ہے؟

A) Isaac ROS Apriltag
B) Isaac ROS DetectNet
C) Isaac ROS Stereo DNN
D) Isaac ROS Image Pipeline

**Answer:** B) Isaac ROS DetectNet

**Explanation:** Isaac ROS DetectNet designed ہے:
- Deep learning use کرتے ے real-time object detection
- GPU-accelerated inference
- Various detection models support
- Isaac Sim synthetic data کے ساتھ integration

### سوال 14: Multi-Sensor Fusion
**Difficulty: Advanced**

Isaac Sim میں multi-sensor fusion کا PRIMARY advantage کیا ہے؟

A) Reduced computational requirements
B) Complementary sensors کے ذریعے improved robustness اور accuracy
C) Simpler system architecture
D) Lower hardware costs

**Answer:** B) Complementary sensors کے ذریعے improved robustness اور accuracy

**Explanation:** Multi-sensor fusion provide کرتा ہے:
- Individual sensors fail ہونے پر redundancy
- Different modalities سے complementary information
- Sensor fusion کے ذریعے improved accuracy
- Challenging conditions میں robust perception

### سوال 15: AI Model Training
**Difficulty: Advanced**

Isaac Sim سے synthetic data AI model training کو کیسے impact کرتا ہے؟

A) یہ کسی بھی real data کی ضرورت eliminate کرتا ہے
B) یہ diverse, labeled training data provide کرتا ہے جو generalization improve کرتا ہے
C) یہ real-world variations کے لیے models کو کم robust بناتا ہے
D) یہ training time کو significantly increase کرتا ہے

**Answer:** B) یہ diverse, labeled training data provide کرتا ہے جو generalization improve کرتا ہے

**Explanation:** Synthetic data benefits:
- Large-scale, diverse training datasets
- Perfect ground truth annotations
- Robustness کے لیے domain randomization
- Cost-effective data generation
- Improved sim-to-real transfer

## Practical Application Questions

### سوال 16: System Integration
**Difficulty: Advanced**

Isaac Sim کو ROS 2 کے ساتھ integrate کرتے وقت MOST important consideration کیا ہے؟

A) Simulation اور ROS nodes کے درمیان network bandwidth
B) Simulation time کا ROS time کے ساتھ synchronization
C) Simulation اور real cameras کے درمیان color matching
D) Sound reproduction quality

**Answer:** B) Simulation time کا ROS time کے ساتھ synchronization

**Explanation:** Time synchronization critical ہے کیونکہ:
- ROS nodes synchronized timestamps expect کرتے ہیں
- Sensor data کو properly timed ہونا چاہیے
- Control commands کو accurate timing chahiye
- Simulation-real world synchronization

### سوال 17: Performance Optimization
**Difficulty: Intermediate**

Real-time applications کے لیے Isaac Sim performance optimize کرنے کا BEST approach کیا ہے؟

A) Visual quality settings maximize کریں
B) Physics update rates کے ساتھ rendering quality balance کریں
C) All physics simulation disable کریں
D) All models کے لیۏ maximum polygon counts use کریں

**Answer:** B) Physics update rates کے ساتھ rendering quality balance کریں

**Explanation:** Performance optimization involves:
- Control requirements کے matching physics update rate (balance کے لیے 200Hz)
- Target frame rate کے لیے rendering quality adjust کرنا
- Scene complexity optimize کرنا
- Level-of-detail (LOD) systems use کرنا

### سوال 18: Hardware Requirements
**Difficulty: Beginner**

Isaac Sim کے لیے MINIMUM GPU requirement کیا ہے؟

A) Integrated graphics chip
B) کوئی بھی NVIDIA GPU CUDA support کے ساتھ
C) AMD Radeon graphics card
D) Intel integrated graphics

**Answer:** B) کوئی بھی NVIDIA GPU CUDA support کے ساتھ

**Explanation:** Isaac Sim require کرتا ہے:
- CUDA acceleration کے لیے NVIDIA GPU
- Optimal performance کے لیے dedicated GPU
- Advanced features کے لیے RTX series recommended
- Rendering اور physics کے لیے GPU memory

## Scenario-Based Questions

### سوال 19: Navigation Challenge
**Difficulty: Advanced**

Isaac Sim میں ایک humanoid robot cluttered environment میں navigation سے struggling ہے۔ Performance improve کرنے کا FIRST step کیا ہوگا؟

A) Robot کی walking speed increase کریں
B) More waypoints کے ساتھ global path planner improve کریں
C) Dynamic obstacle avoidance کے لیے local planner enhance کریں
D) Robot میں more cameras add کریں

**Answer:** C) Dynamic obstacle avoidance کے لیے local planner enhance کریں

**Explanation:** Cluttered environments میں navigation کے لیے:
- Local planner dynamic obstacle handling کرتا ہے
- Real-time path adjustment کے لیے essential ہے
- Humanoid robot safety کے لیے critical ہے
- Global planner کے ساتھ conjunction میں work کرنا چاہیے

### سوال 20: Perception Improvement
**Difficulty:Advanced**

Your Isaac Sim-trained perception model real robot data پر poorly perform کرتا ہے۔ MOST likely cause کیا ہے؟

A) Insufficient training data quantity
B) Synthetic data generation کے دوران domain randomization کی کمی
C) Training کے دوران too many epochs
D) Wrong neural network architecture

**Answer:** B) Synthetic data generation کے دوران domain randomization کی کمی

**Explanation:** Poor sim-to-real transfer often results from:
- Insufficient domain randomization
- Synthetic data میں limited variation
- Simulation اور reality کے درمیان mismatch
- Training میں environmental diversity کی کمی

## Answer Key Summary

1. B - USD supports layered composition
2. B - Physics parameters are critical for humanoid simulation
3. B - GPU acceleration across multiple domains
4. C - Isaac ROS Visual SLAM for SLAM
5. B - Isaac Sim provides automatic noise modeling
6. B - GPU acceleration and optimized pipelines
7. B - Improve sim-to-real transfer
8. B - Multiple annotation types available
9. A - Randomize → Capture → Label → Train
10. B - Balance maintenance is primary challenge
11. A - Faster computation for dynamic environments
12. B - Footstep planner for bipedal locomotion
13. B - Isaac ROS DetectNet for object detection
14. B - Improved robustness through fusion
15. B - Diverse, labeled data improves generalization
16. B - Time synchronization is critical
17. B - Balance quality with performance
18. B - NVIDIA GPU with CUDA support
19. C - Enhance local planner for obstacle avoidance
20. B - Lack of domain randomization causes poor transfer

## Self-Assessment Checklist

### Isaac Sim Fundamentals
- [ ] USD scene structure اور composition سمجڹیں
- [ ] Physics properties configure کرنا جانتے ہیں
- [ ] GPU acceleration benefits سمجڹیں
- [ ] Basic simulation scenes create کر سکتے ہیں

### Isaac ROS Integration
- [ ] Isaac ROS package ecosystem سمجڹیں
- [ ] Sensor bridges configure کرنا جانتے ہیں
- [ ] Navigation stack integration سمجڹیں
- [ ] Common integration issues troubleshoot کر سکتے ہیں

### Synthetic Data Generation
- [ ] Domain randomization concepts سمجڹیں
- [ ] Various annotation types generate کرنا جانتے ہیں
- [ ] Sim-to-real transfer techniques سمجڹیں
- [ ] Synthetic data pipelines create کر سکتے ہیں

### Navigation and Perception
- [ ] Humanoid-specific navigation challenges سمجڹیں
- [ ] Perception systems configure کرنا جانتے ہیں
- [ ] Multi-sensor fusion سمجڹیں
- [ ] Real-time performance optimize کر سکتے ہیں

## Performance Benchmarks

### Beginner Level (70-80% correct)
- Basic Isaac Sim concepts سمجڹیں
- Simple scenes create کر سکتے ہیں
- Fundamental ROS integration جانتے ہیں

### Intermediate Level (80-90% correct)
- Complex simulations configure کر سکتے ہیں
- Perception system design سمجڹیں
- Navigation pipeline setup جانتے ہیں

### Advanced Level (90-100% correct)
- Synthetic data generation میں مہارت
- Complex systems optimize کر سکتے ہیں
- Sim-to-real transfer challenges سمجڹیں
- Performance issues troubleshoot کر سکتے ہیں

## اگلے steps

ان knowledge checks کو مکمل کرنے کے بعد، students یہ کر سکیں گے:
1. Isaac Sim concepts کی سمجھ assess کریں
2. Areas identify کریں جنہیں مزید study کی ضرورت ہے
3. Practical projects پر knowledge apply کریں
4. Advanced Isaac Sim applications کے لیے تیار ہوں

[Summary and Next Steps](./summary-next-steps.md) پر جائیں تاکہ key concepts review کر سکیں اور future learning plan کر سکیں۔
