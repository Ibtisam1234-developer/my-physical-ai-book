# VLA Models and Architectures

## Understanding VLA System Architectures

Vision-Language-Action (VLA) systems represent the integration of three critical AI modalities for embodied intelligence. These systems enable robots to perceive their environment, understand natural language commands, and execute appropriate physical actions in response.

### The VLA Paradigm

VLA systems differ from traditional robotics approaches by creating unified representations that allow vision, language, and action to influence each other bidirectionally:

- **Vision informs Language**: Visual context clarifies ambiguous language commands
- **Language guides Vision**: Language specifies what to attend to visually
- **Action affects Perception**: Actions change the robot's viewpoint and environment
- **Perception guides Action**: Sensory input determines appropriate actions

### Architectural Approaches

There are several architectural patterns for VLA systems:

#### 1. Sequential Pipeline Architecture
```
Vision → Language Understanding → Action Generation
```
- **Pros**: Simple to implement, modular components
- **Cons**: Error propagation, no bidirectional influence
- **Use Case**: Basic command execution, simple environments

#### 2. Late Fusion Architecture
```
Vision → Feature Extraction → Fusion → Action Generation
            ↑                    ↓
Language → Feature Extraction ----
```
- **Pros**: Shared representation, some cross-modal influence
- **Cons**: Limited early integration, potential information loss
- **Use Case**: Moderate complexity tasks, resource-constrained systems

#### 3. Early Fusion Architecture
```
Vision + Language → Joint Encoding → Action Generation
```
- **Pros**: Full cross-modal interaction, unified representation
- **Cons**: Complex architecture, harder to optimize
- **Use Case**: Complex reasoning tasks, natural interaction

#### 4. Transformer-Based Architecture
```
Multi-Modal Transformer (Vision + Language + Action tokens)
                    ↓
           End-to-End Generation
```
- **Pros**: State-of-the-art performance, attention mechanisms
- **Cons**: Large model size, significant compute requirements
- **Use Case**: Advanced robotics, research applications

## Leading VLA Models

### RT-2 (Robotics Transformer 2)

Developed by Google DeepMind, RT-2 represents a foundational approach to VLA systems:

```python
class RT2Model:
    """
    Implementation of Robotics Transformer 2 architecture.
    """

    def __init__(self, vision_encoder, language_model, action_head):
        self.vision_encoder = vision_encoder  # CLIP visual encoder
        self.language_model = language_model  # Large language model
        self.action_head = action_head        # Action token prediction head

    def forward(self, images, instructions):
        """
        Forward pass through RT-2 model.

        Args:
            images: Visual observations [B, C, H, W]
            instructions: Natural language instructions [B, seq_len]

        Returns:
            action_tokens: Predicted action tokens for robot execution
        """
        # Encode visual features
        visual_features = self.vision_encoder(images)

        # Encode language features
        language_features = self.language_model.encode(instructions)

        # Fuse modalities
        fused_features = self.fuse_modalities(visual_features, language_features)

        # Generate actions
        action_tokens = self.action_head(fused_features)

        return action_tokens

    def fuse_modalities(self, visual_features, language_features):
        """
        Fuse vision and language features using cross-attention.
        """
        # Cross-attention between vision and language
        lang_attended_vis = self.vision_language_attention(
            query=language_features,
            key=visual_features,
            value=visual_features
        )

        # Concatenate and process
        combined_features = torch.cat([language_features, lang_attended_vis], dim=-1)
        fused_features = self.fusion_network(combined_features)

        return fused_features
```

### PaLM-E (Embodied Multimodal Language Model)

PaLM-E extends language models to embodied environments:

```python
class PalmEModel:
    """
    PaLM-E: Embodied multimodal language model.
    """

    def __init__(self, base_lm, vision_encoder):
        self.base_language_model = base_lm
        self.vision_encoder = vision_encoder
        self.perception_projection = nn.Linear(768, base_lm.hidden_size)

    def forward(self, visual_input, text_input):
        """
        Process visual and textual input through extended language model.

        Args:
            visual_input: Images or point clouds
            text_input: Natural language text

        Returns:
            multimodal_output: Extended language model output with visual grounding
        """
        # Encode visual input
        visual_embeddings = self.vision_encoder(visual_input)

        # Project to language model space
        projected_visual = self.perception_projection(visual_embeddings)

        # Combine with text embeddings
        text_embeddings = self.base_language_model.embed_tokens(text_input)

        # Concatenate visual and text tokens
        combined_embeddings = torch.cat([text_embeddings, projected_visual], dim=1)

        # Process through language model
        output = self.base_language_model(inputs_embeds=combined_embeddings)

        return output
```

### VIMA (Vision-Language-Model-Agent)

VIMA treats robotics tasks as program generation:

```python
class VIMAModel:
    """
    VIMA: Vision-Language-Model-Agent architecture.
    """

    def __init__(self, vision_encoder, language_encoder, program_generator):
        self.vision_encoder = vision_encoder
        self.language_encoder = language_encoder
        self.program_generator = program_generator
        self.executor = RobotProgramExecutor()

    def forward(self, scene_image, instruction):
        """
        Generate robot program from vision and language input.

        Args:
            scene_image: Current scene image
            instruction: Natural language instruction

        Returns:
            execution_result: Result of program execution
        """
        # Encode scene and instruction
        scene_features = self.vision_encoder(scene_image)
        instruction_features = self.language_encoder(instruction)

        # Generate program
        robot_program = self.program_generator(
            scene_features=scene_features,
            instruction_features=instruction_features
        )

        # Execute program
        execution_result = self.executor.execute(robot_program)

        return execution_result
```

## NVIDIA Isaac Foundation Models

### Isaac Foundation Model Architecture

NVIDIA Isaac provides specialized foundation models for robotics:

```python
class IsaacFoundationModel:
    """
    NVIDIA Isaac Foundation Model for embodied AI.
    """

    def __init__(self, config):
        # Vision backbone (adapted from computer vision models)
        self.vision_backbone = VisionTransformer(
            patch_size=config.vision_patch_size,
            embed_dim=config.vision_embed_dim,
            depth=config.vision_depth,
            num_heads=config.vision_num_heads
        )

        # Language backbone (adapted from LLMs)
        self.language_backbone = TransformerLM(
            vocab_size=config.vocab_size,
            embed_dim=config.lang_embed_dim,
            depth=config.lang_depth,
            num_heads=config.lang_num_heads
        )

        # Action prediction head (robot-specific)
        self.action_head = RobotActionHead(
            input_dim=config.hidden_dim,
            robot_config=config.robot_spec
        )

        # Multimodal fusion transformer
        self.fusion_transformer = MultimodalFusionTransformer(
            hidden_dim=config.hidden_dim,
            num_layers=config.fusion_layers,
            num_heads=config.fusion_heads
        )

    def forward(self, images, text_commands, robot_state):
        """
        Forward pass through Isaac Foundation Model.

        Args:
            images: Robot's camera images
            text_commands: Natural language commands
            robot_state: Current robot state (joint positions, etc.)

        Returns:
            action_predictions: Predicted robot actions
        """
        # Process visual input
        visual_features = self.vision_backbone(images)

        # Process language input
        text_features = self.language_backbone(text_commands)

        # Process robot state
        state_features = self.encode_robot_state(robot_state)

        # Fuse all modalities
        fused_features = self.fusion_transformer(
            visual_features=visual_features,
            text_features=text_features,
            state_features=state_features
        )

        # Generate actions
        action_predictions = self.action_head(fused_features)

        return action_predictions

    def encode_robot_state(self, robot_state):
        """
        Encode robot state (joint positions, velocities, etc.) into features.
        """
        # Normalize and encode joint positions
        joint_positions = F.normalize(robot_state['joint_positions'], dim=-1)
        joint_velocities = F.normalize(robot_state['joint_velocities'], dim=-1)

        # Combine state features
        state_features = torch.cat([joint_positions, joint_velocities], dim=-1)
        state_features = self.state_projection(state_features)

        return state_features
```

## Real-Time VLA Implementation

### Streaming Architecture for Humanoid Control

For humanoid robots requiring real-time control, VLA systems need specialized architectures:

```python
class RealTimeVLA:
    """
    Real-time VLA system optimized for humanoid robot control.
    """

    def __init__(self, config):
        self.vision_pipeline = self.initialize_vision_pipeline(config)
        self.language_pipeline = self.initialize_language_pipeline(config)
        self.action_pipeline = self.initialize_action_pipeline(config)
        self.fusion_module = self.initialize_fusion_module(config)

        # Real-time optimization components
        self.inference_scheduler = InferenceScheduler(config.inference_frequency)
        self.buffer_manager = CircularBufferManager(config.buffer_size)
        self.performance_monitor = RealTimePerformanceMonitor()

    def process_frame_stream(self, frame_stream):
        """
        Process continuous stream of sensor data for real-time control.

        Args:
            frame_stream: Generator yielding (image, command, state) tuples

        Yields:
            action: Real-time robot actions
        """
        for frame_data in frame_stream:
            image, command, robot_state = frame_data

            # Add to processing buffer
            self.buffer_manager.add_frame({
                'image': image,
                'command': command,
                'state': robot_state,
                'timestamp': time.time()
            })

            # Process when scheduler indicates
            if self.inference_scheduler.should_infer():
                # Get latest data
                latest_data = self.buffer_manager.get_latest()

                # Run inference
                start_time = time.time()
                action = self.infer_action(
                    image=latest_data['image'],
                    command=latest_data['command'],
                    state=latest_data['state']
                )
                inference_time = time.time() - start_time

                # Monitor performance
                self.performance_monitor.record_inference_time(inference_time)

                # Yield action
                yield action

    def infer_action(self, image, command, state):
        """
        Infer action from multimodal inputs with latency optimization.
        """
        # Run vision pipeline (parallelizable)
        vision_future = self.vision_pipeline.submit(image)

        # Run language pipeline (parallelizable)
        language_future = self.language_pipeline.submit(command)

        # Run state processing (parallelizable)
        state_future = self.state_pipeline.submit(state)

        # Wait for all futures
        visual_features = vision_future.result()
        language_features = language_future.result()
        state_features = state_future.result()

        # Fuse and generate action
        fused_features = self.fusion_module(
            visual_features, language_features, state_features
        )
        action = self.action_pipeline(fused_features)

        return action

    def optimize_for_latency(self):
        """
        Optimize VLA system for minimal latency.
        """
        # Use TensorRT for inference optimization
        self.vision_pipeline = optimize_with_tensorrt(self.vision_pipeline)
        self.language_pipeline = optimize_with_tensorrt(self.language_pipeline)
        self.action_pipeline = optimize_with_tensorrt(self.action_pipeline)

        # Use mixed precision
        self.use_mixed_precision = True

        # Optimize memory allocation
        self.memory_pool = create_memory_pool()

        # Use multiple GPU streams for parallel processing
        self.gpu_streams = create_gpu_streams(3)  # vision, language, action
```

### GPU-Accelerated Inference

```python
class GPUAcceleratedVLA:
    """
    VLA system optimized for GPU acceleration.
    """

    def __init__(self, config):
        # Move models to GPU
        self.vision_model = self.load_vision_model(config).cuda()
        self.language_model = self.load_language_model(config).cuda()
        self.action_model = self.load_action_model(config).cuda()

        # Use TensorRT optimization
        self.use_tensorrt = config.use_tensorrt
        if self.use_tensorrt:
            self.vision_model = self.optimize_with_tensorrt(self.vision_model)
            self.language_model = self.optimize_with_tensorrt(self.language_model)
            self.action_model = self.optimize_with_tensorrt(self.action_model)

        # Create CUDA streams for parallel processing
        self.vision_stream = torch.cuda.Stream()
        self.language_stream = torch.cuda.Stream()
        self.action_stream = torch.cuda.Stream()

        # Create pinned memory for faster host-device transfers
        self.pinned_memory_pool = self.create_pinned_memory_pool()

    def process_multimodal_batch(self, batch_data):
        """
        Process batch of multimodal data with GPU optimization.
        """
        # Move data to GPU asynchronously using streams
        with torch.cuda.stream(self.vision_stream):
            vision_tensors = self.prepare_vision_tensors(batch_data['images'])

        with torch.cuda.stream(self.language_stream):
            language_tensors = self.prepare_language_tensors(batch_data['commands'])

        with torch.cuda.stream(self.action_stream):
            state_tensors = self.prepare_state_tensors(batch_data['states'])

        # Synchronize streams
        torch.cuda.synchronize()

        # Run inference
        with torch.no_grad():
            if self.use_tensorrt:
                # Use TensorRT optimized inference
                vision_features = self.run_tensorrt_inference(
                    self.vision_model,
                    vision_tensors
                )
                language_features = self.run_tensorrt_inference(
                    self.language_model,
                    language_tensors
                )
            else:
                # Use PyTorch inference
                vision_features = self.vision_model(vision_tensors)
                language_features = self.language_model(language_tensors)

        # Fuse modalities
        fused_features = self.fuse_modalities(
            vision_features,
            language_features,
            state_tensors
        )

        # Generate actions
        actions = self.action_model(fused_features)

        return actions

    def prepare_vision_tensors(self, images):
        """
        Prepare vision tensors with GPU optimization.
        """
        # Convert to tensor and move to GPU
        tensors = torch.stack([torch.from_numpy(img) for img in images]).cuda(non_blocking=True)

        # Normalize on GPU
        tensors = tensors.float() / 255.0
        tensors = (tensors - self.vision_mean) / self.vision_std

        return tensors

    def prepare_language_tensors(self, commands):
        """
        Prepare language tensors with GPU optimization.
        """
        # Tokenize on CPU, move to GPU
        tokenized = self.tokenizer(
            commands,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        input_ids = tokenized['input_ids'].cuda(non_blocking=True)
        attention_mask = tokenized['attention_mask'].cuda(non_blocking=True)

        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    def run_tensorrt_inference(self, engine, input_tensors):
        """
        Run inference using TensorRT engine.
        """
        # Allocate GPU memory for inputs and outputs
        input_buffers = [
            input_tensors.contiguous().data_ptr(),
        ]

        output_shape = self.get_output_shape()
        output_buffer = torch.empty(output_shape, dtype=torch.float32, device='cuda')

        # Run inference
        engine.execute_async_v2(
            bindings=input_buffers + [output_buffer.data_ptr()],
            stream_handle=torch.cuda.current_stream().cuda_stream
        )

        return output_buffer
```

## Humanoid-Specific VLA Considerations

### Bipedal Locomotion Integration

Humanoid robots require special consideration for balance and locomotion:

```python
class HumanoidVLA:
    """
    VLA system specifically designed for humanoid robots with bipedal locomotion.
    """

    def __init__(self, config):
        # Standard VLA components
        self.vision_model = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.language_config)
        self.action_model = ActionModel(config.action_config)

        # Humanoid-specific components
        self.balance_controller = BalanceController(config.balance_config)
        self.footstep_planner = FootstepPlanner(config.footstep_config)
        self.inverse_kinematics = InverseKinematicsSolver(config.robot_urdf)

    def generate_humanoid_action(self, visual_input, language_command, robot_state):
        """
        Generate humanoid-specific actions with balance considerations.

        Args:
            visual_input: Visual observations from robot cameras
            language_command: Natural language command
            robot_state: Current robot state (joint positions, IMU, etc.)

        Returns:
            humanoid_action: Humanoid-specific action with balance constraints
        """
        # Standard VLA processing
        high_level_plan = self.plan_from_vision_language(
            visual_input, language_command
        )

        # Humanoid-specific processing
        locomotion_plan = self.plan_locomotion(high_level_plan, robot_state)
        manipulation_plan = self.plan_manipulation(high_level_plan, robot_state)

        # Integrate with balance control
        balanced_action = self.integrate_balance_control(
            locomotion_plan, manipulation_plan, robot_state
        )

        return balanced_action

    def plan_locomotion(self, high_level_plan, robot_state):
        """
        Plan bipedal locomotion based on high-level plan.
        """
        if high_level_plan.requires_movement():
            # Plan footstep sequence
            footstep_sequence = self.footstep_planner.plan_footsteps(
                goal_position=high_level_plan.get_goal_position(),
                current_state=robot_state
            )

            # Generate locomotion commands
            locomotion_commands = self.generate_locomotion_commands(
                footstep_sequence, robot_state
            )

            return locomotion_commands

        return None

    def plan_manipulation(self, high_level_plan, robot_state):
        """
        Plan manipulation actions based on high-level plan.
        """
        if high_level_plan.requires_manipulation():
            # Plan manipulation sequence
            manipulation_sequence = self.plan_manipulation_sequence(
                target_object=high_level_plan.get_target_object(),
                current_state=robot_state
            )

            # Integrate with balance control
            balanced_manipulation = self.balance_controller.integrate_balance(
                manipulation_sequence, robot_state
            )

            return balanced_manipulation

        return None

    def integrate_balance_control(self, locomotion_plan, manipulation_plan, robot_state):
        """
        Integrate balance control with locomotion and manipulation plans.
        """
        # Create integrated plan that maintains balance
        integrated_plan = {
            'locomotion': locomotion_plan,
            'manipulation': manipulation_plan,
            'balance_control': True,
            'center_of_mass_target': self.calculate_balance_target(
                locomotion_plan, manipulation_plan, robot_state
            )
        }

        return integrated_plan

    def calculate_balance_target(self, locomotion_plan, manipulation_plan, robot_state):
        """
        Calculate balance control target based on planned actions.
        """
        # Calculate desired center of mass position
        if locomotion_plan and manipulation_plan:
            # Both locomotion and manipulation - complex balance planning
            com_target = self.calculate_dynamic_balance_target(
                locomotion_plan, manipulation_plan, robot_state
            )
        elif locomotion_plan:
            # Locomotion only - footstep-based balance
            com_target = self.calculate_locomotion_balance_target(
                locomotion_plan, robot_state
            )
        elif manipulation_plan:
            # Manipulation only - static balance with manipulation compensation
            com_target = self.calculate_manipulation_balance_target(
                manipulation_plan, robot_state
            )
        else:
            # Idle - maintain neutral balance
            com_target = robot_state['neutral_com_position']

        return com_target
```

## Performance Optimization Strategies

### Latency Optimization for Real-Time Control

```python
class OptimizedVLA:
    """
    VLA system optimized for real-time humanoid control.
    """

    def __init__(self, config):
        self.config = config
        self.models = self.load_optimized_models()
        self.buffers = self.initialize_buffers()
        self.schedulers = self.setup_schedulers()

    def load_optimized_models(self):
        """
        Load models with performance optimizations.
        """
        models = {}

        # Vision model - optimized for 60Hz+ inference
        vision_model = VisionModel(self.config.vision)
        if self.config.use_tensorrt:
            models['vision'] = self.optimize_for_tensorrt(vision_model)
        else:
            models['vision'] = self.optimize_for_torchscript(vision_model)

        # Language model - optimized for quick command parsing
        language_model = LanguageModel(self.config.language)
        models['language'] = self.optimize_language_model(language_model)

        # Action model - optimized for fast action generation
        action_model = ActionModel(self.config.action)
        models['action'] = self.optimize_action_model(action_model)

        return models

    def initialize_buffers(self):
        """
        Initialize circular buffers for streaming processing.
        """
        return {
            'vision': CircularBuffer(size=self.config.vision_buffer_size),
            'language': CircularBuffer(size=self.config.language_buffer_size),
            'action': CircularBuffer(size=self.config.action_buffer_size),
        }

    def setup_schedulers(self):
        """
        Setup inference schedulers for different frequencies.
        """
        return {
            'vision': PeriodicScheduler(frequency=self.config.vision_frequency),      # e.g., 30Hz
            'language': EventScheduler(),                                           # On command arrival
            'action': RateLimiter(max_rate=self.config.action_frequency),          # e.g., 200Hz
        }

    def process_streaming_input(self, input_stream):
        """
        Process streaming input with optimized scheduling.
        """
        for input_data in input_stream:
            # Process language commands immediately
            if input_data.type == 'command':
                language_features = self.process_language_now(input_data.command)
                self.buffers['language'].add(language_features)

            # Process vision on schedule
            if self.schedulers['vision'].should_run():
                vision_features = self.process_vision_now(input_data.image)
                self.buffers['vision'].add(vision_features)

            # Generate actions when possible
            if self.can_generate_action():
                action = self.generate_action_now()
                if self.schedulers['action'].can_publish():
                    yield action

    def can_generate_action(self):
        """
        Check if we have enough data to generate an action.
        """
        return (
            not self.buffers['vision'].is_empty() and
            not self.buffers['language'].is_empty()
        )

    def generate_action_now(self):
        """
        Generate action immediately from buffered data.
        """
        # Get latest features from each modality
        vision_features = self.buffers['vision'].get_latest()
        language_features = self.buffers['language'].get_latest()

        # Fuse and generate action
        fused_features = self.fuse_modalities(vision_features, language_features)
        action = self.models['action'](fused_features)

        return action

    def optimize_for_tensorrt(self, model):
        """
        Optimize model using TensorRT.
        """
        import tensorrt as trt
        from torch_tensorrt import compile

        # Convert to TensorRT
        optimized_model = compile(
            model,
            inputs=[torch_tensorrt.Input(shape=[1, 3, 224, 224])],  # Example input
            enabled_precisions={torch.float, torch.half},
            workspace_size=2000000000,  # 2GB
        )

        return optimized_model

    def optimize_for_torchscript(self, model):
        """
        Optimize model using TorchScript.
        """
        model.eval()
        traced_model = torch.jit.trace(model, torch.rand(1, 3, 224, 224))
        traced_model = traced_model.optimize_for_inference()
        return traced_model
```

## Evaluation Metrics for VLA Systems

### Quantitative Metrics

```python
class VLAEvaluator:
    """
    Evaluation metrics for VLA system performance.
    """

    def __init__(self):
        self.metrics = {
            'task_success_rate': 0.0,
            'response_latency': [],
            'action_accuracy': 0.0,
            'language_understanding': 0.0,
            'vision_grounding': 0.0,
        }

    def evaluate_episode(self, episode_data):
        """
        Evaluate complete robot episode.

        Args:
            episode_data: Dictionary containing:
                         - initial_state: Starting state
                         - command: Natural language command
                         - actions: Executed actions
                         - final_state: Final state
                         - success: Whether task was completed
                         - execution_time: Total execution time

        Returns:
            episode_metrics: Dictionary of episode metrics
        """
        episode_metrics = {}

        # Task success
        episode_metrics['success'] = episode_data['success']
        episode_metrics['completion_time'] = episode_data['execution_time']

        # Action accuracy (compared to expert demonstrations)
        if 'expert_actions' in episode_data:
            episode_metrics['action_accuracy'] = self.calculate_action_accuracy(
                episode_data['actions'],
                episode_data['expert_actions']
            )

        # Language understanding (command following accuracy)
        episode_metrics['language_following'] = self.evaluate_command_following(
            episode_data['command'],
            episode_data['actions'],
            episode_data['final_state']
        )

        # Vision grounding (did robot attend to correct objects?)
        episode_metrics['vision_grounding'] = self.evaluate_vision_grounding(
            episode_data['command'],
            episode_data['actions'],
            episode_data['attended_objects']
        )

        return episode_metrics

    def calculate_action_accuracy(self, predicted_actions, expert_actions):
        """
        Calculate accuracy of predicted actions vs expert actions.
        """
        if len(predicted_actions) != len(expert_actions):
            # Use DTW for sequence alignment
            return self.calculate_dtw_accuracy(predicted_actions, expert_actions)

        correct_actions = 0
        total_actions = len(expert_actions)

        for pred, expert in zip(predicted_actions, expert_actions):
            if self.actions_match(pred, expert):
                correct_actions += 1

        return correct_actions / total_actions if total_actions > 0 else 0.0

    def evaluate_command_following(self, command, actions, final_state):
        """
        Evaluate how well the robot followed the command.
        """
        # Parse expected behavior from command
        expected_behavior = self.parse_command_expectations(command)

        # Evaluate actual behavior
        actual_behavior = self.extract_behavior_from_actions(actions, final_state)

        # Compare behaviors
        similarity = self.compare_behaviors(expected_behavior, actual_behavior)

        return similarity

    def evaluate_vision_grounding(self, command, actions, attended_objects):
        """
        Evaluate whether robot attended to correct objects in the scene.
        """
        # Extract object references from command
        referenced_objects = self.extract_referenced_objects(command)

        # Check if robot attended to correct objects
        correct_attentions = 0
        total_references = len(referenced_objects)

        for ref_obj in referenced_objects:
            if self.object_was_attended(ref_obj, attended_objects):
                correct_attentions += 1

        return correct_attentions / total_references if total_references > 0 else 0.0

    def aggregate_evaluation(self, all_episodes):
        """
        Aggregate evaluation across multiple episodes.
        """
        aggregated = {}

        for metric in self.metrics:
            values = [episode.get(metric, 0) for episode in all_episodes]
            aggregated[f'{metric}_mean'] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values)
            aggregated[f'{metric}_median'] = np.median(values)

        return aggregated
```

## Learning Outcomes

After completing this section, students will understand:
1. Different VLA architectural approaches and their trade-offs
2. How to implement GPU-accelerated VLA systems
3. Humanoid-specific considerations for VLA integration
4. Performance optimization techniques for real-time operation
5. Evaluation methodologies for VLA systems
6. How to select appropriate architectures for specific applications

## Next Steps

Continue to [VLA Implementation Patterns](./vla-implementation-patterns.md) to learn about practical implementation strategies and code patterns for VLA systems.