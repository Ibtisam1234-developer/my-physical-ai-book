# LLM-Powered Task Planning

## Introduction to LLM-Based Task Planning

Large Language Models (LLMs) have revolutionized the field of task planning for robotic systems. By leveraging the reasoning capabilities of LLMs, robots can now understand high-level natural language commands and decompose them into executable action sequences. This approach enables more intuitive human-robot interaction and flexible task execution.

### The Evolution of Task Planning

Traditional task planning in robotics relied on:
- **Symbolic planners**: Required hand-coded domain knowledge
- **Finite state machines**: Limited flexibility and scalability
- **Hierarchical task networks**: Complex to maintain and update
- **Rule-based systems**: Brittle and difficult to modify

LLM-based task planning offers:
- **Natural language understanding**: Accept commands in everyday language
- **Commonsense reasoning**: Leverage world knowledge for planning
- **Flexibility**: Handle novel situations without reprogramming
- **Learning**: Improve through interaction and feedback

### Architecture Overview

```python
class LLMTaskPlanner:
    """
    LLM-powered task planning system for humanoid robots.
    """

    def __init__(self, llm_client, robot_capabilities, environment_knowledge):
        self.llm_client = llm_client  # OpenAI, Claude, or local LLM
        self.robot_capabilities = robot_capabilities
        self.environment_knowledge = environment_knowledge
        self.task_memory = TaskMemory()

    def plan_task(self, natural_language_command, current_state):
        """
        Plan a task from natural language command.

        Args:
            natural_language_command: Human-readable task description
            current_state: Current robot and environment state

        Returns:
            TaskPlan: Decomposed task plan with actions
        """
        # 1. Parse the command
        parsed_task = self.parse_command(natural_language_command)

        # 2. Analyze current state
        state_analysis = self.analyze_state(current_state)

        # 3. Generate plan using LLM
        raw_plan = self.generate_plan_with_llm(parsed_task, state_analysis)

        # 4. Validate and refine plan
        validated_plan = self.validate_plan(raw_plan)

        # 5. Convert to executable actions
        executable_plan = self.convert_to_actions(validated_plan)

        return executable_plan

    def parse_command(self, command):
        """Parse natural language command into structured format."""
        # Use LLM to extract intent, objects, locations, constraints
        pass

    def analyze_state(self, state):
        """Analyze current environment and robot state."""
        # Extract relevant information for planning
        pass

    def generate_plan_with_llm(self, task, state):
        """Generate task plan using LLM reasoning."""
        # Construct prompt with task, state, and capabilities
        # Call LLM API
        # Parse response into structured plan
        pass

    def validate_plan(self, plan):
        """Validate plan against robot capabilities and environment."""
        # Check feasibility
        # Verify constraints
        # Handle conflicts
        pass

    def convert_to_actions(self, plan):
        """Convert high-level plan to executable robot actions."""
        # Map to robot-specific action space
        # Add error handling
        # Include safety checks
        pass
```

## LLM Integration for Robotics

### System Architecture

```python
class RoboticLLMInterface:
    """
    Interface between LLMs and robotic systems.
    """

    def __init__(self, llm_provider="openai", model="gpt-4-turbo"):
        self.llm_provider = llm_provider
        self.model = model
        self.client = self.initialize_client()

        # Robot-specific tools and functions
        self.available_tools = [
            "navigation", "manipulation", "perception",
            "communication", "navigation", "manipulation"
        ]

        # Safety and validation layer
        self.safety_validator = SafetyValidator()

    def initialize_client(self):
        """Initialize LLM client based on provider."""
        if self.llm_provider == "openai":
            return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.llm_provider == "anthropic":
            return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        elif self.llm_provider == "local":
            return LocalLLMClient(model_path=os.getenv("LOCAL_LLM_PATH"))

    def generate_task_plan(self, user_command, robot_state, environment_context):
        """
        Generate task plan using LLM with robotic context.

        Args:
            user_command: Natural language command from user
            robot_state: Current robot state and capabilities
            environment_context: Environmental information

        Returns:
            dict: Task plan with actions and parameters
        """
        # Construct system message with robot context
        system_message = self.construct_system_prompt(robot_state)

        # Construct user message with command and context
        user_message = self.construct_user_prompt(
            user_command, environment_context
        )

        # Prepare tools/functions for LLM
        tools = self.prepare_robotic_tools()

        # Call LLM with structured input
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            tools=tools,
            tool_choice="auto",
            temperature=0.3,  # Lower temperature for more consistent planning
            max_tokens=2048,
        )

        # Process and validate response
        task_plan = self.process_llm_response(response)

        # Validate for safety and feasibility
        validated_plan = self.safety_validator.validate(task_plan)

        return validated_plan

    def construct_system_prompt(self, robot_state):
        """
        Construct system prompt with robot capabilities and constraints.
        """
        capabilities = self.describe_robot_capabilities(robot_state)

        system_prompt = f"""
        You are an AI task planner for a humanoid robot. Your role is to decompose high-level human commands into executable robotic actions.

        ROBOT CAPABILITIES:
        {capabilities}

        ENVIRONMENT INTERACTION:
        - Always consider physical constraints and safety
        - Use object detection before manipulation
        - Plan for bipedal locomotion when navigating
        - Include perception steps before action execution

        TASK PLANNING RULES:
        1. Break down complex tasks into atomic actions
        2. Include necessary perception steps
        3. Consider preconditions and postconditions
        4. Plan for error recovery
        5. Ensure safety at each step

        ACTION FORMAT:
        - Navigation: move_to(location), avoid_obstacles()
        - Manipulation: pick_object(object_id), place_object(object_id, location)
        - Perception: detect_objects(), recognize_speakers()
        - Communication: speak(text), gesture(type)

        Respond with structured JSON containing the task plan.
        """

        return system_prompt

    def construct_user_prompt(self, command, context):
        """
        Construct user prompt with command and environmental context.
        """
        user_prompt = f"""
        USER COMMAND: {command}

        CURRENT CONTEXT:
        - Environment: {context.get('environment_description', 'unknown')}
        - Detected Objects: {context.get('detected_objects', [])}
        - Robot Location: {context.get('robot_location', 'unknown')}
        - Available Tools: {context.get('available_tools', [])}
        - Safety Constraints: {context.get('safety_constraints', [])}

        Please generate a detailed task plan that accomplishes the user's request.
        Include all necessary steps, safety considerations, and error handling.
        """

        return user_prompt

    def prepare_robotic_tools(self):
        """
        Prepare robotic tools/functions for LLM tool calling.
        """
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "move_to_location",
                    "description": "Navigate robot to specified location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "Target location"},
                            "speed": {"type": "number", "description": "Movement speed (0.1-1.0)"}
                        },
                        "required": ["location"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "pick_object",
                    "description": "Pick up an object with robot hand",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "object_id": {"type": "string", "description": "ID of object to pick"},
                            "hand": {"type": "string", "description": "Hand to use (left/right)"}
                        },
                        "required": ["object_id", "hand"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "place_object",
                    "description": "Place object at specified location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "object_id": {"type": "string", "description": "ID of object to place"},
                            "location": {"type": "string", "description": "Target placement location"},
                            "hand": {"type": "string", "description": "Hand holding object"}
                        },
                        "required": ["object_id", "location", "hand"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "detect_objects",
                    "description": "Detect and identify objects in current view",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "area_of_interest": {"type": "string", "description": "Area to scan"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "speak_text",
                    "description": "Speak text through robot's speech system",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "Text to speak"},
                            "tone": {"type": "string", "description": "Speech tone"}
                        },
                        "required": ["text"]
                    }
                }
            }
        ]

        return tools

    def process_llm_response(self, response):
        """
        Process LLM response into structured task plan.
        """
        # Extract tool calls or text response
        if response.choices[0].message.tool_calls:
            # Process function calls
            plan_steps = []
            for tool_call in response.choices[0].message.tool_calls:
                plan_steps.append({
                    "action": tool_call.function.name,
                    "parameters": json.loads(tool_call.function.arguments),
                    "id": tool_call.id
                })
        else:
            # Process text response
            text_content = response.choices[0].message.content
            plan_steps = self.parse_text_plan(text_content)

        return {
            "plan_steps": plan_steps,
            "confidence": response.choices[0].finish_reason == "stop",
            "usage": response.usage
        }

    def parse_text_plan(self, text_plan):
        """
        Parse text-based plan into structured format.
        """
        # Use regex or structured parsing to extract plan steps
        # This is a simplified example - real implementation would be more robust
        steps = []
        lines = text_plan.split('\n')

        for line in lines:
            if line.strip().startswith('- '):
                step_text = line.strip()[2:]  # Remove '- ' prefix
                # Parse action and parameters
                steps.append({
                    "action": "generic_action",
                    "description": step_text,
                    "parameters": {}
                })

        return steps
```

## Hierarchical Task Decomposition

### Multi-Level Planning Architecture

```python
class HierarchicalTaskPlanner:
    """
    Hierarchical task planner with multiple levels of abstraction.
    """

    def __init__(self):
        self.high_level_planner = HighLevelPlanner()
        self.mid_level_planner = MidLevelPlanner()
        self.low_level_planner = LowLevelPlanner()

    def plan_task_hierarchically(self, high_level_goal):
        """
        Plan task using hierarchical decomposition.

        Args:
            high_level_goal: High-level task description

        Returns:
            HierarchicalPlan: Multi-level plan structure
        """
        # Level 1: High-level goal decomposition
        mid_level_tasks = self.high_level_planner.decompose(high_level_goal)

        # Level 2: Mid-level task refinement
        detailed_tasks = []
        for task in mid_level_tasks:
            refined_tasks = self.mid_level_planner.refine(task)
            detailed_tasks.extend(refined_tasks)

        # Level 3: Low-level action generation
        action_sequences = []
        for task in detailed_tasks:
            actions = self.low_level_planner.generate_actions(task)
            action_sequences.append(actions)

        return HierarchicalPlan(
            high_level=mid_level_tasks,
            mid_level=detailed_tasks,
            low_level=action_sequences
        )


class HighLevelPlanner:
    """
    High-level planner for goal decomposition.
    """

    def __init__(self):
        self.llm_interface = RoboticLLMInterface()

    def decompose(self, goal):
        """
        Decompose high-level goal into mid-level subtasks.

        Example:
        Goal: "Prepare coffee in kitchen"
        Subtasks: ["Navigate to kitchen", "Find coffee maker", "Prepare coffee", "Serve coffee"]
        """
        system_prompt = """
        You are a high-level task decomposition expert for robotics.
        Decompose the given goal into 3-7 high-level subtasks that can be executed by a humanoid robot.
        Each subtask should be achievable and logically sequenced.
        """

        user_prompt = f"Goal: {goal}\n\nDecompose this goal into high-level subtasks:"

        response = self.llm_interface.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )

        subtasks = self.parse_subtasks(response.choices[0].message.content)
        return subtasks

    def parse_subtasks(self, text_response):
        """
        Parse LLM response into structured subtasks.
        """
        # Simple parsing - in practice would use more robust parsing
        lines = text_response.strip().split('\n')
        subtasks = []

        for line in lines:
            line = line.strip()
            if line and (line.startswith('1.') or line.startswith('2.') or
                        line.startswith('3.') or line.startswith('-')):
                # Extract task description
                task_desc = line.split('.', 1)[-1].split('-', 1)[-1].strip()
                if task_desc:
                    subtasks.append({
                        "id": len(subtasks) + 1,
                        "description": task_desc,
                        "dependencies": [],
                        "estimated_duration": self.estimate_duration(task_desc)
                    })

        return subtasks

    def estimate_duration(self, task_description):
        """
        Estimate task duration based on description.
        """
        # Simple heuristics - in practice would use learned models
        if any(word in task_description.lower() for word in ["navigate", "go to", "move to"]):
            return 30  # seconds
        elif any(word in task_description.lower() for word in ["pick", "grasp", "take", "place"]):
            return 10
        elif any(word in task_description.lower() for word in ["find", "locate", "search"]):
            return 20
        else:
            return 15


class MidLevelPlanner:
    """
    Mid-level planner for task refinement and constraint checking.
    """

    def __init__(self):
        self.constraint_checker = ConstraintChecker()
        self.feasibility_analyzer = FeasibilityAnalyzer()

    def refine(self, high_level_task):
        """
        Refine high-level task with constraints and feasibility checks.

        Args:
            high_level_task: High-level task description

        Returns:
            List[DetailedTask]: Refined tasks with constraints
        """
        # Identify required resources
        required_resources = self.identify_resources(high_level_task["description"])

        # Check feasibility
        is_feasible = self.feasibility_analyzer.check(
            task=high_level_task["description"],
            resources=required_resources
        )

        if not is_feasible:
            raise TaskPlanningError(f"Task not feasible: {high_level_task['description']}")

        # Generate detailed steps
        detailed_steps = self.generate_detailed_steps(high_level_task)

        # Add constraints and dependencies
        for step in detailed_steps:
            step["constraints"] = self.constraint_checker.get_constraints(step)
            step["safety_requirements"] = self.get_safety_requirements(step)

        return detailed_steps

    def identify_resources(self, task_description):
        """
        Identify resources required for task execution.
        """
        resources = {
            "manipulation": False,
            "navigation": False,
            "perception": False,
            "communication": False,
            "specific_tools": []
        }

        desc_lower = task_description.lower()

        if any(word in desc_lower for word in ["pick", "grasp", "take", "place", "move"]):
            resources["manipulation"] = True

        if any(word in desc_lower for word in ["go", "navigate", "move to", "walk", "travel"]):
            resources["navigation"] = True

        if any(word in desc_lower for word in ["find", "locate", "identify", "recognize"]):
            resources["perception"] = True

        if any(word in desc_lower for word in ["speak", "say", "tell", "communicate"]):
            resources["communication"] = True

        return resources

    def generate_detailed_steps(self, task):
        """
        Generate detailed steps for high-level task.
        """
        # This would use more sophisticated planning in practice
        # For now, we'll create a simple mapping
        desc = task["description"].lower()

        if "navigate" in desc or "go to" in desc:
            return self.generate_navigation_steps(task)
        elif "manipulation" in desc or "pick" in desc or "grasp" in desc:
            return self.generate_manipulation_steps(task)
        elif "perception" in desc or "find" in desc or "locate" in desc:
            return self.generate_perception_steps(task)
        else:
            return [task]  # Return as-is for other tasks

    def generate_navigation_steps(self, task):
        """
        Generate detailed steps for navigation tasks.
        """
        return [
            {
                "id": f"{task['id']}_1",
                "type": "navigation",
                "action": "path_planning",
                "description": f"Plan path to destination in {task['description']}",
                "parameters": {"destination": self.extract_destination(task["description"])},
                "preconditions": ["robot_is_mobile", "destination_is_known"],
                "postconditions": ["path_is_computed"]
            },
            {
                "id": f"{task['id']}_2",
                "type": "navigation",
                "action": "locomotion",
                "description": "Execute bipedal locomotion to destination",
                "parameters": {"gait_type": "walking", "speed": 0.5},
                "preconditions": ["path_is_computed"],
                "postconditions": ["robot_at_destination"]
            }
        ]


class LowLevelPlanner:
    """
    Low-level planner for action sequence generation.
    """

    def __init__(self):
        self.action_library = ActionLibrary()
        self.motion_planner = MotionPlanner()

    def generate_actions(self, detailed_task):
        """
        Generate low-level robot actions for detailed task.

        Args:
            detailed_task: Detailed task with constraints

        Returns:
            List[RobotAction]: Sequence of executable robot actions
        """
        action_sequence = []

        if detailed_task["type"] == "navigation":
            actions = self.generate_navigation_actions(detailed_task)
        elif detailed_task["type"] == "manipulation":
            actions = self.generate_manipulation_actions(detailed_task)
        elif detailed_task["type"] == "perception":
            actions = self.generate_perception_actions(detailed_task)
        else:
            actions = self.generate_generic_actions(detailed_task)

        # Add safety checks and error handling
        action_sequence = self.add_safety_wrappers(actions, detailed_task)

        return action_sequence

    def generate_navigation_actions(self, task):
        """
        Generate navigation actions for humanoid robot.
        """
        actions = []

        # 1. Localize robot
        actions.append({
            "name": "localize_robot",
            "type": "perception",
            "parameters": {},
            "duration": 0.5
        })

        # 2. Plan footstep sequence
        actions.append({
            "name": "plan_footsteps",
            "type": "planning",
            "parameters": {
                "destination": task["parameters"]["destination"],
                "terrain_type": "flat"
            },
            "duration": 1.0
        })

        # 3. Execute bipedal locomotion
        actions.append({
            "name": "execute_locomotion",
            "type": "motion",
            "parameters": {
                "gait_pattern": "walking",
                "step_frequency": 1.0,
                "balance_control": True
            },
            "duration": task["estimated_duration"]
        })

        # 4. Verify arrival
        actions.append({
            "name": "verify_arrival",
            "type": "verification",
            "parameters": {
                "tolerance": 0.1  # 10cm tolerance
            },
            "duration": 0.5
        })

        return actions

    def add_safety_wrappers(self, actions, task):
        """
        Add safety checks and error handling to action sequence.
        """
        wrapped_actions = []

        for action in actions:
            # Add precondition check
            wrapped_actions.append({
                "name": f"check_precondition_{action['name']}",
                "type": "verification",
                "parameters": {
                    "condition": action.get("precondition", "always_true")
                },
                "wrapped_action": None
            })

            # Add main action
            wrapped_actions.append(action)

            # Add postcondition verification
            wrapped_actions.append({
                "name": f"verify_postcondition_{action['name']}",
                "type": "verification",
                "parameters": {
                    "condition": action.get("postcondition", "always_true"),
                    "timeout": 5.0
                },
                "wrapped_action": action["name"]
            })

        return wrapped_actions
```

## Context-Aware Planning

### Situation Assessment and Adaptation

```python
class ContextAwarePlanner:
    """
    Context-aware task planner that adapts to environment and situation.
    """

    def __init__(self):
        self.context_analyzer = ContextAnalyzer()
        self.adaptation_engine = AdaptationEngine()
        self.knowledge_base = KnowledgeBase()

    def plan_with_context(self, goal, current_context):
        """
        Plan task considering current context and situational factors.

        Args:
            goal: High-level goal to achieve
            current_context: Current situation context

        Returns:
            AdaptivePlan: Context-aware plan with adaptations
        """
        # Analyze current context
        context_analysis = self.context_analyzer.analyze(current_context)

        # Identify relevant contextual factors
        factors = self.extract_contextual_factors(context_analysis)

        # Adapt plan based on context
        adapted_plan = self.adaptation_engine.adapt(
            original_goal=goal,
            contextual_factors=factors,
            environment_knowledge=self.knowledge_base.get_environment_info()
        )

        return adapted_plan

    def extract_contextual_factors(self, context_analysis):
        """
        Extract relevant contextual factors from analysis.
        """
        factors = {
            "environmental": {
                "lighting": context_analysis.get("lighting_conditions", "normal"),
                "noise_level": context_analysis.get("noise_level", "low"),
                "crowd_density": context_analysis.get("crowd_density", "low"),
                "obstacle_density": context_analysis.get("obstacle_density", "low")
            },
            "temporal": {
                "time_of_day": context_analysis.get("time_of_day", "day"),
                "urgency": context_analysis.get("task_urgency", "normal"),
                "operating_hours": context_analysis.get("operating_hours", True)
            },
            "social": {
                "people_present": context_analysis.get("people_count", 0),
                "interaction_mode": context_analysis.get("preferred_interaction", "direct"),
                "privacy_concerns": context_analysis.get("privacy_sensitivity", False)
            },
            "robot_state": {
                "battery_level": context_analysis.get("battery_level", 1.0),
                "capability_availability": context_analysis.get("available_capabilities", []),
                "recent_performance": context_analysis.get("recent_success_rate", 0.95)
            }
        }

        return factors

    def adapt_plan_for_context(self, original_plan, contextual_factors):
        """
        Adapt original plan based on contextual factors.
        """
        adapted_plan = copy.deepcopy(original_plan)

        # Adapt for lighting conditions
        if contextual_factors["environmental"]["lighting"] == "poor":
            adapted_plan = self.adapt_for_low_light(adapted_plan)

        # Adapt for noise level
        if contextual_factors["environmental"]["noise_level"] == "high":
            adapted_plan = self.adapt_for_high_noise(adapted_plan)

        # Adapt for crowd density
        if contextual_factors["environmental"]["crowd_density"] == "high":
            adapted_plan = self.adapt_for_crowds(adapted_plan)

        # Adapt for urgency
        if contextual_factors["temporal"]["urgency"] == "high":
            adapted_plan = self.adapt_for_urgency(adapted_plan)

        # Adapt for battery level
        if contextual_factors["robot_state"]["battery_level"] < 0.3:
            adapted_plan = self.adapt_for_low_battery(adapted_plan)

        return adapted_plan

    def adapt_for_low_light(self, plan):
        """
        Adapt plan for low-light conditions.
        """
        # Add extra perception steps
        for step in plan.steps:
            if step.type == "navigation":
                step.parameters["caution_level"] = "high"
                step.parameters["speed"] = step.parameters.get("speed", 1.0) * 0.5
            elif step.type == "manipulation":
                step.preconditions.append("adequate_lighting_verified")

        # Add lighting actions if needed
        lighting_action = {
            "name": "activate_lighting",
            "type": "perception",
            "parameters": {"source": "head_mounted_light"},
            "priority": "high"
        }

        plan.steps.insert(0, lighting_action)

        return plan

    def adapt_for_high_noise(self, plan):
        """
        Adapt plan for high-noise environments.
        """
        # Modify communication actions
        for step in plan.steps:
            if step.type == "communication":
                step.parameters["volume"] = "loud"
                step.parameters["repetition"] = 2  # Repeat important information

        return plan

    def adapt_for_crowds(self, plan):
        """
        Adapt plan for crowded environments.
        """
        # Modify navigation behavior
        for step in plan.steps:
            if step.type == "navigation":
                step.parameters["social_navigation"] = True
                step.parameters["personal_space_respect"] = True

        # Add social awareness steps
        crowd_aware_actions = [
            {
                "name": "scan_for_people",
                "type": "perception",
                "parameters": {"scan_radius": 2.0},
                "frequency": 1.0  # Hz
            },
            {
                "name": "yield_to_pedestrians",
                "type": "navigation",
                "parameters": {"priority": "pedestrian"},
                "active": True
            }
        ]

        # Insert crowd-aware actions at appropriate points
        plan.steps = crowd_aware_actions + plan.steps

        return plan

    def adapt_for_urgency(self, plan):
        """
        Adapt plan for high-urgency tasks.
        """
        # Optimize for speed
        for step in plan.steps:
            if step.type == "navigation":
                step.parameters["speed"] = min(step.parameters.get("speed", 1.0) * 1.5, 1.0)
            elif step.type == "waiting":
                step.parameters["max_wait_time"] = 5.0  # Reduce wait times

        # Remove non-essential verification steps
        essential_steps = [step for step in plan.steps if step.essential]
        plan.steps = essential_steps

        return plan

    def adapt_for_low_battery(self, plan):
        """
        Adapt plan for low battery conditions.
        """
        # Optimize for energy efficiency
        for step in plan.steps:
            if step.type == "navigation":
                step.parameters["energy_efficient"] = True
                step.parameters["speed"] = max(step.parameters.get("speed", 1.0) * 0.7, 0.3)
            elif step.type == "computation":
                step.parameters["power_mode"] = "low_power"

        # Add battery charging step if needed
        if plan.expected_duration > self.estimate_battery_life(plan):
            charging_step = {
                "name": "charge_battery",
                "type": "maintenance",
                "parameters": {"minimum_charge": 0.8},
                "essential": True
            }
            plan.steps.insert(-1, charging_step)  # Before final step

        return plan

    def estimate_battery_life(self, plan):
        """
        Estimate battery life for given plan.
        """
        # Simplified battery estimation
        total_energy = 0
        for step in plan.steps:
            energy_usage = self.estimate_step_energy(step)
            total_energy += energy_usage

        current_battery = self.get_current_battery_level()
        estimated_life = (current_battery * self.max_battery_capacity) - total_energy

        return estimated_life


class ContextAnalyzer:
    """
    Analyze current context for planning adaptation.
    """

    def __init__(self):
        self.sensors = ContextSensors()
        self.ontology = ContextOntology()

    def analyze(self, current_situation):
        """
        Analyze current situation context.

        Args:
            current_situation: Current sensor data and state

        Returns:
            dict: Context analysis with relevant factors
        """
        analysis = {}

        # Environmental analysis
        analysis.update(self.analyze_environment(current_situation))

        # Temporal analysis
        analysis.update(self.analyze_temporal_context(current_situation))

        # Social analysis
        analysis.update(self.analyze_social_context(current_situation))

        # Robot state analysis
        analysis.update(self.analyze_robot_state(current_situation))

        # Task-specific analysis
        analysis.update(self.analyze_task_context(current_situation))

        return analysis

    def analyze_environment(self, situation):
        """
        Analyze environmental context factors.
        """
        env_analysis = {
            "lighting_conditions": self.assess_lighting(situation),
            "acoustic_environment": self.assess_acoustics(situation),
            "spatial_layout": self.assess_layout(situation),
            "obstacle_map": self.map_obstacles(situation),
            "navigable_areas": self.identify_navigable_areas(situation)
        }

        return env_analysis

    def assess_lighting(self, situation):
        """
        Assess current lighting conditions.
        """
        illuminance = situation.get("ambient_light", {}).get("lux", 500)

        if illuminance < 50:
            return "poor"
        elif illuminance < 200:
            return "dim"
        elif illuminance < 1000:
            return "normal"
        else:
            return "bright"

    def assess_acoustics(self, situation):
        """
        Assess acoustic environment.
        """
        noise_level = situation.get("ambient_sound", {}).get("db", 40)

        if noise_level > 70:
            return "high"
        elif noise_level > 50:
            return "moderate"
        else:
            return "low"

    def assess_layout(self, situation):
        """
        Assess spatial layout.
        """
        # This would use semantic mapping data
        # For now, return simplified analysis
        return {
            "room_type": "office",  # Would come from semantic map
            "open_spaces": 3,
            "narrow_corridors": 1,
            "obstacle_density": 0.2
        }
```

## Learning and Adaptation

### Plan Learning and Improvement

```python
class AdaptiveTaskPlanner:
    """
    Task planner that learns and adapts from execution experience.
    """

    def __init__(self):
        self.experience_database = ExperienceDatabase()
        self.plan_optimizer = PlanOptimizer()
        self.failure_analyzer = FailureAnalyzer()

    def execute_and_learn(self, task_plan, execution_context):
        """
        Execute plan and learn from the experience.

        Args:
            task_plan: Plan to execute
            execution_context: Context during execution

        Returns:
            ExecutionResult: Result with learning data
        """
        start_time = time.time()

        try:
            # Execute plan
            execution_result = self.execute_plan(task_plan, execution_context)

            # Analyze execution for learning
            learning_data = self.analyze_execution(
                task_plan, execution_result, execution_context
            )

            # Update experience database
            self.experience_database.store_experience(
                task_plan=task_plan,
                execution_result=execution_result,
                context=execution_context,
                learning_data=learning_data
            )

            # Update plan templates if beneficial
            self.update_plan_templates(task_plan, learning_data)

            execution_result.learning_data = learning_data

            return execution_result

        except Exception as e:
            # Handle execution failure
            failure_data = self.analyze_failure(e, task_plan, execution_context)
            self.experience_database.store_failure(failure_data)

            raise e

    def analyze_execution(self, plan, result, context):
        """
        Analyze plan execution for improvement opportunities.
        """
        analysis = {
            "efficiency_metrics": self.calculate_efficiency_metrics(plan, result),
            "adaptation_opportunities": self.identify_adaptation_opportunities(plan, result),
            "optimization_suggestions": self.generate_optimization_suggestions(plan, result),
            "failure_modes": self.identify_failure_modes(result),
            "success_factors": self.identify_success_factors(result)
        }

        return analysis

    def calculate_efficiency_metrics(self, plan, result):
        """
        Calculate efficiency metrics for plan execution.
        """
        metrics = {
            "time_efficiency": plan.expected_duration / result.actual_duration,
            "energy_efficiency": result.energy_consumed / plan.expected_energy,
            "success_rate": result.successful_steps / len(plan.steps),
            "deviation_from_plan": self.calculate_plan_deviation(plan, result),
            "resource_utilization": self.calculate_resource_utilization(result)
        }

        return metrics

    def identify_adaptation_opportunities(self, plan, result):
        """
        Identify opportunities for plan adaptation.
        """
        opportunities = []

        # Check for repeated subtasks that could be optimized
        subtask_patterns = self.find_subtask_patterns(result.executed_steps)
        for pattern in subtask_patterns:
            if pattern.frequency > 2 and pattern.total_time_saved > 1.0:
                opportunities.append({
                    "type": "subtask_optimization",
                    "pattern": pattern,
                    "potential_savings": pattern.total_time_saved
                })

        # Check for frequently failing steps
        failure_patterns = self.find_failure_patterns(result.executed_steps)
        for pattern in failure_patterns:
            if pattern.failure_rate > 0.3:
                opportunities.append({
                    "type": "failure_mitigation",
                    "step": pattern.step_name,
                    "failure_rate": pattern.failure_rate,
                    "recommended_fix": self.recommend_fix(pattern)
                })

        # Check for redundant perception steps
        redundant_perceptions = self.find_redundant_perceptions(result.executed_steps)
        for redundancy in redundant_perceptions:
            opportunities.append({
                "type": "perception_optimization",
                "step": redundancy.step,
                "redundancy_count": redundancy.count,
                "time_saved": redundancy.time_saved
            })

        return opportunities

    def update_plan_templates(self, executed_plan, learning_data):
        """
        Update plan templates based on execution experience.
        """
        # Get similar plans from experience database
        similar_plans = self.experience_database.find_similar_plans(executed_plan)

        # Analyze common patterns and improvements
        common_improvements = self.analyze_common_improvements(
            [executed_plan] + similar_plans,
            learning_data
        )

        # Update template with beneficial changes
        template_id = self.get_plan_template_id(executed_plan)
        current_template = self.experience_database.get_template(template_id)

        updated_template = self.apply_beneficial_changes(
            current_template,
            common_improvements
        )

        # Validate updated template
        if self.validate_template(updated_template):
            self.experience_database.update_template(template_id, updated_template)

    def generate_adaptive_responses(self, unexpected_situation):
        """
        Generate adaptive responses to unexpected situations.
        """
        # Find similar past situations
        similar_situations = self.experience_database.find_similar_situations(
            unexpected_situation
        )

        # Extract successful adaptation patterns
        adaptation_patterns = self.extract_adaptation_patterns(similar_situations)

        # Generate possible responses
        possible_responses = []
        for pattern in adaptation_patterns:
            response = self.generate_response_from_pattern(
                pattern,
                unexpected_situation
            )
            possible_responses.append({
                "response": response,
                "success_probability": pattern.success_rate,
                "estimated_time": pattern.average_time
            })

        # Rank responses by expected utility
        ranked_responses = sorted(
            possible_responses,
            key=lambda x: x["success_probability"] * (1 / max(x["estimated_time"], 1)),
            reverse=True
        )

        return ranked_responses[:3]  # Return top 3 responses

    def learn_from_human_correction(self, observed_behavior, human_correction):
        """
        Learn from human corrections to robot behavior.
        """
        # Extract correction pattern
        correction_pattern = self.extract_correction_pattern(
            observed_behavior,
            human_correction
        )

        # Update behavioral models
        self.update_behavioral_model(correction_pattern)

        # Adjust plan generation heuristics
        self.adjust_generation_heuristics(correction_pattern)

        # Store for future reference
        self.experience_database.store_correction(
            observed_behavior=observed_behavior,
            human_correction=human_correction,
            learned_pattern=correction_pattern
        )

    def predict_execution_outcomes(self, plan, context):
        """
        Predict likely outcomes of plan execution.
        """
        # Use historical data to predict outcomes
        similar_executions = self.experience_database.find_similar_executions(
            plan_template=plan,
            context=context
        )

        if not similar_executions:
            # No historical data, use heuristic estimates
            return self.estimate_outcomes_heuristically(plan, context)

        # Analyze historical outcomes
        success_rate = np.mean([e.success for e in similar_executions])
        average_duration = np.mean([e.duration for e in similar_executions])
        common_failures = self.analyze_common_failures(similar_executions)

        prediction = {
            "predicted_success_rate": success_rate,
            "predicted_duration": average_duration,
            "likely_failures": common_failures,
            "confidence": min(len(similar_executions) / 10.0, 1.0),  # Confidence based on data amount
            "adaptation_recommendations": self.generate_adaptation_recommendations(
                plan, common_failures
            )
        }

        return prediction

    def generate_adaptation_recommendations(self, plan, likely_failures):
        """
        Generate recommendations for plan adaptation based on likely failures.
        """
        recommendations = []

        for failure in likely_failures:
            if failure.type == "navigation_failure":
                recommendation = {
                    "type": "navigation_adaptation",
                    "action": "increase_caution_level",
                    "target_step": "navigation_steps",
                    "expected_impact": "reduce_failure_probability_by_0.2"
                }
            elif failure.type == "manipulation_failure":
                recommendation = {
                    "type": "manipulation_adaptation",
                    "action": "add_pre_grasp_verification",
                    "target_step": "grasping_steps",
                    "expected_impact": "reduce_failure_probability_by_0.3"
                }
            elif failure.type == "perception_failure":
                recommendation = {
                    "type": "perception_adaptation",
                    "action": "increase_sensor_sampling_rate",
                    "target_step": "detection_steps",
                    "expected_impact": "improve_detection_rate_by_0.15"
                }

            recommendations.append(recommendation)

        return recommendations
```

## Error Handling and Recovery

### Robust Plan Execution

```python
class RobustPlanExecutor:
    """
    Robust plan execution with error handling and recovery.
    """

    def __init__(self):
        self.recovery_strategies = RecoveryStrategies()
        self.monitoring_system = ExecutionMonitor()
        self.fallback_manager = FallbackManager()

    def execute_plan_robustly(self, plan, execution_context):
        """
        Execute plan with robust error handling and recovery.

        Args:
            plan: Task plan to execute
            execution_context: Context for execution

        Returns:
            ExecutionResult: Result of execution with error info
        """
        result = ExecutionResult()
        result.start_time = time.time()

        for step_idx, step in enumerate(plan.steps):
            try:
                # Monitor execution state
                self.monitoring_system.start_monitoring_step(step)

                # Execute step
                step_result = self.execute_step_with_monitoring(
                    step,
                    execution_context,
                    step_idx
                )

                # Validate step completion
                if self.validate_step_completion(step, step_result):
                    result.add_successful_step(step, step_result)
                    execution_context.update_from_step(step_result)
                else:
                    # Handle validation failure
                    recovery_result = self.attempt_recovery(
                        step, step_result, execution_context, plan, step_idx
                    )

                    if recovery_result.success:
                        result.add_recovered_step(step, recovery_result)
                        execution_context.update_from_step(recovery_result)
                    else:
                        result.add_failed_step(step, recovery_result)
                        if self.should_abort_execution(plan, step_idx, result):
                            break

            except ExecutionError as e:
                # Handle execution error with recovery
                recovery_result = self.handle_execution_error(
                    step, e, execution_context, plan, step_idx
                )

                if recovery_result.success:
                    result.add_recovered_step(step, recovery_result)
                    execution_context.update_from_step(recovery_result)
                else:
                    result.add_failed_step(step, recovery_result)
                    if self.should_abort_execution(plan, step_idx, result):
                        break

            except Exception as e:
                # Handle unexpected error
                error_result = ExecutionResult(
                    success=False,
                    error_type="unexpected",
                    error_message=str(e)
                )
                result.add_failed_step(step, error_result)

                if self.should_abort_execution(plan, step_idx, result):
                    break

        result.end_time = time.time()
        return result

    def execute_step_with_monitoring(self, step, context, step_idx):
        """
        Execute a single step with monitoring.
        """
        # Set up monitors for this step
        monitors = self.monitoring_system.setup_monitors(step, context)

        try:
            # Start execution timer
            start_time = time.time()

            # Execute the step
            step_result = self.execute_single_step(step, context)

            # Stop monitors and collect data
            monitor_data = self.monitoring_system.stop_monitors(monitors)

            # Add execution metrics
            step_result.execution_time = time.time() - start_time
            step_result.monitor_data = monitor_data

            return step_result

        except Exception as e:
            # Collect error data from monitors
            error_data = self.monitoring_system.collect_error_data(monitors)

            raise ExecutionError(
                step=step,
                error_type="execution_failed",
                error_message=str(e),
                monitor_data=error_data
            )

    def attempt_recovery(self, failed_step, failure_result, context, full_plan, step_idx):
        """
        Attempt to recover from step failure.
        """
        # Determine appropriate recovery strategy
        recovery_strategy = self.recovery_strategies.select_strategy(
            failed_step, failure_result, context
        )

        if recovery_strategy is None:
            # No recovery strategy available
            return ExecutionResult(
                success=False,
                error_type="no_recovery",
                error_message="No suitable recovery strategy found"
            )

        # Apply recovery strategy
        recovery_result = recovery_strategy.apply(
            failed_step, failure_result, context, full_plan, step_idx
        )

        if recovery_result.success:
            # Recovery successful, continue with plan
            return recovery_result
        else:
            # Recovery failed, try alternative strategies
            alternative_recovery = self.try_alternative_strategies(
                failed_step, failure_result, context, full_plan, step_idx
            )

            return alternative_recovery

    def handle_execution_error(self, step, error, context, plan, step_idx):
        """
        Handle specific types of execution errors.
        """
        if isinstance(error, TimeoutError):
            return self.handle_timeout_error(step, error, context, plan, step_idx)
        elif isinstance(error, CollisionError):
            return self.handle_collision_error(step, error, context, plan, step_idx)
        elif isinstance(error, PerceptionError):
            return self.handle_perception_error(step, error, context, plan, step_idx)
        elif isinstance(error, ManipulationError):
            return self.handle_manipulation_error(step, error, context, plan, step_idx)
        else:
            # Generic error handling
            return self.handle_generic_error(step, error, context, plan, step_idx)

    def handle_timeout_error(self, step, error, context, plan, step_idx):
        """
        Handle timeout errors with appropriate recovery.
        """
        # Increase timeout and retry
        if step.retries < step.max_retries:
            step.timeout *= 1.5  # Increase timeout
            step.retries += 1

            # Retry the step
            retry_result = self.execute_step_with_monitoring(
                step, context, step_idx
            )

            return retry_result

        # If retries exhausted, try alternative approach
        alternative_result = self.find_alternative_approach(
            step, "timeout", context, plan, step_idx
        )

        return alternative_result

    def handle_collision_error(self, step, error, context, plan, step_idx):
        """
        Handle collision errors during navigation or manipulation.
        """
        # Update collision map
        self.update_collision_map(error.collision_info, context)

        # Plan alternative route/path
        if step.type == "navigation":
            alternative_route = self.find_alternative_navigation_path(
                current_position=error.position,
                original_goal=step.parameters.get("destination"),
                collision_map=context.collision_map
            )

            if alternative_route:
                # Modify step with new route
                step.parameters["route"] = alternative_route
                retry_result = self.execute_step_with_monitoring(
                    step, context, step_idx
                )
                return retry_result

        # If no alternative route, try different approach
        alternative_result = self.find_alternative_approach(
            step, "collision", context, plan, step_idx
        )

        return alternative_result

    def validate_step_completion(self, step, step_result):
        """
        Validate that step completed successfully according to requirements.
        """
        # Check success criteria
        if not step_result.success:
            return False

        # Verify post-conditions
        if step.postconditions:
            for condition in step.postconditions:
                if not self.verify_condition(condition, step_result, step):
                    return False

        # Check expected outcomes
        if step.expected_outcomes:
            for outcome in step.expected_outcomes:
                if not self.verify_outcome(outcome, step_result):
                    return False

        # Validate safety requirements
        if not self.validate_safety_requirements(step, step_result):
            return False

        return True

    def should_abort_execution(self, plan, current_step_idx, execution_result):
        """
        Determine if execution should be aborted based on failures.
        """
        # Check failure rate
        total_steps = len(plan.steps)
        failed_steps = len(execution_result.failed_steps)

        failure_rate = failed_steps / max(total_steps, 1)
        if failure_rate > 0.3:  # More than 30% failure rate
            return True

        # Check consecutive failures
        consecutive_failures = self.count_consecutive_failures(execution_result)
        if consecutive_failures >= 3:  # 3+ consecutive failures
            return True

        # Check critical step failures
        current_step = plan.steps[current_step_idx]
        if current_step.critical and execution_result.has_recent_failure():
            return True

        # Check resource exhaustion
        if self.resources_exhausted(execution_result):
            return True

        return False

    def find_alternative_approach(self, failed_step, error_type, context, plan, step_idx):
        """
        Find alternative approach when primary approach fails.
        """
        # Look for alternative implementations in knowledge base
        alternatives = self.fallback_manager.get_alternatives(
            step_type=failed_step.type,
            error_type=error_type,
            context=context
        )

        for alternative in alternatives:
            try:
                # Modify step with alternative approach
                alt_step = self.modify_step_for_alternative(failed_step, alternative)

                # Execute alternative
                alt_result = self.execute_step_with_monitoring(
                    alt_step, context, step_idx
                )

                if alt_result.success:
                    return alt_result

            except Exception:
                continue  # Try next alternative

        # If no alternatives work, return failure
        return ExecutionResult(
            success=False,
            error_type="alternative_failed",
            error_message="All alternative approaches failed"
        )


class RecoveryStrategies:
    """
    Collection of recovery strategies for different failure types.
    """

    def __init__(self):
        self.strategies = {
            "retry": RetryStrategy(),
            "fallback": FallbackStrategy(),
            "replan": ReplanningStrategy(),
            "skip": SkippingStrategy(),
            "manual": ManualInterventionStrategy()
        }

    def select_strategy(self, failed_step, failure_result, context):
        """
        Select appropriate recovery strategy based on failure type and context.
        """
        failure_type = self.classify_failure(failure_result)
        step_type = failed_step.type

        # Rule-based strategy selection
        if failure_type == "temporary":
            return self.strategies["retry"]
        elif failure_type == "permanent_local":
            if step_type == "navigation":
                return self.strategies["replan"]
            elif step_type == "manipulation":
                return self.strategies["fallback"]
        elif failure_type == "resource_unavailable":
            return self.strategies["fallback"]
        elif failure_type == "critical":
            return self.strategies["manual"]  # Require human intervention

        # Default to fallback strategy
        return self.strategies["fallback"]

    def classify_failure(self, failure_result):
        """
        Classify failure type for appropriate recovery.
        """
        error_msg = failure_result.error_message.lower()

        if any(keyword in error_msg for keyword in ["timeout", "stuck", "blocked"]):
            return "temporary"
        elif any(keyword in error_msg for keyword in ["broken", "malfunction", "unreachable"]):
            return "permanent_local"
        elif any(keyword in error_msg for keyword in ["battery", "memory", "cpu"]):
            return "resource_unavailable"
        elif any(keyword in error_msg for keyword in ["safety", "emergency", "critical"]):
            return "critical"
        else:
            return "unknown"


class ExecutionMonitor:
    """
    Monitor plan execution for anomalies and failures.
    """

    def __init__(self):
        self.anomaly_detectors = {
            "timing": TimingAnomalyDetector(),
            "resource": ResourceAnomalyDetector(),
            "behavior": BehaviorAnomalyDetector()
        }

    def setup_monitors(self, step, context):
        """
        Set up monitors for specific step execution.
        """
        monitors = {}

        # Timing monitor
        if step.expected_duration:
            monitors["timing"] = self.anomaly_detectors["timing"].setup_monitor(
                expected_duration=step.expected_duration
            )

        # Resource monitor
        monitors["resource"] = self.anomaly_detectors["resource"].setup_monitor(
            step_type=step.type,
            context=context
        )

        # Behavior monitor
        if step.monitored_behaviors:
            monitors["behavior"] = self.anomaly_detectors["behavior"].setup_monitor(
                expected_behaviors=step.monitored_behaviors
            )

        return monitors

    def stop_monitors(self, monitors):
        """
        Stop monitors and collect data.
        """
        monitor_data = {}
        for monitor_type, monitor in monitors.items():
            monitor_data[monitor_type] = monitor.collect_data()
            monitor.cleanup()

        return monitor_data

    def detect_anomalies(self, monitor_data, step, context):
        """
        Detect anomalies in execution based on monitor data.
        """
        anomalies = []

        for monitor_type, data in monitor_data.items():
            detector = self.anomaly_detectors[monitor_type]
            step_anomalies = detector.detect_anomalies(data, step, context)
            anomalies.extend(step_anomalies)

        return anomalies
```

## Assessment and Evaluation

### Planning Quality Metrics

```python
class PlanningEvaluator:
    """
    Evaluate quality of task plans and execution.
    """

    def __init__(self):
        self.metrics = {
            "completeness": self.evaluate_completeness,
            "feasibility": self.evaluate_feasibility,
            "optimality": self.evaluate_optimality,
            "robustness": self.evaluate_robustness,
            "safety": self.evaluate_safety
        }

    def evaluate_plan(self, plan, context=None):
        """
        Evaluate plan quality across multiple dimensions.

        Args:
            plan: Task plan to evaluate
            context: Execution context

        Returns:
            dict: Evaluation results with scores
        """
        evaluation = {}

        for metric_name, evaluator in self.metrics.items():
            evaluation[metric_name] = evaluator(plan, context)

        # Overall quality score
        evaluation["overall_quality"] = self.calculate_overall_quality(evaluation)

        return evaluation

    def evaluate_completeness(self, plan, context):
        """
        Evaluate if plan is complete and addresses all requirements.
        """
        score = 0
        max_score = 10

        # Check if all goal conditions are addressed
        if plan.goal_conditions:
            satisfied_conditions = 0
            for condition in plan.goal_conditions:
                if self.condition_is_addressed(condition, plan.steps):
                    satisfied_conditions += 1

            score += (satisfied_conditions / len(plan.goal_conditions)) * 4

        # Check for logical sequence
        if self.has_logical_sequencing(plan.steps):
            score += 3

        # Check for necessary preconditions
        if self.all_preconditions_covered(plan.steps):
            score += 3

        return min(score, max_score)

    def evaluate_feasibility(self, plan, context):
        """
        Evaluate if plan is feasible given current context.
        """
        score = 0
        max_score = 10

        # Check resource availability
        required_resources = self.extract_required_resources(plan.steps)
        available_resources = context.get("available_resources", {})
        resource_feasibility = self.calculate_resource_feasibility(
            required_resources, available_resources
        )
        score += resource_feasibility * 4

        # Check physical constraints
        if self.respects_physical_constraints(plan.steps, context):
            score += 3

        # Check temporal feasibility
        if self.within_time_constraints(plan, context):
            score += 3

        return min(score, max_score)

    def evaluate_optimality(self, plan, context):
        """
        Evaluate if plan is optimal in terms of efficiency.
        """
        score = 0
        max_score = 10

        # Compare to theoretical minimum steps
        theoretical_min = self.calculate_theoretical_minimum_steps(plan)
        if len(plan.steps) <= theoretical_min * 1.2:  # Allow 20% overhead
            score += 4
        elif len(plan.steps) <= theoretical_min * 1.5:
            score += 2

        # Evaluate energy efficiency
        estimated_energy = self.estimate_plan_energy(plan, context)
        if self.is_energy_efficient(estimated_energy, context):
            score += 3

        # Evaluate path efficiency (for navigation tasks)
        if self.has_navigation_steps(plan):
            path_efficiency = self.evaluate_path_efficiency(plan, context)
            score += path_efficiency * 3

        return min(score, max_score)

    def evaluate_robustness(self, plan, context):
        """
        Evaluate plan's ability to handle uncertainties.
        """
        score = 0
        max_score = 10

        # Check for error handling steps
        error_handling_steps = [s for s in plan.steps if s.type == "error_handling"]
        if len(error_handling_steps) > 0:
            score += 3

        # Check for alternative paths/strategies
        if self.has_alternatives_in_plan(plan):
            score += 2

        # Evaluate sensor validation steps
        validation_steps = [s for s in plan.steps if s.type == "validation"]
        if len(validation_steps) > 0:
            score += 2

        # Check for adaptive elements
        if self.has_adaptive_elements(plan):
            score += 3

        return min(score, max_score)

    def evaluate_safety(self, plan, context):
        """
        Evaluate plan safety for humanoid robot execution.
        """
        score = 0
        max_score = 10

        # Check for safety verification steps
        safety_checks = [s for s in plan.steps if "safety" in s.tags]
        if len(safety_checks) > 0:
            score += 3

        # Evaluate balance preservation (for humanoid)
        if self.preserves_balance_in_manipulation(plan.steps):
            score += 3

        # Check for collision avoidance
        if self.has_collision_avoidance(plan.steps):
            score += 2

        # Evaluate emergency stops
        if self.has_emergency_procedures(plan.steps):
            score += 2

        return min(score, max_score)

    def calculate_overall_quality(self, evaluation_scores):
        """
        Calculate overall plan quality score.
        """
        # Weighted average of all metrics
        weights = {
            "completeness": 0.2,
            "feasibility": 0.25,
            "optimality": 0.2,
            "robustness": 0.2,
            "safety": 0.15
        }

        overall_score = 0
        for metric, score in evaluation_scores.items():
            if metric in weights:
                overall_score += score * weights[metric]

        return overall_score


def main():
    """
    Example usage of LLM-powered task planning.
    """
    # Initialize LLM task planner
    llm_planner = LLMTaskPlanner(
        llm_client=openai.Client(api_key=os.getenv("OPENAI_API_KEY")),
        robot_capabilities=load_robot_capabilities(),
        environment_knowledge=load_environment_knowledge()
    )

    # Example command
    command = "Go to the kitchen, find a red apple, and bring it to me in the living room"

    # Get current state
    current_state = get_robot_state()

    # Generate plan
    task_plan = llm_planner.plan_task(command, current_state)

    # Execute plan
    executor = RobustPlanExecutor()
    execution_result = executor.execute_plan_robustly(task_plan, current_state)

    # Evaluate execution
    evaluator = PlanningEvaluator()
    evaluation = evaluator.evaluate_plan(task_plan, current_state)

    print(f"Plan execution result: {execution_result.success}")
    print(f"Plan quality score: {evaluation['overall_quality']:.2f}/10.0")

    return execution_result


if __name__ == "__main__":
    main()
```

## Learning Outcomes

After completing this module, students will be able to:
1. Design and implement LLM-powered task planning systems
2. Create hierarchical task decomposition pipelines
3. Implement context-aware planning with adaptation
4. Build robust execution systems with error handling
5. Evaluate plan quality and optimize for performance
6. Integrate planning with perception and control systems

## Next Steps

Continue to [VLA Integration](./vla-integration.md) to learn about combining vision, language, and action systems for complete AI-powered humanoid robot behavior.