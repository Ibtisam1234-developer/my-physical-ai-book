# Whisper Voice Command Integration

## Introduction to Voice Command Systems

Voice command systems enable natural human-robot interaction through spoken language. For humanoid robots, voice interfaces provide an intuitive way for humans to issue commands, ask questions, and engage in conversation with the robot.

### Why Voice Commands for Humanoid Robots?

Humanoid robots are designed to interact with humans in natural environments. Voice commands provide several advantages:

- **Natural Interaction**: Humans naturally communicate through speech
- **Hands-Free Operation**: Users don't need to touch interfaces
- **Accessibility**: Supports users with mobility limitations
- **Intuitive Control**: No need to learn complex interfaces
- **Multimodal Integration**: Works with vision and action systems

## OpenAI Whisper for Robot Voice Commands

OpenAI Whisper is a state-of-the-art speech recognition model that can convert spoken language to text. It's particularly well-suited for robotics applications due to:

- **High Accuracy**: Excellent performance across accents and languages
- **Robustness**: Performs well in noisy environments
- **Real-time Capability**: Can operate with low latency
- **Open Source**: Free to use and customize
- **Multilingual Support**: Works with multiple languages

### Whisper Architecture for Robotics

```python
class WhisperRobotInterface:
    """
    Whisper-based voice command interface for humanoid robots.
    """

    def __init__(self, model_size="medium"):
        # Load Whisper model
        self.whisper_model = whisper.load_model(model_size)

        # Voice activity detection
        self.vad_detector = VoiceActivityDetector()

        # Wake word detection
        self.wake_word_detector = WakeWordDetector(
            wake_words=["hey robot", "robot", "attention"]
        )

        # Command parser
        self.command_parser = RobotCommandParser()

        # Audio input configuration
        self.audio_input = AudioInput(
            sample_rate=16000,
            channels=1,
            chunk_size=1024
        )

        # State management
        self.is_listening = False
        self.command_history = []

    def process_audio_stream(self):
        """
        Continuously process audio stream for voice commands.
        """
        audio_queue = queue.Queue()

        def audio_callback(indata, frames, time, status):
            audio_queue.put(indata.copy())

        # Start audio stream
        with sd.InputStream(
            callback=audio_callback,
            channels=1,
            samplerate=16000,
            blocksize=1024
        ):
            while True:
                # Get audio data
                audio_data = audio_queue.get()

                # Check for wake word
                if self.wake_word_detector.detect(audio_data):
                    self.start_listening()

                # Process voice commands if listening
                if self.is_listening:
                    # Detect voice activity
                    if self.vad_detector.is_speech(audio_data):
                        # Record command
                        command_audio = self.record_command()

                        # Transcribe with Whisper
                        transcription = self.transcribe_audio(command_audio)

                        # Parse command
                        robot_command = self.command_parser.parse(transcription)

                        # Execute command
                        self.execute_command(robot_command)

                        # Stop listening
                        self.stop_listening()

    def transcribe_audio(self, audio_data):
        """
        Transcribe audio to text using Whisper.

        Args:
            audio_data: Raw audio data

        Returns:
            text: Transcribed text
        """
        # Convert audio to appropriate format
        audio_tensor = self.preprocess_audio(audio_data)

        # Transcribe with Whisper
        result = self.whisper_model.transcribe(
            audio_tensor,
            language="en",
            temperature=0.0,
            best_of=1,
            patience=None,
            suppress_tokens="-1",
            initial_prompt="Humanoid robot commands:",
            condition_on_previous_text=False
        )

        return result["text"]

    def preprocess_audio(self, audio_data):
        """
        Preprocess audio data for Whisper.
        """
        # Convert to float32
        audio_float = audio_data.astype(np.float32)

        # Normalize
        audio_normalized = audio_float / np.max(np.abs(audio_float))

        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_normalized)

        return audio_tensor
```

## Advanced Voice Command Processing

### Intent Recognition and Command Parsing

```python
class RobotCommandParser:
    """
    Parse voice commands into robot actions.
    """

    def __init__(self):
        # Define command intents
        self.intents = {
            "navigation": {
                "patterns": [
                    r"go to (?P<location>\w+)",
                    r"move to (?P<location>\w+)",
                    r"walk to (?P<location>\w+)",
                    r"navigate to (?P<location>\w+)"
                ]
            },
            "manipulation": {
                "patterns": [
                    r"pick up (?P<object>\w+)",
                    r"grasp (?P<object>\w+)",
                    r"take (?P<object>\w+)",
                    r"get (?P<object>\w+)",
                    r"put (?P<object>\w+) in (?P<container>\w+)"
                ]
            },
            "information": {
                "patterns": [
                    r"what is (?P<query>.+)",
                    r"tell me about (?P<query>.+)",
                    r"describe (?P<query>.+)",
                    r"explain (?P<query>.+)"
                ]
            },
            "social_interaction": {
                "patterns": [
                    r"hello",
                    r"hi",
                    r"good morning",
                    r"good afternoon",
                    r"good evening",
                    r"how are you",
                    r"what's up"
                ]
            }
        }

        # Entity extraction models
        self.entity_extractor = EntityExtractor()

    def parse(self, transcription):
        """
        Parse voice transcription into robot command.

        Args:
            transcription: Transcribed text from Whisper

        Returns:
            command: Parsed robot command
        """
        transcription = transcription.lower().strip()

        # Match intents
        for intent_name, intent_config in self.intents.items():
            for pattern in intent_config["patterns"]:
                match = re.search(pattern, transcription)
                if match:
                    # Extract entities
                    entities = match.groupdict()

                    # Additional entity extraction
                    entities.update(self.entity_extractor.extract(transcription))

                    # Create command
                    command = {
                        "intent": intent_name,
                        "entities": entities,
                        "original_text": transcription
                    }

                    return command

        # If no intent matched, treat as information query
        return {
            "intent": "information",
            "entities": {"query": transcription},
            "original_text": transcription
        }

    def validate_command(self, command):
        """
        Validate parsed command before execution.
        """
        # Check if command is safe
        if self.is_command_safe(command):
            return True

        # Check if command is feasible
        if self.is_command_feasible(command):
            return True

        return False

    def is_command_safe(self, command):
        """
        Check if command is safe for robot to execute.
        """
        # Safety validation logic
        unsafe_keywords = ["harm", "destroy", "attack", "break"]

        for keyword in unsafe_keywords:
            if keyword in command["original_text"]:
                return False

        return True

    def is_command_feasible(self, command):
        """
        Check if robot can physically execute the command.
        """
        # Feasibility validation based on robot capabilities
        if command["intent"] == "manipulation":
            # Check if object exists in environment
            if "object" in command["entities"]:
                obj_name = command["entities"]["object"]
                if not self.object_exists(obj_name):
                    return False

        return True
```

### Voice Activity Detection (VAD)

```python
class VoiceActivityDetector:
    """
    Detect voice activity in audio streams.
    """

    def __init__(self, threshold=0.3, silence_duration=1.0):
        self.threshold = threshold
        self.silence_duration = silence_duration
        self.energy_history = []
        self.min_energy_samples = 100

    def is_speech(self, audio_data):
        """
        Detect if audio contains speech.

        Args:
            audio_data: Audio data to analyze

        Returns:
            bool: True if speech detected, False otherwise
        """
        # Calculate energy of audio frame
        energy = self.calculate_energy(audio_data)

        # Add to history
        self.energy_history.append(energy)
        if len(self.energy_history) > self.min_energy_samples:
            self.energy_history.pop(0)

        # Check if energy exceeds threshold
        if energy > self.threshold:
            return True

        # Check for recent speech activity
        recent_energies = self.energy_history[-int(self.silence_duration * 100):]  # Assuming 100Hz sampling
        if any(e > self.threshold for e in recent_energies):
            return True

        return False

    def calculate_energy(self, audio_data):
        """
        Calculate energy of audio signal.
        """
        # Convert to numpy array if needed
        if not isinstance(audio_data, np.ndarray):
            audio_data = np.array(audio_data)

        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio_data ** 2))
        return rms

    def adaptive_threshold(self):
        """
        Calculate adaptive threshold based on background noise.
        """
        if len(self.energy_history) < self.min_energy_samples:
            return self.threshold

        # Calculate threshold as percentile of background energy
        background_energy = np.percentile(
            self.energy_history[:self.min_energy_samples],
            90
        )
        adaptive_threshold = background_energy * 1.5  # 50% above background

        return min(adaptive_threshold, 1.0)  # Cap at 1.0
```

### Wake Word Detection

```python
class WakeWordDetector:
    """
    Detect wake words to activate voice command system.
    """

    def __init__(self, wake_words, detection_threshold=0.8):
        self.wake_words = [word.lower() for word in wake_words]
        self.detection_threshold = detection_threshold

        # Use keyword spotting model
        self.keyword_model = self.load_keyword_model()

    def load_keyword_model(self):
        """
        Load keyword spotting model.
        """
        # Could use models like Picovoice Porcupine or custom models
        # For now, we'll use a simple approach with Whisper
        return whisper.load_model("tiny.en")  # Smaller model for keyword detection

    def detect(self, audio_data):
        """
        Detect wake words in audio data.

        Args:
            audio_data: Audio data to analyze

        Returns:
            bool: True if wake word detected, False otherwise
        """
        # Transcribe short audio segment
        transcription = self.transcribe_short_segment(audio_data)

        # Check for wake words
        transcription_lower = transcription.lower()

        for wake_word in self.wake_words:
            if wake_word in transcription_lower:
                return True

        return False

    def transcribe_short_segment(self, audio_data):
        """
        Transcribe short audio segment for wake word detection.
        """
        # Use Whisper for short segment transcription
        audio_tensor = self.preprocess_audio_for_detection(audio_data)

        result = self.keyword_model.transcribe(
            audio_tensor,
            language="en",
            temperature=0.2,
            best_of=1
        )

        return result["text"]

    def preprocess_audio_for_detection(self, audio_data):
        """
        Preprocess audio for wake word detection.
        """
        # Convert to appropriate format
        if isinstance(audio_data, np.ndarray):
            audio_tensor = torch.from_numpy(audio_data.astype(np.float32))
        else:
            audio_tensor = torch.tensor(audio_data, dtype=torch.float32)

        # Normalize
        audio_tensor = audio_tensor / torch.max(torch.abs(audio_tensor))

        return audio_tensor
```

## Real-Time Voice Command System

### Streaming Audio Processing

```python
class RealTimeVoiceCommandSystem:
    """
    Real-time voice command system with Whisper integration.
    """

    def __init__(self, robot_interface):
        self.whisper_interface = WhisperRobotInterface()
        self.robot_interface = robot_interface

        # Audio stream parameters
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.buffer_size = 4 * self.chunk_size  # 4 chunks for processing

        # Audio buffer
        self.audio_buffer = np.array([])

        # Processing flags
        self.is_processing = False
        self.command_queue = queue.Queue()

        # Performance monitoring
        self.stats = {
            "transcription_latency": [],
            "command_accuracy": [],
            "false_wake_rate": []
        }

    def start_listening(self):
        """
        Start real-time voice command processing.
        """
        print("Starting voice command system...")

        # Start audio input stream
        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            callback=self.audio_callback
        )

        self.stream.start()

        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_commands)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        print("Voice command system active")

    def audio_callback(self, indata, frames, time, status):
        """
        Callback for audio input stream.
        """
        # Add to buffer
        self.audio_buffer = np.concatenate([self.audio_buffer, indata.flatten()])

        # Process when buffer is large enough
        if len(self.audio_buffer) >= self.buffer_size:
            self.process_audio_buffer()

    def process_audio_buffer(self):
        """
        Process accumulated audio buffer.
        """
        if self.is_processing:
            return  # Skip if already processing

        self.is_processing = True

        try:
            # Process for wake word detection
            if self.whisper_interface.wake_word_detector.detect(self.audio_buffer):
                print("Wake word detected! Listening...")

                # Record full command
                command_audio = self.record_full_command()

                # Transcribe
                start_time = time.time()
                transcription = self.whisper_interface.transcribe_audio(command_audio)
                latency = time.time() - start_time

                # Update stats
                self.stats["transcription_latency"].append(latency)

                print(f"Transcribed: '{transcription}' (Latency: {latency:.2f}s)")

                # Parse and queue command
                command = self.whisper_interface.command_parser.parse(transcription)
                self.command_queue.put(command)

        except Exception as e:
            print(f"Error processing audio: {e}")
        finally:
            # Clear buffer and reset processing flag
            self.audio_buffer = np.array([])
            self.is_processing = False

    def record_full_command(self):
        """
        Record a complete voice command after wake word.
        """
        # Record for a few seconds after wake word
        recording_duration = 3.0  # seconds
        samples_needed = int(recording_duration * self.sample_rate)

        recorded_audio = []
        start_time = time.time()

        while time.time() - start_time < recording_duration:
            if len(self.audio_buffer) > 0:
                # Take chunk from buffer
                chunk_size = min(len(self.audio_buffer), self.chunk_size)
                chunk = self.audio_buffer[:chunk_size]
                recorded_audio.extend(chunk)
                self.audio_buffer = self.audio_buffer[chunk_size:]

            time.sleep(0.01)  # Small delay

        return np.array(recorded_audio)

    def process_commands(self):
        """
        Process queued commands in separate thread.
        """
        while True:
            try:
                # Get command from queue
                command = self.command_queue.get(timeout=1.0)

                # Validate command
                if self.whisper_interface.command_parser.validate_command(command):
                    print(f"Executing command: {command}")

                    # Execute via robot interface
                    success = self.robot_interface.execute_command(command)

                    # Provide feedback
                    self.provide_feedback(success, command)
                else:
                    print(f"Invalid command: {command}")
                    self.provide_feedback(False, command, reason="invalid")

            except queue.Empty:
                continue  # No commands to process
            except Exception as e:
                print(f"Error processing command: {e}")

    def provide_feedback(self, success, command, reason=None):
        """
        Provide audio feedback to user.
        """
        if success:
            feedback_text = f"I have {command['intent']} as requested."
        else:
            if reason == "invalid":
                feedback_text = "I couldn't understand that command."
            else:
                feedback_text = "I couldn't execute that command."

        # Convert to speech (using text-to-speech)
        self.speak_response(feedback_text)

    def speak_response(self, text):
        """
        Speak response using text-to-speech.
        """
        # Could use pyttsx3, gTTS, or other TTS engines
        # Implementation would depend on available TTS system
        pass

    def stop_listening(self):
        """
        Stop the voice command system.
        """
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
```

## Integration with VLA Systems

### Voice-to-VLA Pipeline

```python
class VoiceToVLA:
    """
    Bridge between voice commands and VLA systems.
    """

    def __init__(self, vla_model, voice_interface):
        self.vla_model = vla_model
        self.voice_interface = voice_interface

        # Scene understanding for context
        self.scene_analyzer = SceneAnalyzer()

        # Command executor
        self.command_executor = RobotCommandExecutor()

    def process_voice_command(self, voice_command):
        """
        Process voice command through VLA pipeline.

        Args:
            voice_command: Parsed voice command dictionary

        Returns:
            execution_result: Result of command execution
        """
        # Get current scene context
        current_scene = self.scene_analyzer.get_current_scene()

        # Integrate voice command with scene context
        vla_input = self.prepare_vla_input(voice_command, current_scene)

        # Execute through VLA system
        action_sequence = self.vla_model(vla_input)

        # Execute actions on robot
        execution_result = self.command_executor.execute_sequence(action_sequence)

        return execution_result

    def prepare_vla_input(self, voice_command, scene_context):
        """
        Prepare input for VLA model from voice command and scene.

        Args:
            voice_command: Parsed voice command
            scene_context: Current scene understanding

        Returns:
            vla_input: Dictionary with vision, language, and context
        """
        # Extract language component
        language_instruction = self.format_language_instruction(voice_command)

        # Use scene context for vision component
        vision_input = scene_context["visual_features"]

        # Add spatial and object context
        spatial_context = self.extract_spatial_context(voice_command, scene_context)

        vla_input = {
            "instruction": language_instruction,
            "vision": vision_input,
            "spatial_context": spatial_context,
            "scene_objects": scene_context["detected_objects"],
            "robot_state": scene_context["robot_state"]
        }

        return vla_input

    def format_language_instruction(self, voice_command):
        """
        Format voice command into language instruction for VLA.
        """
        intent = voice_command["intent"]
        entities = voice_command["entities"]

        if intent == "navigation":
            if "location" in entities:
                instruction = f"Navigate to the {entities['location']}."
            else:
                instruction = voice_command["original_text"]

        elif intent == "manipulation":
            instruction = self.create_manipulation_instruction(entities)

        elif intent == "information":
            instruction = f"Provide information about {entities.get('query', 'the environment')}."

        else:
            instruction = voice_command["original_text"]

        return instruction

    def create_manipulation_instruction(self, entities):
        """
        Create detailed manipulation instruction from entities.
        """
        instruction_parts = []

        if "object" in entities:
            obj = entities["object"]
            instruction_parts.append(f"Manipulate the {obj}")

            if "action" in entities:
                action = entities["action"]
                instruction_parts.append(f"by {action}ing it")

            if "destination" in entities:
                dest = entities["destination"]
                instruction_parts.append(f"and place it in the {dest}")

        return " ".join(instruction_parts) + "."

    def extract_spatial_context(self, voice_command, scene_context):
        """
        Extract spatial context relevant to the voice command.
        """
        spatial_context = {}

        # Identify spatial entities in command
        command_text = voice_command["original_text"]
        spatial_entities = self.identify_spatial_entities(command_text)

        # Get spatial relationships from scene
        for entity in spatial_entities:
            if entity in scene_context["detected_objects"]:
                obj_info = scene_context["detected_objects"][entity]
                spatial_context[entity] = {
                    "position": obj_info["position"],
                    "orientation": obj_info["orientation"],
                    "relationships": obj_info.get("spatial_relationships", {})
                }

        return spatial_context

    def identify_spatial_entities(self, text):
        """
        Identify spatially relevant entities in text.
        """
        # Simple pattern matching for spatial entities
        spatial_patterns = [
            r"\b(table|chair|door|window|counter|shelf|cabinet)\b",
            r"\b(room|kitchen|living room|bedroom|office|hallway)\b",
            r"\b(left|right|front|back|center|middle)\b"
        ]

        entities = []
        for pattern in spatial_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.extend(matches)

        return list(set(entities))  # Remove duplicates
```

## Privacy and Security Considerations

### Secure Voice Processing

```python
class SecureVoiceProcessor:
    """
    Secure voice processing with privacy protections.
    """

    def __init__(self):
        # Encryption for audio data
        self.audio_encryptor = AudioEncryptor()

        # Privacy filters
        self.privacy_filter = PrivacyFilter()

        # Local processing where possible
        self.local_models = self.load_local_models()

    def process_privately(self, audio_data):
        """
        Process audio data with privacy protections.
        """
        # Encrypt audio before processing
        encrypted_audio = self.audio_encryptor.encrypt(audio_data)

        # Filter sensitive information
        filtered_audio = self.privacy_filter.remove_sensitive_content(encrypted_audio)

        # Process locally when possible
        if self.can_process_locally(filtered_audio):
            result = self.process_locally(filtered_audio)
        else:
            result = self.process_securely(filtered_audio)

        return result

    def can_process_locally(self, audio_data):
        """
        Determine if audio can be processed locally.
        """
        # Check if command is simple enough for local model
        return len(audio_data) < 10 * 16000  # 10 seconds of audio

    def process_locally(self, encrypted_audio):
        """
        Process audio using local models.
        """
        # Use local Whisper model
        transcription = self.local_models["whisper"].transcribe(encrypted_audio)

        # Use local command parser
        command = self.local_models["parser"].parse(transcription)

        return command

    def process_securely(self, encrypted_audio):
        """
        Process audio securely with external services.
        """
        # Send to secure processing service
        # (with encrypted data and proper authentication)
        pass
```

## Performance Optimization

### Optimized Whisper Processing

```python
class OptimizedWhisperProcessor:
    """
    Optimized Whisper processing for real-time performance.
    """

    def __init__(self, model_size="small"):
        # Use smaller model for faster processing
        self.model = whisper.load_model(model_size)

        # Enable GPU acceleration if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        # Use half precision for faster inference
        self.model = self.model.half()

        # Compile model for performance
        if hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)

    def transcribe_streaming(self, audio_stream):
        """
        Transcribe audio stream in chunks for real-time performance.
        """
        # Process audio in overlapping chunks
        chunk_size = 16000 * 2  # 2 seconds
        overlap = 0.5  # 50% overlap

        for chunk in self.chunk_audio_stream(audio_stream, chunk_size, overlap):
            # Transcribe chunk
            result = self.model.transcribe(chunk)

            # Yield partial results
            yield result["text"]

    def chunk_audio_stream(self, audio_stream, chunk_size, overlap_ratio):
        """
        Create overlapping chunks from audio stream.
        """
        overlap_size = int(chunk_size * overlap_ratio)
        step_size = chunk_size - overlap_size

        for i in range(0, len(audio_stream) - chunk_size, step_size):
            chunk = audio_stream[i:i + chunk_size]

            # Apply fade in/out to reduce artifacts
            chunk = self.apply_fade(chunk, overlap_size)

            yield chunk

    def apply_fade(self, audio_chunk, fade_length):
        """
        Apply fade in/out to reduce discontinuity artifacts.
        """
        fade_curve = np.hanning(2 * fade_length)
        fade_in = fade_curve[:fade_length]
        fade_out = fade_curve[fade_length:]

        audio_chunk[:fade_length] *= fade_in
        audio_chunk[-fade_length:] *= fade_out

        return audio_chunk
```

## Testing and Evaluation

### Voice Command Testing Framework

```python
class VoiceCommandTester:
    """
    Testing framework for voice command systems.
    """

    def __init__(self, system_under_test):
        self.system = system_under_test
        self.test_scenarios = self.load_test_scenarios()

    def load_test_scenarios(self):
        """
        Load test scenarios for voice commands.
        """
        return [
            {
                "command": "Go to the kitchen",
                "expected_intent": "navigation",
                "expected_entities": {"location": "kitchen"},
                "environment": "household"
            },
            {
                "command": "Pick up the red cup",
                "expected_intent": "manipulation",
                "expected_entities": {"object": "red cup"},
                "environment": "table_setting"
            },
            {
                "command": "What is the weather like?",
                "expected_intent": "information",
                "expected_entities": {"query": "weather"},
                "environment": "general"
            }
        ]

    def run_comprehensive_tests(self):
        """
        Run comprehensive tests on voice command system.
        """
        results = {
            "accuracy": 0,
            "latency": [],
            "robustness": 0,
            "false_positive_rate": 0
        }

        for scenario in self.test_scenarios:
            test_result = self.run_scenario_test(scenario)
            results = self.aggregate_results(results, test_result)

        return results

    def run_scenario_test(self, scenario):
        """
        Run a single test scenario.
        """
        # Simulate voice command
        audio_simulation = self.simulate_audio_command(scenario["command"])

        # Process through system
        start_time = time.time()
        parsed_command = self.system.process_audio_stream(audio_simulation)
        latency = time.time() - start_time

        # Evaluate results
        accuracy = self.evaluate_command_accuracy(parsed_command, scenario)
        robustness = self.evaluate_robustness(scenario)

        return {
            "accuracy": accuracy,
            "latency": latency,
            "robustness": robustness
        }

    def simulate_audio_command(self, text_command):
        """
        Simulate audio from text command (for testing).
        """
        # Use TTS to generate audio from text
        # This would normally come from real microphone input
        pass

    def evaluate_command_accuracy(self, parsed_command, expected_scenario):
        """
        Evaluate accuracy of command parsing.
        """
        expected_intent = expected_scenario["expected_intent"]
        expected_entities = expected_scenario["expected_entities"]

        # Check intent accuracy
        intent_correct = parsed_command["intent"] == expected_intent

        # Check entity accuracy
        entity_correct = True
        for key, expected_value in expected_entities.items():
            if key not in parsed_command["entities"]:
                entity_correct = False
                break
            if parsed_command["entities"][key] != expected_value:
                entity_correct = False
                break

        return intent_correct and entity_correct

    def evaluate_robustness(self, scenario):
        """
        Evaluate robustness to noise and variations.
        """
        # Test with various noise conditions
        noise_levels = [0.0, 0.1, 0.2, 0.3]
        successes = 0

        for noise_level in noise_levels:
            noisy_audio = self.add_noise(scenario["command"], noise_level)
            try:
                result = self.system.process_audio_stream(noisy_audio)
                if self.evaluate_command_accuracy(result, scenario):
                    successes += 1
            except:
                pass  # Count as failure

        return successes / len(noise_levels)
```

## Learning Outcomes

After completing this section, students will be able to:
1. Integrate OpenAI Whisper for speech-to-text conversion
2. Implement wake word detection for activation
3. Design voice activity detection systems
4. Parse voice commands into robot actions
5. Optimize voice processing for real-time performance
6. Ensure privacy and security in voice systems
7. Test and evaluate voice command accuracy

## Next Steps

Continue to [Large Language Model Task Planning](./llm-task-planning.md) to learn how to use LLMs for high-level task planning in VLA systems.