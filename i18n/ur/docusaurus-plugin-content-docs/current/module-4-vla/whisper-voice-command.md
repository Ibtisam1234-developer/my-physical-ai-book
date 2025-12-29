---
slug: /module-4-vla/whisper-voice-command
title: "Whisper Voice Command Integration"
hide_table_of_contents: false
---

# Whisper Voice Command Integration (آواز کے حکم کی انٹیگریشن)

Voice command systems natural human-robot interaction enable کرتے ہیں through spoken language۔ Humanoid robots کے لیے، voice interfaces humans کے لیے intuitive way provide کرتے ہیں commands issue کرنے، questions ask کرنے، اور robot کے ساتھ conversation کرنے کے لیے۔

## OpenAI Whisper for Robot Voice Commands

OpenAI Whisper state-of-the-art speech recognition model ہے جو spoken language کو text میں convert کر سکتا ہے۔ یہ robotics applications کے لیے particularly well-suited ہے:

```python
class WhisperRobotInterface:
    """
    Whisper-based voice command interface humanoid robots کے لیے۔
    """

    def __init__(self, model_size="medium"):
        # Whisper model load کریں
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

    def process_audio_stream(self):
        """
        Voice commands کے لیے continuously audio stream process کریں۔
        """
        audio_queue = queue.Queue()

        def audio_callback(indata, frames, time, status):
            audio_queue.put(indata.copy())

        # Audio stream start کریں
        with sd.InputStream(
            callback=audio_callback,
            channels=1,
            samplerate=16000,
            blocksize=1024
        ):
            while True:
                audio_data = audio_queue.get()

                # Wake word check کریں
                if self.wake_word_detector.detect(audio_data):
                    self.start_listening()

                # Voice commands process کریں if listening
                if self.is_listening:
                    if self.vad_detector.is_speech(audio_data):
                        command_audio = self.record_command()
                        transcription = self.transcribe_audio(command_audio)
                        robot_command = self.command_parser.parse(transcription)
                        self.execute_command(robot_command)
                        self.stop_listening()
```

## Intent Recognition اور Command Parsing

```python
class RobotCommandParser:
    """
    Voice commands کو robot actions میں parse کریں۔
    """

    def __init__(self):
        # Command intents define کریں
        self.intents = {
            "navigation": {
                "patterns": [
                    r"go to (?P<location>\w+)",
                    r"move to (?P<location>\w+)",
                    r"walk to (?P<location>\w+)"
                ]
            },
            "manipulation": {
                "patterns": [
                    r"pick up (?P<object>\w+)",
                    r"grasp (?P<object>\w+)",
                    r"take (?P<object>\w+)"
                ]
            },
            "information": {
                "patterns": [
                    r"what is (?P<query>.+)",
                    r"tell me about (?P<query>.+)"
                ]
            }
        }

    def parse(self, transcription):
        """
        Voice transcription کو robot command میں parse کریں۔
        """
        transcription = transcription.lower().strip()

        # Intents match کریں
        for intent_name, intent_config in self.intents.items():
            for pattern in intent_config["patterns"]:
                match = re.search(pattern, transcription)
                if match:
                    entities = match.groupdict()
                    command = {
                        "intent": intent_name,
                        "entities": entities,
                        "original_text": transcription
                    }
                    return command

        return {
            "intent": "information",
            "entities": {"query": transcription},
            "original_text": transcription
        }
```

## Voice Activity Detection (VAD)

```python
class VoiceActivityDetector:
    """
    Audio streams میں voice activity detect کریں۔
    """

    def __init__(self, threshold=0.3, silence_duration=1.0):
        self.threshold = threshold
        self.silence_duration = silence_duration
        self.energy_history = []
        self.min_energy_samples = 100

    def is_speech(self, audio_data):
        """
        Detect کریں کہ audio میں speech ہے یا نہیں۔
        """
        energy = self.calculate_energy(audio_data)

        self.energy_history.append(energy)
        if len(self.energy_history) > self.min_energy_samples:
            self.energy_history.pop(0)

        if energy > self.threshold:
            return True

        recent_energies = self.energy_history[-int(self.silence_duration * 100):]
        if any(e > self.threshold for e in recent_energies):
            return True

        return False

    def calculate_energy(self, audio_data):
        """
        Audio signal کا energy calculate کریں۔
        """
        rms = np.sqrt(np.mean(audio_data ** 2))
        return rms
```

## Wake Word Detection

```python
class WakeWordDetector:
    """
    Voice command system activate کرنے کے لیے wake words detect کریں۔
    """

    def __init__(self, wake_words, detection_threshold=0.8):
        self.wake_words = [word.lower() for word in wake_words]
        self.detection_threshold = detection_threshold

        self.keyword_model = self.load_keyword_model()

    def detect(self, audio_data):
        """
        Audio data میں wake words detect کریں۔
        """
        transcription = self.transcribe_short_segment(audio_data)

        transcription_lower = transcription.lower()

        for wake_word in self.wake_words:
            if wake_word in transcription_lower:
                return True

        return False
```

## Privacy اور Security Considerations

```python
class SecureVoiceProcessor:
    """
    Privacy protections کے ساتھ secure voice processing۔
    """

    def __init__(self):
        # Audio data کے لیے encryption
        self.audio_encryptor = AudioEncryptor()

        # Privacy filters
        self.privacy_filter = PrivacyFilter()

        # جہاں possible ہو وہاں local processing
        self.local_models = self.load_local_models()

    def process_privately(self, audio_data):
        """
        Privacy protections کے ساتھ audio data process کریں۔
        """
        # Processing سے پہلے audio encrypt کریں
        encrypted_audio = self.audio_encryptor.encrypt(audio_data)

        # Sensitive information filter کریں
        filtered_audio = self.privacy_filter.remove_sensitive_content(encrypted_audio)

        # جہاں possible ہو وہاں locally process کریں
        if self.can_process_locally(filtered_audio):
            result = self.process_locally(filtered_audio)
        else:
            result = self.process_securely(filtered_audio)

        return result
```

## Learning Outcomes

اس section کو مکمل کرنے کے بعد، students یہ کر سکیں گے:

1. OpenAI Whisper کو speech-to-text conversion کے لیے integrate کریں
2. Activation کے لیے wake word detection implement کریں
3. Voice activity detection systems design کریں
4. Voice commands کo robot actions m parse کریں
5. Real-time performance کے لیے voice processing optimize کریں
6. Voice systems میں privacy اور security ensure کریں
7. Voice command accuracy test اور evaluate کریں

## اگلے steps

[Large Language Model Task Planning](./llm-task-planning.md) پڑھیں تاکہ VLA systems میں high-level task planning کے لیے LLMs کو use کرنا سیکھ سکیں۔
