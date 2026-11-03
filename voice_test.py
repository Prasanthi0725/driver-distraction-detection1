
import speech_recognition as sr

recognizer = sr.Recognizer()
recognizer.dynamic_energy_threshold = True
recognizer.pause_threshold = 0.8

print("=== Driver Voice Command Test ===")
print("Speak clearly in a quiet room.")
print("Say: Start monitoring")
print("")

try:
    with sr.Microphone() as source:
        print("Calibrating microphone...")
        recognizer.adjust_for_ambient_noise(source, duration=2)

        print("Energy threshold:", recognizer.energy_threshold)
        print("Speak now...")
        audio = recognizer.listen(
            source,
            timeout=10,
            phrase_time_limit=5
        )

    print("Processing audio...")

    for language in ["en-IN", "te-IN", "hi-IN"]:
        try:
            result = recognizer.recognize_google(
                audio,
                language=language
            )
            print(f"Recognized ({language}): {result}")
            break
        except sr.UnknownValueError:
            print(f"Could not understand using {language}.")
        except sr.RequestError as error:
            print("Recognition service error:", error)
            break

except sr.WaitTimeoutError:
    print("No speech detected. Check the microphone.")

except OSError as error:
    print("Microphone error:", error)
