import pyttsx3

try:
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.setProperty("volume", 1.0)

    print("Speaking test message...")
    engine.say("Prasanthi, please be alert. Please do not get distracted.")
    engine.runAndWait()
    print("Voice test finished.")

except Exception as error:
    print("VOICE ERROR:", repr(error))