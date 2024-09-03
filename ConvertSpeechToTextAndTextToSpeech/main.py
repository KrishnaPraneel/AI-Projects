import speech_recognition as sr
import pyttsx3

# Initialize the recognizer
r = sr.Recognizer()


# Function to convert text to speech
def SpeakText(command):
    # Initialize the engine
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()


# Loop infinitely for user to speak
while True:
    try:
        # Use the microphone as source for input
        with sr.Microphone() as source:
            # Wait for a second to let the recognizer adjust the energy threshold based on the surrounding noise level
            r.adjust_for_ambient_noise(source, duration=0.2)

            # Listens for the user's input
            print("Listening...")
            audio = r.listen(source)

            # Using Google to recognize audio
            MyText = r.recognize_google(audio)
            MyText = MyText.lower()

            print(f"Did you say: {MyText}")
            SpeakText(MyText)

    except sr.RequestError as e:
        print(f"Could not request results; {e}")

    except sr.UnknownValueError:
        print("Unknown error occurred")