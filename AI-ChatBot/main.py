# libraries
import random
import numpy as np
import pickle
import json
from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok  # Uncomment if you need ngrok
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load model and data
model = load_model("chatbot_model1.keras")
with open("/Users/krishnapeddibhotla/Downloads/intents.json") as data_file:
    intents = json.load(data_file)
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

app = Flask(__name__)


# run_with_ngrok(app)  # Uncomment if you need ngrok

@app.route("/")
def home():
    return render_template("/Users/krishnapeddibhotla/IdeaProjects/Index/index.html")  # Ensure this file is in the templates directory


@app.route("/get", methods=["POST"])
def chatbot_response():
    msg = request.form["msg"]

    if msg.startswith('my name is'):
        name = msg[11:]
        response = process_message(msg, name)
    elif msg.startswith('hi my name is'):
        name = msg[14:]
        response = process_message(msg, name)
    else:
        response = process_message(msg)

    return response


def process_message(msg, name=None):
    ints = predict_class(msg)
    response = getResponse(ints)
    if name:
        response = response.replace("{n}", name)
    return response


# Chat functionalities
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def predict_class(sentence):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    return return_list


def getResponse(ints):
    if not ints:
        return "Sorry, I didn't understand that."
    tag = ints[0]["intent"]
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Sorry, I didn't find a response."


if __name__ == "__main__":
    app.run(port=5001, debug=True)  # Change port to 5001 or any other available port

if __name__ == "__main__":
    app.run(debug=True)  # Set debug=True for development