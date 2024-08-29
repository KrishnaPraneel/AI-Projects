import random
import json
import pickle

import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

import numpy as np
lemmatizer = WordNetLemmatizer()

intents = json.loads(open("/Users/krishnapeddibhotla/Downloads/intents2.json").read())

words = []
classes = []
documents = []

ignore_letters = ["?", "!", ".", ","]

for intent in intents["intents"]:
	for pattern in intent["patterns"]:
		word_list = nltk.word_tokenize(pattern)
		words.extend(word_list)
		documents.append((word_list, intent["tag"]))

		if intent["tag"] not in classes:
			classes.append(intent["tag"])
words = [lemmatizer.lemmatize(word)
		for word in words if word not in ignore_letters]

words = sorted(set(words))
classes = sorted(set(classes))
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))
dataset = []
template = [0]*len(classes)

for document in documents:
	bag = []
	word_patterns = document[0]
	word_patterns = [lemmatizer.lemmatize(
		word.lower()) for word in word_patterns]

	for word in words:
		bag.append(1) if word in word_patterns else bag.append(0)

	output_row = list(template)
	output_row[classes.index(document[1])] = 1
	dataset.append([bag, output_row])

random.shuffle(dataset)
dataset = np.array(dataset)

train_x = list(dataset[:, 0])
train_y = list(dataset[:, 1])
model = Sequential()
model.add(Dense(256, input_shape=(len(train_x[0]),),
				activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))


sgd = SGD(learning_rate=0.01, decay=1e-6,
		momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
			optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y),
				epochs=200, batch_size=5, verbose=1)

model.save("chatbot_model.keras", hist)
print("Done!")
import random
import json
import pickle

import nltk
nltk.download('all')
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

import numpy as np

lemmatizer = WordNetLemmatizer()

intents = json.loads(open("/Users/krishnapeddibhotla/Downloads/intents2.json").read())

words = []
classes = []
documents = []

ignore_letters = ["?", "!", ".", ","]

for intent in intents["intents"]:
	for pattern in intent["patterns"]:
		word_list = nltk.word_tokenize(pattern)
		words.extend(word_list)
		documents.append((word_list, intent["tag"]))

		if intent["tag"] not in classes:
			classes.append(intent["tag"])
words = [lemmatizer.lemmatize(word)
		for word in words if word not in ignore_letters]

words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

dataset = []
template = [0]*len(classes)

for document in documents:
	bag = []
	word_patterns = document[0]
	word_patterns = [lemmatizer.lemmatize(word.lower())
					for word in word_patterns]

	for word in words:
		bag.append(1) if word in word_patterns else bag.append(0)

	output_row = list(template)
	output_row[classes.index(document[1])] = 1
	dataset.append([bag, output_row])

random.shuffle(dataset)
dataset = np.array(dataset)

train_x = list(dataset[:, 0])
train_y = list(dataset[:, 1])

model = Sequential()
model.add(Dense(256, input_shape=(len(train_x[0]),),
				activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))


sgd = SGD(learning_rate=0.01,
		momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
			optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y),
				epochs=200, batch_size=5, verbose=1)

model.save("chatbot_model.keras", hist)
print("Done!")
# This function will take the voice input
# converted into string as input and predict
# and return the result in both
# text as well as voice format

def calling_the_bot(txt):
	global res
	predict = predict_class(txt)
	res = get_response(predict, intents)

	engine.say("Found it. From our Database we found that" + res)
	# engine.say(res)
	engine.runAndWait()
	print("Your Symptom was : ", text)
	print("Result found in our Database : ", res)
if __name__ == '__main__':
	print("Bot is Running")

	recognizer = sr.Recognizer()
	mic = sr.Microphone()

	engine = pyttsx3.init()
	rate = engine.getProperty('rate')

	# Increase the rate of the bot according to need,
	# Faster the rate, faster it will speak, vice versa for slower.

	engine.setProperty('rate', 175)

	# Increase or decrease the bot's volume
	volume = engine.getProperty('volume')
	engine.setProperty('volume', 1.0)

	voices = engine.getProperty('voices')

	engine.say(
		"Hello user, I am Bagley, your personal Talking Healthcare Chatbot.")
	engine.runAndWait()

	engine.say(
		"IF YOU WANT TO CONTINUE WITH MALE VOICE PLEASE SAY MALE.\
		OTHERWISE SAY FEMALE.")
	engine.runAndWait()

	# Asking for the MALE or FEMALE voice.
	with mic as source:
		recognizer.adjust_for_ambient_noise(source, duration=0.2)
		audio = recognizer.listen(source)

	audio = recognizer.recognize_google(audio)

	# If the user says Female then the bot will speak in female voice.
	if audio == "Female".lower():
		engine.setProperty('voice', voices[1].id)
		print("You have chosen to continue with Female Voice")
	else:
		engine.setProperty('voice', voices[0].id)
		print("You have chosen to continue with Male Voice")

	while True or final.lower() == 'True':
		with mic as symptom:
			print("Say Your Symptoms. The Bot is Listening")
			engine.say("You may tell me your symptoms now. I am listening")
			engine.runAndWait()
			try:
				recognizer.adjust_for_ambient_noise(symptom, duration=0.2)
				symp = recognizer.listen(symptom)
				text = recognizer.recognize_google(symp)
				engine.say("You said {}".format(text))
				engine.runAndWait()

				engine.say(
					"Scanning our database for your symptom. Please wait.")
				engine.runAndWait()

				time.sleep(1)

				# Calling the function by passing the voice
				# inputted symptoms converted into string
				calling_the_bot(text)
			except sr.UnknownValueError:
				engine.say(
					"Sorry, Either your symptom is unclear to me \
					or it is not present in our database. Please Try Again.")
				engine.runAndWait()
				print(
					"Sorry, Either your symptom is unclear to me\
					or it is not present in our database. Please Try Again.")
			finally:
				engine.say(
					"If you want to continue please say True otherwise\
					say False.")
				engine.runAndWait()

		with mic as ans:
			recognizer.adjust_for_ambient_noise(ans, duration=0.2)
			voice = recognizer.listen(ans)
			final = recognizer.recognize_google(voice)

		if final.lower() == 'no' or final.lower() == 'please exit':
			engine.say("Thank You. Shutting Down now.")
			engine.runAndWait()
			print("Bot has been stopped by the user")
			exit(0)
import random
import json
import pickle

import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

import numpy as np
import speech_recognition as sr
import pyttsx3
import time

lemmatizer = WordNetLemmatizer()
intents = json.loads(open("/Users/krishnapeddibhotla/Downloads/intents2.json").read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')


def clean_up_sentence(sentence):
	sentence_words = nltk.word_tokenize(sentence)
	sentence_words = [lemmatizer.lemmatize(word)
					for word in sentence_words]

	return sentence_words


def bag_of_words(sentence):
	sentence_words = clean_up_sentence(sentence)
	bag = [0] * len(words)

	for w in sentence_words:
		for i, word in enumerate(words):
			if word == w:
				bag[i] = 1
	return np.array(bag)


def predict_class(sentence):
	bow = bag_of_words(sentence)
	res = model.predict(np.array([bow]))[0]

	ERROR_THRESHOLD = 0.25

	results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

	results.sort(key=lambda x: x[1], reverse=True)

	return_list = []

	for r in results:
		return_list.append({'intent': classes[r[0]],
							'probability': str(r[1])})
	return return_list


def get_response(intents_list, intents_json):
	tag = intents_list[0]['intent']
	list_of_intents = intents_json['intents']

	result = ''

	for i in list_of_intents:
		if i['tag'] == tag:
			result = random.choice(i['responses'])
			break
	return result


# This function will take the voice input converted
# into string as input and predict and return the result in both
# text as well as voice format.
def calling_the_bot(txt):
	global res
	predict = predict_class(txt)
	res = get_response(predict, intents)

	engine.say("Found it. From our Database we found that" + res)
	# engine.say(res)
	engine.runAndWait()
	print("Your Symptom was : ", text)
	print("Result found in our Database : ", res)


if __name__ == '__main__':
	print("Bot is Running")

	recognizer = sr.Recognizer()
	mic = sr.Microphone()

	engine = pyttsx3.init()
	rate = engine.getProperty('rate')

	# Increase the rate of the bot according to need,
	# Faster the rate, faster it will speak, vice versa for slower.

	engine.setProperty('rate', 175)

	# Increase or decrease the bot's volume
	volume = engine.getProperty('volume')
	engine.setProperty('volume', 1.0)

	voices = engine.getProperty('voices')

	"""User Might Skip the following Part till the start of While Loop"""
	engine.say(
		"Hello user, I am Bagley, your personal Talking Healthcare Chatbot.")
	engine.runAndWait()

	engine.say(
		"IF YOU WANT TO CONTINUE WITH MALE VOICE PLEASE\
		SAY MALE. OTHERWISE SAY FEMALE.")
	engine.runAndWait()

	# Asking for the MALE or FEMALE voice.
	with mic as source:
		recognizer.adjust_for_ambient_noise(source, duration=0.2)
		audio = recognizer.listen(source)

	audio = recognizer.recognize_google(audio)

	# If the user says Female then the bot will speak in female voice.
	if audio == "Female".lower():
		engine.setProperty('voice', voices[1].id)
		print("You have chosen to continue with Female Voice")
	else:
		engine.setProperty('voice', voices[0].id)
		print("You have chosen to continue with Male Voice")

	"""User might skip till HERE"""

	while True or final.lower() == 'True':
		with mic as symptom:
			print("Say Your Symptoms. The Bot is Listening")
			engine.say("You may tell me your symptoms now. I am listening")
			engine.runAndWait()
			try:
				recognizer.adjust_for_ambient_noise(symptom, duration=0.2)
				symp = recognizer.listen(symptom)
				text = recognizer.recognize_google(symp)
				engine.say("You said {}".format(text))
				engine.runAndWait()

				engine.say(
					"Scanning our database for your symptom. Please wait.")
				engine.runAndWait()

				time.sleep(1)

				# Calling the function by passing the voice inputted
				# symptoms converted into string
				calling_the_bot(text)
			except sr.UnknownValueError:
				engine.say(
					"Sorry, Either your symptom is unclear to me or it is\
					not present in our database. Please Try Again.")
				engine.runAndWait()
				print(
					"Sorry, Either your symptom is unclear to me or it is\
					not present in our database. Please Try Again.")
			finally:
				engine.say(
					"If you want to continue please say True otherwise say\
					False.")
				engine.runAndWait()

		with mic as ans:
			recognizer.adjust_for_ambient_noise(ans, duration=0.2)
			voice = recognizer.listen(ans)
			final = recognizer.recognize_google(voice)

		if final.lower() == 'no' or final.lower() == 'please exit':
			engine.say("Thank You. Shutting Down now.")
			engine.runAndWait()
			print("Bot has been stopped by the user")
			exit(0)
import random
import json
import pickle

import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

import numpy as np
import speech_recognition as sr
import pyttsx3
import time

lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json").read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')


def clean_up_sentence(sentence):
	sentence_words = nltk.word_tokenize(sentence)
	sentence_words = [lemmatizer.lemmatize(word)
					for word in sentence_words]

	return sentence_words


def bag_of_words(sentence):
	sentence_words = clean_up_sentence(sentence)
	bag = [0] * len(words)

	for w in sentence_words:
		for i, word in enumerate(words):
			if word == w:
				bag[i] = 1
	return np.array(bag)


def predict_class(sentence):
	bow = bag_of_words(sentence)
	res = model.predict(np.array([bow]))[0]

	ERROR_THRESHOLD = 0.25

	results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

	results.sort(key=lambda x: x[1], reverse=True)

	return_list = []

	for r in results:
		return_list.append({'intent': classes[r[0]],
							'probability': str(r[1])})
	return return_list


def get_response(intents_list, intents_json):
	tag = intents_list[0]['intent']
	list_of_intents = intents_json['intents']

	result = ''

	for i in list_of_intents:
		if i['tag'] == tag:
			result = random.choice(i['responses'])
			break
	return result


# This function will take the voice input converted
# into string as input and predict and return the result in both
# text as well as voice format.
def calling_the_bot(txt):
	global res
	predict = predict_class(txt)
	res = get_response(predict, intents)

	engine.say("Found it. From our Database we found that" + res)
	# engine.say(res)
	engine.runAndWait()
	print("Your Symptom was : ", text)
	print("Result found in our Database : ", res)


if __name__ == '__main__':
	print("Bot is Running")

	recognizer = sr.Recognizer()
	mic = sr.Microphone()

	engine = pyttsx3.init()
	rate = engine.getProperty('rate')

	# Increase the rate of the bot according to need,
	# Faster the rate, faster it will speak, vice versa for slower.

	engine.setProperty('rate', 175)

	# Increase or decrease the bot's volume
	volume = engine.getProperty('volume')
	engine.setProperty('volume', 1.0)

	voices = engine.getProperty('voices')

	"""User Might Skip the following Part till the start of While Loop"""
	engine.say(
		"Hello user, I am Bagley, your personal Talking Healthcare Chatbot.")
	engine.runAndWait()

	engine.say(
		"IF YOU WANT TO CONTINUE WITH MALE VOICE PLEASE\
		SAY MALE. OTHERWISE SAY FEMALE.")
	engine.runAndWait()

	# Asking for the MALE or FEMALE voice.
	with mic as source:
		recognizer.adjust_for_ambient_noise(source, duration=0.2)
		audio = recognizer.listen(source)

	audio = recognizer.recognize_google(audio)

	# If the user says Female then the bot will speak in female voice.
	if audio == "Female".lower():
		engine.setProperty('voice', voices[1].id)
		print("You have chosen to continue with Female Voice")
	else:
		engine.setProperty('voice', voices[0].id)
		print("You have chosen to continue with Male Voice")

	"""User might skip till HERE"""

	while True or final.lower() == 'True':
		with mic as symptom:
			print("Say Your Symptoms. The Bot is Listening")
			engine.say("You may tell me your symptoms now. I am listening")
			engine.runAndWait()
			try:
				recognizer.adjust_for_ambient_noise(symptom, duration=0.2)
				symp = recognizer.listen(symptom)
				text = recognizer.recognize_google(symp)
				engine.say("You said {}".format(text))
				engine.runAndWait()

				engine.say(
					"Scanning our database for your symptom. Please wait.")
				engine.runAndWait()

				time.sleep(1)

				# Calling the function by passing the voice inputted
				# symptoms converted into string
				calling_the_bot(text)
			except sr.UnknownValueError:
				engine.say(
					"Sorry, Either your symptom is unclear to me or it is\
					not present in our database. Please Try Again.")
				engine.runAndWait()
				print(
					"Sorry, Either your symptom is unclear to me or it is\
					not present in our database. Please Try Again.")
			finally:
				engine.say(
					"If you want to continue please say True otherwise say\
					False.")
				engine.runAndWait()

		with mic as ans:
			recognizer.adjust_for_ambient_noise(ans, duration=0.2)
			voice = recognizer.listen(ans)
			final = recognizer.recognize_google(voice)

		if final.lower() == 'no' or final.lower() == 'please exit':
			engine.say("Thank You. Shutting Down now.")
			engine.runAndWait()
			print("Bot has been stopped by the user")
			exit(0)