import json
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

with open("intents.json", "r") as file:
    data = json.load(file)

patterns = []
tags = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(intent["tag"])

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(patterns)

model = LogisticRegression(max_iter=1000)
model.fit(X, tags)

print("===================================")
print(" AI Chatbot Started ")
print(" Type 'quit' to exit ")
print("===================================")

while True:
    user_input = input("\nYou: ")

    if user_input.lower() == "quit":
        print("Bot: Goodbye!")
        break

    X_test = vectorizer.transform([user_input])
    predicted_tag = model.predict(X_test)[0]

    found = False

    for intent in data["intents"]:
        if intent["tag"] == predicted_tag:
            print("Bot:", random.choice(intent["responses"]))
            found = True
            break

    if not found:
        print("Bot: Sorry, I don't understand that question.")