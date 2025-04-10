import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r', encoding='utf-8') as json_data:

    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Fawaaz"
print("👋 Hi! I’m Fawaaz, your fun and personal Cape Town tour guide 🌍 (type 'quit' to exit)")
while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ").lower()

    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        fallback_responses = [
        "Eish, that’s out of my comfort zone bru 😅 Try asking me about the lovely Mother City 🌍",
        "Hmm, that’s not really my thing hey. I’m more of a Cape Town kinda guy 🇿🇦",
        "Sho! I only specialize in the Mother City – ask me about food, fun, and places to vibe! 😎",
        "Yoh! That one went over my head. Let’s stick to Cape Town stuff, yeah? 🏖️",
        "I don’t know about that one, but I know Cape Town like the back of my hand 😄"
    ]
        print(f"{bot_name}: {random.choice(fallback_responses)}")
