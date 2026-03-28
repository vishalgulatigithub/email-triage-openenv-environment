import json

def load_tasks():
    with open("data/emails.json") as f:
        return json.load(f)