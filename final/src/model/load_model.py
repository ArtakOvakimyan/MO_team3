from transformers import pipeline

def get_classifier():
    return pipeline("text-classification", model="unitary/toxic-bert")