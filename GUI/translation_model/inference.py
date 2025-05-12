from .vectorizer import english_vectorizer, arabic_vectorizer
import numpy as np
from .config import CONFIG

target_vocab = arabic_vectorizer.get_vocabulary()
target_index_lookup = dict(zip(range(len(target_vocab)), target_vocab))

def decode_sequence(input_sentence,model):
    tokenized_input_sentence = english_vectorizer([input_sentence])
    decoded_sentence = "[start]"
    for i in range(CONFIG["SEQUENCE_LENGTH"]):
        tokenized_target_sentence = arabic_vectorizer(
            [decoded_sentence])[:, :-1]
        predictions = model(
            [tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = target_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == "[end]":
            break
    return decoded_sentence.replace("[start]", "").replace("end", "").strip()