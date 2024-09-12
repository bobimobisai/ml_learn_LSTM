import threading
import numpy as np
import tensorflow as tf
import pickle
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import logging

logging.basicConfig(level=logging.INFO)

model = tf.keras.models.load_model("model_data/seq2seq_model.h5")


with open("model_data/word_to_index.pkl", "rb") as f:
    word_to_index = pickle.load(f)

with open("model_data/max_length.pkl", "rb") as f:
    max_length = pickle.load(f)

index_to_word = {index: word for word, index in word_to_index.items()}


def preprocess_text(texts, word_to_index):
    sequences = []
    for text in texts:
        sequence = [word_to_index.get(word, 0) for word in text.lower().split()]
        sequences.append(sequence)
    return sequences


def generate_response(input_text, model, word_to_index, index_to_word, max_length):
    input_seq = preprocess_text([input_text], word_to_index)
    input_seq = pad_sequences(input_seq, maxlen=max_length, padding="post")

    decoder_input = np.zeros((1, max_length))
    decoder_input[0, 1:] = input_seq[0, :-1]

    prediction = model.predict([input_seq, decoder_input])[0]

    response = []
    probabilities = []
    for t in range(prediction.shape[0]):
        pred_word_index = np.argmax(prediction[t, :])
        prob = np.max(prediction[t, :])
        word = index_to_word.get(pred_word_index, "")
        if pred_word_index != 0:
            response.append(word)
            probabilities.append(prob)
        if pred_word_index == word_to_index.get("<end>", 0):
            break

    response_text = " ".join(response).strip()
    confidence = np.mean(probabilities) if probabilities else 0.0

    return response_text, confidence


def _generate_response(queue, output_queue):
    while True:
        input_text = queue.get()
        if input_text:
            response, confidence = generate_response(
                input_text, model, word_to_index, index_to_word, max_length
            )
            output_queue.put((response, confidence))


def start_update_thread(queue, output_queue):
    update_thread = threading.Thread(
        target=_generate_response, args=(queue, output_queue)
    )
    update_thread.start()
