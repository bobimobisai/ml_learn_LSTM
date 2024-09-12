import numpy as np
import tensorflow as tf
import pickle
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import (
    Input,
    Embedding,
    LSTM,
    Dense,
    Concatenate,
    Dot,
    Activation,
)

with open("config/data.json", "r") as f:
    data = json.load(f)

with open("config/model_settings.json", "r") as f:
    model_conf = json.load(f)

_embedding_dim = model_conf["embedding_dim"]
_epochs = model_conf["epochs"]

def preprocess_text(texts):
    word_to_index = {}
    index = 1
    for category, subcategories in data.items():
        for subcategory, items in subcategories.items():
            for item in items:
                text = item["input"] + " " + item["output"]
                for word in text.lower().split():
                    if word not in word_to_index:
                        word_to_index[word] = index
                        index += 1
    return word_to_index


def texts_to_sequences(texts, word_to_index):
    sequences = []
    for text in texts:
        sequence = [word_to_index.get(word, 0) for word in text.lower().split()]
        sequences.append(sequence)
    return sequences


word_to_index = preprocess_text(
    [
        item["input"]
        for category in data.values()
        for subcategory in category.values()
        for item in subcategory
    ]
    + [
        item["output"]
        for category in data.values()
        for subcategory in category.values()
        for item in subcategory
    ]
)


def create_seq2seq_with_attention_model(vocab_size, embedding_dim, max_length):
    encoder_input = Input(shape=(max_length,))
    encoder_embedding = Embedding(vocab_size, embedding_dim)(encoder_input)
    encoder_lstm = LSTM(128, return_sequences=True, return_state=True)
    encoder_output, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]

    decoder_input = Input(shape=(max_length,))
    decoder_embedding = Embedding(vocab_size, embedding_dim)(decoder_input)
    decoder_lstm = LSTM(128, return_sequences=True)
    decoder_output = decoder_lstm(decoder_embedding, initial_state=encoder_states)

    attention = Dot(axes=[2, 2])([decoder_output, encoder_output])
    attention = Activation("softmax")(attention)
    context_vector = Dot(axes=[2, 1])([attention, encoder_output])
    decoder_combined_context = Concatenate(axis=-1)([context_vector, decoder_output])

    decoder_dense = Dense(vocab_size, activation="softmax")
    output = decoder_dense(decoder_combined_context)

    model = tf.keras.Model([encoder_input, decoder_input], output)
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def prepare_data():
    questions = []
    answers = []
    for category, subcategories in data.items():
        for subcategory, items in subcategories.items():
            for item in items:
                questions.append(item["input"])
                answers.append(item["output"])

    questions_seq = texts_to_sequences(questions, word_to_index)
    answers_seq = texts_to_sequences(answers, word_to_index)

    X = pad_sequences(questions_seq, maxlen=max_length, padding="post")
    y = pad_sequences(answers_seq, maxlen=max_length, padding="post")
    y = np.array([to_categorical(seq, num_classes=vocab_size) for seq in y])

    decoder_input = np.zeros_like(X)
    decoder_input[:, 1:] = X[:, :-1]

    return X, y, decoder_input


vocab_size = len(word_to_index) + 1
embedding_dim = _embedding_dim
max_length = max(
    max(
        len(seq)
        for seq in texts_to_sequences(
            [
                item["input"]
                for category in data.values()
                for subcategory in category.values()
                for item in subcategory
            ],
            word_to_index,
        )
    ),
    max(
        len(seq)
        for seq in texts_to_sequences(
            [
                item["output"]
                for category in data.values()
                for subcategory in category.values()
                for item in subcategory
            ],
            word_to_index,
        )
    ),
)

X, y, decoder_input = prepare_data()

model = create_seq2seq_with_attention_model(vocab_size, embedding_dim, max_length)

model.fit([X, decoder_input], y, epochs=_epochs)

model.save("model_data/seq2seq_model.h5")

with open("model_data/word_to_index.pkl", "wb") as f:
    pickle.dump(word_to_index, f)

with open("model_data/max_length.pkl", "wb") as f:
    pickle.dump(max_length, f)
