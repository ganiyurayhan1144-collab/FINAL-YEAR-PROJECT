
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Embedding,
    Bidirectional, LSTM, Concatenate, GlobalMaxPooling1D
)
from tensorflow.keras.models import Model

def build_model(sequence_length=200, vocab_size=128):

    # URL character input
    url_input = Input(shape=(sequence_length,), name="url_input")

    # BiLSTM branch
    x = Embedding(vocab_size, 32)(url_input)
    x = Bidirectional(LSTM(32, return_sequences=True))(x)
    x = GlobalMaxPooling1D()(x)

    # ANN numerical feature input
    feature_input = Input(shape=(14,), name="feature_input")

    y = Dense(64, activation="relu")(feature_input)
    y = Dropout(0.3)(y)
    y = Dense(32, activation="relu")(y)

    # Combine both branches
    combined = Concatenate()([x, y])

    output = Dense(1, activation="sigmoid")(combined)

    model = Model(
        inputs=[url_input, feature_input],
        outputs=output
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model
