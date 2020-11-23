import keras
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense, Input, Dropout, Embedding, SpatialDropout1D


def create_model(vectorizer, embedding_matrix, vocab, dimension=300, lrate=1e4):
    
    num_tokens = len(vocab) + 1
    embedding_dim = dimension
    hits = 0
    misses = 0
    
    embedding_layer = Embedding(
    num_tokens,
    embedding_dim,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    trainable=False,
    mask_zero=True
    )
    
    
    # activation changed from tanh to relu
    # make sure to add the shape parameter, otherwise the loss and val_loss wont be calculated and will show as NaN
    inputs = keras.Input(shape=(1,), dtype=tf.string)
    x = vectorizer(inputs)
    embedding_sequences = embedding_layer(x)

    x = SpatialDropout1D(0.2)(embedding_sequences)
    x = Conv1D(64, 5, activation='relu')(x)
    x = Bidirectional(LSTM(128, dropout=0.2))(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)  # set dropout rate to 0.5 to prevent over-fitting
    x = Dense(512, activation='relu')(x)

    predictions = Dense(1, activation='sigmoid')(x)
    model =  keras.Model(inputs, predictions)

    # default learning rate is 0.001 for Adam optimizer
    model.compile(
        loss="binary_crossentropy", optimizer=Adam(lrate), metrics=["acc"]
    )
    return model
