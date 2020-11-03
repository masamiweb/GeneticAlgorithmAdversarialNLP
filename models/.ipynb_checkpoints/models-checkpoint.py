import tensorflow as tf

from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense, Input, Dropout
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau

# for more info see:
# https://keras.io/examples/nlp/pretrained_word_embeddings/

class TrainModel(object):
    def __init__(self,
                 embeddings_matrix,
                 train_x,
                 train_y,
                 test_x,
                 test_y,
                 epochs=10,
                 batch_size=64,
                 input_dim=25000,  # i.e maximum vocabulary size
                 input_length=256,  # i.e. length of sequence after adding padding
                 trainable=True,  # pre-trained embeddings, so set to False to prevents weights update during training
                 output_dim=100,  # i.e. embeddings dimension
                 learning_rate=1e-3  # 0.001 for default learning rate
                 ):
        self.embeddings_matrix = embeddings_matrix
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.epochs = epochs
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.input_length = input_length
        self.trainable = trainable
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        self.model, self.history_dict = self.build_model()

    def build_model(self):
        print("Building model, please wait ...\n")
        # First create the embeddings layer
        embedding_layer = self.create_embeddings_layer()
        model_definition = self.define_model(embedding_layer)
        return self.create_model(model_definition)

    def create_embeddings_layer(self):

        return tf.keras.layers.Embedding(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            weights=[self.embeddings_matrix],
            input_length=self.input_length,
            trainable=self.trainable
        )

    def define_model(self, embedding_layer):
        sequence_input = Input(shape=(self.input_length,), dtype='int32')
        embedding_sequences = embedding_layer(sequence_input)
        x = SpatialDropout1D(0.2)(embedding_sequences)
        x = Conv1D(64, 5, activation='tanh')(x)
        x = Bidirectional(LSTM(64, dropout=0.2))(x)
        x = Dense(512, activation='tanh')(x)
        x = Dropout(0.5)(x)  # set dropout rate to 0.5 to prevent over-fitting
        x = Dense(512, activation='tanh')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        return tf.keras.Model(sequence_input, outputs)

    def create_model(self, model_definition):
        model_definition.compile(optimizer=Adam(learning_rate=self.learning_rate),
                                 loss='binary_crossentropy',
                                 metrics=['accuracy'])

        # stop early if validation loss plateaus
        # patience = number of epochs after which we stop if no further improvement
        # mode = i.e. if quantity being monitored stops decreasing then stop training (currently monitoring val_loss)
        # early_stopping = EarlyStopping(monitor='val_loss', patience=7, verbose=0, min_delta=1e-4, mode='min')

        # save model at defined checkpoints so we can continue training from saved state
        # mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')

        # adjust learning rate if validation loss plateaus
        reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, mode='min')
        # make sure we find a GPU, or training will take a very long time!
        print("Training on GPU...") if tf.config.list_physical_devices('GPU') \
            else print("Training on CPU...")

        # Save the training history to a dictionary called history
        history = model_definition.fit(self.train_x, self.train_y,
                                       batch_size=self.batch_size,
                                       epochs=self.epochs,
                                       verbose=1,
                                       validation_data=(self.test_x, self.test_y),
                                       callbacks=[reduce_lr_on_plateau])
        print("\nFinished building model!")
        return model_definition, history

    def get_model(self):
        return self.model

    def get_history(self):
        return self.history_dict
    
    # Add save methods for model and dictionary
    # add predict method to test model
