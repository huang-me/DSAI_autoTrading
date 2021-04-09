from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import os

from preprocessing import get_data

def createModel(shape):
    model = Sequential()
    model.add(LSTM(4, input_length=shape[1], input_dim=shape[2], return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss="mse", optimizer="adam")
    return model

def createTrain(x, y):
    if(os.path.exists('model/lstm')):
        model = load_model('model/lstm')
    else:
        model = createModel(x.shape)
    model.summary()
    callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
    model.fit(x, y, epochs=500, batch_size=64, callbacks=[callback], validation_split=0.3)
    model.save('model/lstm')
    return model

if __name__ == "__main__":
    x_train, y_train = get_data()
    model = createTrain(x_train, y_train)