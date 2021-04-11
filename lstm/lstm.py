from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import os

from preprocessing import get_data, shuffle_train, build_train

def _init_gpu():
	gpus = tf.config.list_physical_devices('GPU')
	if gpus:
		try:
			# Currently, memory growth needs to be the same across GPUs
			for gpu in gpus:
				tf.config.experimental.set_memory_growth(gpu, True)
				logical_gpus = tf.config.experimental.list_logical_devices('GPU')
				print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
		except RuntimeError as e:
			# Memory growth must be set before GPUs have been initialized
			print(e)
	return

def createModel(shape):
	model = Sequential()
	model.add(LSTM(32, input_shape=(shape[1], shape[2]), return_sequences=True))
	model.add(Dropout(0.2))
	# model.add(LSTM(128, return_sequences=True))
	# model.add(Dropout(0.2))
	# model.add(LSTM(64, return_sequences=True))
	# model.add(Dropout(0.2))
	model.add(LSTM(20))
	model.add(Dense(20, activation='linear'))
	model.compile(loss="mse", optimizer="adam")
	return model

def createTrain(x, y):
    _init_gpu()
    if(os.path.exists('model/lstm')):
        model = load_model('model/lstm')
    else:
        model = createModel(x.shape)
    # callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
    # model.fit(x, y, epochs=2000, batch_size=128, callbacks=[callback], validation_split=0.5)
    model.fit(x, y, epochs=100, batch_size=128, validation_split=0.5)
    model.save('model/lstm')
    model.summary()
    return model

if __name__ == "__main__":
    train_data = get_data()
    x_train, y_train = build_train(train_data)
    x_train, y_train = shuffle_train(x_train, y_train)
    model = createTrain(x_train, y_train)