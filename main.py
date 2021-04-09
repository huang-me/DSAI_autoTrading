from preprocessing import get_data, shuffle_train
from lstm import createTrain

import matplotlib.pyplot as plt

if __name__ == "__main__":
    x_train, y_train = get_data('data/training.csv')
    x_train, y_train = shuffle_train(x_train, y_train)
    model = createTrain(x_train, y_train)
    x_predict, y_predict = get_data('data/testing.csv')
    test = model.predict(x_predict)
    print(test[-1])
    plt.plot(test[-1])
    plt.show()