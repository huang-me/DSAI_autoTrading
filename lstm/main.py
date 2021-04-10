from preprocessing import get_data, shuffle_train, build_train
from lstm import createTrain

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # load train datas
    train_data = get_data('../data/training.csv')
    x_train, y_train = build_train(train_data)
    x_train, y_train = shuffle_train(x_train, y_train)
    # split validate
    x_train = x_train[:-1]
    x_validate = x_train[-1:]
    y_train = y_train[:-1]
    y_validate = y_train[-1:]
    # train model
    model = createTrain(x_train, y_train)
    # load predict data
    # predict_data = get_data('../data/testing.csv')
    # x_predict = np.array(predict_data)[np.newaxis, :, :]
    # test = model.predict(x_predict)
    # print(test)
    # test validate
    val = model.predict(x_validate)
    plt.figure()
    x = np.arange(20)
    plt.plot(x, val[0], 'b', x, y_validate[0], 'r')
    plt.legend(labels=['predict', 'ground truth'])
    plt.savefig('compare.jpg')
    print('done')
