import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_data(train_fname="../data/training.csv"):
    '''
    read data into dataframe
    '''
    header = ['open', 'high', 'low', 'close']
    train_data = pd.read_csv(train_fname, names=header)
    '''
    show data
    '''
    # _, ax1 = plt.subplots(2, 2)
    # train_data.plot(x=None, y='close', ax=ax1[0, 0])
    # train_data.plot(x=None, y='open', ax=ax1[0, 1])
    # train_data.plot(x=None, y='high', ax=ax1[1, 0])
    # train_data.plot(x=None, y='low', ax=ax1[1, 1])
    # plt.show()
    return train_data

def shuffle_train(x, y):
    np.random.seed(10)
    randomlist = np.arange(x.shape[0])
    np.random.shuffle(randomlist)
    return x[randomlist], y[randomlist]

def build_train(data, past=20, future=20):
    X_train, Y_train = [], []
    for i in range(data.shape[0]-past-future):
        X_train.append( np.array(data.iloc[i:i+past]) )
        Y_train.append( np.array(data.iloc[i+past:i+past+future]['open']) )
    return np.array(X_train), np.array(Y_train)

if __name__ == "__main__":
    data = get_data()
    x, y = build_train(data)
    print(x.shape, y.shape)