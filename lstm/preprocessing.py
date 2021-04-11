from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def scale(df):
    header = ['open', 'high', 'low', 'close']
    sc = MinMaxScaler((0, 1))
    transformed = sc.fit_transform( df )
    transformed = pd.DataFrame( transformed, columns=header )
    return transformed, sc

def shuffle_train(x, y):
    np.random.seed(10)
    randomlist = np.arange(x.shape[0])
    np.random.shuffle(randomlist)
    return x[randomlist], y[randomlist]

def build_train(data, past=20):
    X_train, Y_train = [], []
    for i in range(data.shape[0]-past-1):
        X_train.append( np.array(data.iloc[i:i+past]) )
        Y_train.append( np.array(data.iloc[i+past]['open']) )
    return np.array(X_train), np.array(Y_train)

def load_all(train='./data/training.csv', test='./data/testing.csv'):
    header = ['open', 'high', 'low', 'close']
    train_df = pd.read_csv(train, names=header)
    test_df = pd.read_csv(test, names=header)
    whole_df = train_df.append(test_df, ignore_index=True)
    scaled, sc = scale(whole_df)
    x_train, y_train = build_train(scaled)
    x_pred = x_train[-20:]
    x_train = x_train[:-20]
    y_pred = y_train[-20:]
    y_train = y_train[:-20]
    return x_train, y_train, x_pred, y_pred, sc

if __name__ == "__main__":
    x_train, y_train, x_pred, y_pred, sc = load_all()
    print(x.shape, y.shape)