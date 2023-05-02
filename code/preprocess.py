import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

def prepare_data(w, scaled_data, N, f):
    X=[]
    window = w + f
    Q = len(scaled_data)
    for i in range(Q-window+1):
        X.append(scaled_data[i:i+window, 0])

    X = np.array(X)
    X = np.reshape(X, (X.shape[0],X.shape[1],1))

    trainX, trainY = X[0:N,0:w], X[0:N,w:w+f]
    testX, testY = X[N:Q-w,0:w], X[N:Q-w,w:w+f]
    return trainX, trainY, testX, testY, X

def baselinef(U,f):
    last = U.shape[0]
    yhat = np.zeros((last, f))
    for j in range(0,last):
        yhat[j,0:f] = np.repeat(U[j,U.shape[1]-1], f)
    return yhat

def scaleit(DATAX):
    mima = np.zeros((DATAX.shape[0], 2)) #To save min and max values
    for i in range(DATAX.shape[0]):
        mima[i,0],mima[i,1] = DATAX[i,:].min(), DATAX[i,:].max()
        DATAX[i,:] = (DATAX[i,:]-DATAX[i,:].min())/(DATAX[i,:].max()-DATAX[i,:].min())
    return DATAX, mima

def load_data_from_csv(filename):
    return np.genfromtxt(filename, delimiter=',')


window_size = 60
test_samples = 251
future_time_steps = 20

def get_data(window_size, test_samples, future_time_steps):
    # Data Preparation
    data = load_data_from_csv('../data/BANKEX.csv')
    scaled_data, min_max = scaleit(data)
    scaled_data = data[0, :]
    scaled_data = np.reshape(scaled_data, (len(scaled_data), 1))
    print(scaled_data.shape) # check 


    num_samples = len(scaled_data) - test_samples - window_size

    # Scaling Data
    
    trainX, trainY, testX, testY, X = prepare_data(window_size, scaled_data, num_samples, future_time_steps)
    print(trainX.shape, trainY.shape, testX.shape, testY.shape)

    return trainX, trainY, testX, testY, X, min_max

# if __name__ == "__main__":
#     # Add your parameters here
#     main(window_size, test_samples, future_time_steps)
