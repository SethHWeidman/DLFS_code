import numpy as np
import pandas as pd
import random
random.seed(1104)


def generate_x_y(random_seed=1104):
    random.seed(random_seed)
    randX = random.sample(range(1,9), 8)
    mapping_dict = {1:[0,0,0],
                 2:[0,0,1],
                 3:[0,1,0],
                 4:[0,1,1],
                 5:[1,0,0],
                 6:[1,0,1],
                 7:[1,1,0],
                 8:[1,1,1]}
    X = []
    for r in randX:
        X.append(mapping_dict[r])

    y = []
    randy = random.sample(range(1,9), 8)
    for el in randy:
        if el % 2 == 0:
            y.append([0])
        else:
            y.append([1])


    X = np.array(X)
    y = np.array(y)

    df = pd.DataFrame(data=np.concatenate([X, y], axis=1),
                      columns=['X1', 'X2', 'X3', 'y'])

    return df, X, y

def array_print(array, round_num=2):
    '''
    `array` is 2 dimensional
    '''
    assert array.ndim == 2
    print("The array:\n", 
          np.round(array, round_num))
    text_lookup = {"rows": {"one": "row", "other": "rows"}, "columns": {"one": "column", "other": "columns"}} 
    if array.shape[0] == 1:
        rows_text = text_lookup['rows']['one']
    else:
        rows_text = text_lookup['rows']['other']
    if array.shape[1] == 1:
        columns_text = text_lookup['columns']['one']
    else:
        columns_text = text_lookup['columns']['other']
    print("The dimensions are", 
          array.shape[0], 
          rows_text,
          "and",
          array.shape[1],
          columns_text)
    

def df_print(df, round_num=2):
    print(np.round(df, 2))

    
def target_to_y(target):
    Y = np.zeros((len(target), 10))
    for i in range(len(target)):
        Y[i][int(target[i])] = 1
    return Y


def data_to_x(data):
    X = (data - data.min()) * 1.0 / (data.max() - data.min())
    return X


def get_mnist_X_Y(mnist):
    data = mnist.data
    target = mnist.target
    X = data_to_x(data)
    Y = target_to_y(target)
    return X, Y

###
### Neural net functions
###

def initialize_weights(num_in=3, num_hidden=4, num_out=1):
    '''
    Randomly initializes weights
    '''
    np.random.seed(1104)
    V = np.random.randn(num_in, num_hidden)
    W = np.random.randn(num_hidden, num_out)
    return V, W


def shuffle_x_y(X, Y):
    '''
    Each array must be two dimensional
    '''
    np.random.seed(1104)
    train_size = X.shape[0]
    indices = list(range(train_size))
    np.random.shuffle(indices)
    return X[indices], Y[indices]


def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))


def learn(V, W, x_batch, y_batch):
    # forward pass
    A = np.dot(x_batch,V)
    B = sigmoid(A)
    C = np.dot(B,W)
    P = sigmoid(C)
    
    # loss
    L = 0.5 * (y_batch - P) ** 2
    
    # backpropogation
    dLdP = -1.0 * (y_batch - P)
    dPdC = sigmoid(C) * (1-sigmoid(C))
    dLdC = dLdP * dPdC
    dCdW = B.T
    dLdW = np.dot(dCdW, dLdC)
    dCdB = W.T
    dBdA = sigmoid(A) * (1-sigmoid(A))
    dAdV = x_batch.T
    dLdV = np.dot(dAdV, np.dot(dLdP * dPdC, dCdB) * dBdA)
    
    # update the weights
    W -= dLdW
    V -= dLdV
    
    return V, W


def one_epoch(X, Y, V, W):
    '''
    Run one epoch an element at a time through the net.
    '''
    for index in range(X.shape[0]):
        x_batch = np.array(X[index], ndmin=2)
        y_batch = np.array(Y[index], ndmin=2)
        learn(V, W, x_batch, y_batch)
    return V, W


def predict(x_batch, V, W):
    '''
    Make a prediction given a batch of observations and the weights.
    '''
    A = np.dot(x_batch, V)
    B = sigmoid(A)
    C = np.dot(B, W)
    P = sigmoid(C)
    return P


def loss(prediction, actual, print_loss=False):
    '''
    Calculate the loss as mean squared error.
    '''
    return np.mean((prediction - actual) ** 2)


def train(X, Y, V, W, epochs=100):
    '''
    Train the net for a number of epochs.
    '''
    losses = []
    epochs_list = []
    for i in range(epochs+1):
        V, W = one_epoch(X, Y, V, W)
        if i % (epochs / 10) == 0:
            preds = predict(X, V, W)
            loss_epoch = loss(preds, Y)
            epochs_list.append(i)
            losses.append(loss_epoch)
    return pd.DataFrame({'epoch' : epochs_list, 
                         'loss' : losses})


def train_and_display(X, Y, num_epochs=1000, num_hidden=8):
    X, Y = shuffle_x_y(X, Y)
    V, W = initialize_weights(num_in=X.shape[1], 
                              num_hidden=num_hidden, 
                              num_out=Y.shape[1])
    df = train(X, Y, V, W, num_epochs)
    df_print(df)
    return V, W


def accuracy_binary(X, Y, V, W):
    
    def _df_actual_predicted(X, Y, V, W):
        return pd.DataFrame(np.round(np.concatenate([Y, predict(X, V, W)], axis=1), 2),
                            columns=["Actual", "Predicted"])

    df = _df_actual_predicted(X, Y, V, W)
    print("The data frame of the predictions this neural net produces is:\n",
          df)
    df['Prediction'] = df['Predicted'] > 0.5

    def _correct_prediction(row):
        return bool(row['Actual']) == row['Prediction']
    
    df['Correct'] = df.apply(lambda x: _correct_prediction(x), axis=1)
        
    print("The accuracy of this trained neural net is", 
          df['Correct'].sum() / len(df['Correct']))
    
    return df['Correct'].sum() / len(df['Correct'])


def accuracy_multiclass(X, Y, V, W):
    predictions = predict(X, V, W)

    preds = [np.argmax(x) for x in predictions]
    actuals = [np.argmax(x) for x in Y]
    accuracy = sum(np.array(preds) == np.array(actuals)) * 1.0 / len(preds)
    return accuracy

