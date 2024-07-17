"""
neuralnet.py

What you need to do:
- Complete random_init
- Implement SoftMaxCrossEntropy methods
- Implement Sigmoid methods
- Implement Linear methods
- Implement NN methods

It is ***strongly advised*** that you finish the Written portion -- at the
very least, problems 1 and 2 -- before you attempt this programming 
assignment; the code for forward and backprop relies heavily on the formulas
you derive in those problems.

Sidenote: We annotate our functions and methods with type hints, which
specify the types of the parameters and the returns. For more on the type
hinting syntax, see https://docs.python.org/3/library/typing.html.
"""

import numpy as np
import argparse
from typing import Callable, List, Tuple
import matplotlib.pyplot as plt

# This takes care of command line argument parsing for you!
# To access a specific argument, simply access args.<argument name>.
parser = argparse.ArgumentParser()
parser.add_argument('train_input', type=str,
                    help='path to training input .csv file')
parser.add_argument('validation_input', type=str,
                    help='path to validation input .csv file')
parser.add_argument('train_out', type=str,
                    help='path to store prediction on training data')
parser.add_argument('validation_out', type=str,
                    help='path to store prediction on validation data')
parser.add_argument('metrics_out', type=str,
                    help='path to store training and testing metrics')
parser.add_argument('num_epoch', type=int,
                    help='number of training epochs')
parser.add_argument('hidden_units', type=int,
                    help='number of hidden units')
parser.add_argument('init_flag', type=int, choices=[1, 2],
                    help='weight initialization functions, 1: random')
parser.add_argument('learning_rate', type=float,
                    help='learning rate')


def args2data(args) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
str, str, str, int, int, int, float]:
    """
    DO NOT modify this function.

    Parse command line arguments, create train/test data and labels.
    :return:
    X_tr: train data *without label column and without bias folded in
        (numpy array)
    y_tr: train label (numpy array)
    X_te: test data *without label column and without bias folded in*
        (numpy array)
    y_te: test label (numpy array)
    out_tr: file for predicted output for train data (file)
    out_te: file for predicted output for test data (file)
    out_metrics: file for output for train and test error (file)
    n_epochs: number of train epochs
    n_hid: number of hidden units
    init_flag: weight initialize flag -- 1 means random, 2 means zero
    lr: learning rate
    """
    # Get data from arguments
    out_tr = args.train_out
    out_te = args.validation_out
    out_metrics = args.metrics_out
    n_epochs = args.num_epoch
    n_hid = args.hidden_units
    init_flag = args.init_flag
    lr = args.learning_rate

    X_tr = np.loadtxt(args.train_input, delimiter=',')
    y_tr = X_tr[:, 0].astype(int)
    X_tr = X_tr[:, 1:]  # cut off label column

    X_te = np.loadtxt(args.validation_input, delimiter=',')
    y_te = X_te[:, 0].astype(int)
    X_te = X_te[:, 1:]  # cut off label column

    return (X_tr, y_tr, X_te, y_te, out_tr, out_te, out_metrics,
            n_epochs, n_hid, init_flag, lr)


def shuffle(X, y, epoch):
    """
    DO NOT modify this function.

    Permute the training data for SGD.
    :param X: The original input data in the order of the file.
    :param y: The original labels in the order of the file.
    :param epoch: The epoch number (0-indexed).
    :return: Permuted X and y training data for the epoch.
    """
    np.random.seed(epoch)
    N = len(y)
    ordering = np.random.permutation(N)
    return X[ordering], y[ordering]


def zero_init(shape):
    """
    DO NOT modify this function.

    ZERO Initialization: All weights are initialized to 0.

    :param shape: list or tuple of shapes
    :return: initialized weights
    """
    return np.zeros(shape=shape)


def random_init(shape):
    """

    RANDOM Initialization: The weights are initialized randomly from a uniform
        distribution from -0.1 to 0.1.

    :param shape: list or tuple of shapes
    :return: initialized weights
    """
    M, D = shape
    np.random.seed(M * D)  # Don't change this line!

    # create a matrix of shape (M, D) with random values 
    # from -0.1 to 0.1 from a uniform distribution
    return np.random.uniform(-0.1, 0.1, size=(M, D))
    


class SoftMaxCrossEntropy:

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """
        Implement softmax function.
        :param z: input logits of shape (num_classes,)
        :return: softmax output of shape (num_classes,)
        """
        # take exponential of each element in z
        exp_z = np.exp(z)
        # sum of all elements in exp_z
        sum_exp_z = np.sum(exp_z)
        # divide each element in exp_z by sum_exp_z
        return exp_z / sum_exp_z



    def _cross_entropy(self, y: int, y_hat: np.ndarray) -> float:
        """
        Compute cross entropy loss.
        :param y: integer class label
        :param y_hat: prediction with shape (num_classes,)
        :return: cross entropy loss
        """
        # construct an np array whose y-th element is 1 and all other elements are 0
        y_one_hot = np.zeros(y_hat.shape)
        # set the y-th element to 1
        y_one_hot[y] = 1
        # compute single cross entropy vector
        loss_vector = -y_one_hot * np.log(y_hat)
        # compute cross entropy loss
        return np.sum(loss_vector)
    

    def forward(self, z: np.ndarray, y: int) -> Tuple[np.ndarray, float]:
        """
        Compute softmax and cross entropy loss.
        :param z: input logits of shape (num_classes,)
        :param y: integer class label
        :return:
            y: predictions from softmax as an np.ndarray
            loss: cross entropy loss
        """
        # calculate softmax output and cross entropy loss
        softwmax_out = self._softmax(z)
        loss_out = self._cross_entropy(y, softwmax_out)
        # return softmax_out and loss_out as a Tuple[np.ndarray, float]
        return (softwmax_out, loss_out)
        

    def backward(self, y: int, y_hat: np.ndarray) -> np.ndarray:
        """
        Compute gradient of loss w.r.t. ** softmax input **.
        Note that here instead of calculating the gradient w.r.t. the softmax
        probabilities, we are directly computing gradient w.r.t. the softmax
        input.

        Try deriving the gradient yourself (see Question 1.2(b) on the written),
        and you'll see why we want to calculate this in a single step.

        :param y: integer class label
        :param y_hat: predicted softmax probability with shape (num_classes,)
        :return: gradient with shape (num_classes,)
        """
        # get dldb
        # b has the same shape as z : (num_classes,) (which is the b in written)
        # dldb has the same shape as z : (num_classes,)

        # construct an np array whose y-th element is 1 and all other elements are 0
        y_one_hot = np.zeros(y_hat.shape)
        # set the y-th element to 1
        y_one_hot[y] = 1
        # subtract y_one_hot from y_hat
        dl_db = y_hat - y_one_hot
        # return dldb
        return dl_db



class Sigmoid:
    def __init__(self):
        """
        Initialize state for sigmoid activation layer
        """
        # TODO Initialize any additional values you may need to store for the
        #  backward pass here
        self.z = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Take sigmoid of input x.

        :param x: Input to activation function (i.e. output of the previous 
                  linear layer), with shape (output_size,)
                  Same thing as a in the written

        :return: Output of sigmoid activation function with shape
            (output_size,)
        """
        # return sigmoid(x) and store sigmoid(x) in self.z
        self.z = 1 / (1 + np.exp(-x))
        return self.z

    def backward(self, dz: np.ndarray) -> np.ndarray:
        """
        :param dz: partial derivative of loss with respect to output of
            sigmoid activation
            Same thing as dl/dz in the written

        :return: partial derivative of loss with respect to input of
            sigmoid activation
            Same thing as dl/da in the written
        """
        # return dz * sigmoid(x) * (1 - sigmoid(x))
        # dz/da = sigmoid(x) * (1 - sigmoid(x))
        return dz * self.z * (1 - self.z)


# This refers to a function type that takes in a tuple of 2 integers (row, col)
# and returns a numpy array (which should have the specified dimensions).
INIT_FN_TYPE = Callable[[Tuple[int, int]], np.ndarray]


class Linear:
    def __init__(self, input_size: int, output_size: int,
                 weight_init_fn: INIT_FN_TYPE, learning_rate: float):
        """
        :param input_size: number of units in the input of the layer 
                           *not including* the folded bias
        :param output_size: number of units in the output of the layer
        :param weight_init_fn: function that creates and initializes weight 
                               matrices for layer. This function takes in a 
                               tuple (row, col) and returns a matrix with
                               shape row x col.
        :param learning_rate: learning rate for SGD training updates
        """
        # Initialize learning rate for SGD
        self.lr = learning_rate

        # TODO: Initialize weight matrix for this layer - since we are
        #  folding the bias into the weight matrix, be careful about the
        #  shape you pass in.
        #  To be consistent with the formulas you derived in the written and
        #  in order for the unit tests to work correctly,
        #  the first dimension should be the output size
        self.W = weight_init_fn((output_size, input_size + 1))

        # TODO: set the bias terms to zero
        # this is for the case where it is generated by uniform distribution
        self.W[:, 0] = 0

        # TODO: Initialize matrix to store gradient with respect to weights
        self.dW = np.zeros(self.W.shape)

        # TODO: Initialize any additional values you may need to store for the
        #  backward pass here
        self._x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: Input to linear layer with shape (input_size,)
                  where input_size *does not include* the folded bias.
                  In other words, the input does not contain the bias column 
                  and you will need to add it in yourself in this method.
                  Since we train on 1 example at a time, batch_size should be 1
                  at training.

                  Here x is a 1D array, and corresponds to a single row in the
                  training data matrix X. Should do that in a loop in the NN
                  class.
                  
        :return: output z of linear layer with shape (output_size,)

        HINT: You may want to cache some of the values you compute in this
        function. Inspect your expressions for backprop to see which values
        should be cached.
        """
        
        # prepend 1 to x
        x = np.insert(x, 0, 1)
        # cache x
        self._x = x
        # return z
        return np.dot(self.W, x)


    def backward(self, dz: np.ndarray) -> np.ndarray:
        """
        :param dz: partial derivative of loss with respect to output z
            of linear
            Same as dl/da in the written

        :return: dx, partial derivative of loss with respect to input x
            of linear
            Same as dl/d(alpha) in the written
        
        Note that this function should set self.dw
            (gradient of weights with respect to loss)
            but not directly modify self.w; NN.step() is responsible for
            updating the weights.

        HINT: You may want to use some of the values you previously cached in 
        your forward() method.
        """
        # create a variable that stores transpose of self._x
        x_T = self._x.T
        # calculate self.dW
        self.dW = np.outer(dz, x_T)
        # calculate dx
        dx = np.dot(self.W.T, dz)
        # return dx
        return dx[1:]

        

    def step(self) -> None:
        """
        Apply SGD update to weights using self.dw, which should have been 
        set in NN.backward().
        """
        # update self.W
        self.W -= self.lr * self.dW



class NN:
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 weight_init_fn: INIT_FN_TYPE, learning_rate: float):
        """
        Initalize neural network (NN) class. Note that this class is composed
        of the layer objects (Linear, Sigmoid) defined above.

        :param input_size: number of units in input to network
        :param hidden_size: number of units in the hidden layer of the network
        :param output_size: number of units in output of the network - this
                            should be equal to the number of classes
        :param weight_init_fn: function that creates and initializes weight 
                               matrices for layer. This function takes in a 
                               tuple (row, col) and returns a matrix with 
                               shape row x col.
        :param learning_rate: learning rate for SGD training updates
        """
        self.weight_init_fn = weight_init_fn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # TODO: initialize modules (see section 9.1.2 of the writeup)
        #  Hint: use the classes you've implemented above!
        
        """
        # initialize dl_da and dl_db
        # these will be used for sgd update
        self.dl_da = None
        self.dl_db = None

        # initailize dl_dz and dl_dx
        # not sure these will every be used, but anyways
        self.dl_dz = None
        self.dl_dx = None
        """

        # initialize first linear layer module
        self.linear1 = Linear(input_size, hidden_size, weight_init_fn, learning_rate)
        # initialize the sigmoid module
        self.sigmoid = Sigmoid()
        # initialize the second linear layer module
        self.linear2 = Linear(hidden_size, output_size, weight_init_fn, learning_rate)
        # initialize the softmax module
        self.softmax = SoftMaxCrossEntropy()

    def forward(self, x: np.ndarray, y: int) -> Tuple[np.ndarray, float]:
        """
        Neural network forward computation. 
        Follow the pseudocode!
        :param x: input data point *without the bias folded in*
        :param y: prediction with shape (num_classes,)
        :return:
            y_hat: output prediction with shape (num_classes,). This should be
                a valid probability distribution over the classes.
            loss: the cross_entropy loss for a given example
        """
        # first linear layer
        # input: x, output: a
        a = self.linear1.forward(x)

        # sigmoid layer
        # input: a, output: z
        z = self.sigmoid.forward(a)

        # second linear layer
        # input: z, output: b
        b = self.linear2.forward(z)

        # softmax layer
        # input: b, output: y_hat
        y_hat, loss = self.softmax.forward(b, y)

        # return a tuple of y_hat and loss
        return (y_hat, loss)

    def backward(self, y: int, y_hat: np.ndarray) -> None:
        """
        Neural network backward computation.
        Follow the pseudocode!
        :param y: label (a number or an array containing a single element)
        :param y_hat: prediction with shape (num_classes,)
        """
        # softmax layer
        # input: y, y_hat, output: dl/db
        dl_db = self.softmax.backward(y, y_hat)

        # second linear layer
        # input: dl/db, output: dl/dz
        dl_dz = self.linear2.backward(dl_db)

        # sigmoid layer
        # input: dl/dz, output: dl/da
        dl_da = self.sigmoid.backward(dl_dz)

        # first linear layer
        # input: dl/da, output: dl/dx
        dl_dx = self.linear1.backward(dl_da)

    def step(self):
        """
        Apply SGD update to weights.
        """
        # update weights for first linear layer
        self.linear1.step()
        
        # update weights for second linear layer
        self.linear2.step()

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute nn's average (cross entropy) loss over the dataset (X, y)
        :param X: Input dataset of shape (num_points, input_size)
        :param y: Input labels of shape (num_points,)
        :return: Mean cross entropy loss
        """
        # TODO: compute loss over the entire dataset
        #  Hint: reuse your forward function
        
        # get the number of rows of X
        num_points = X.shape[0]
        # initialize loss
        loss = 0
        # loop through each row of X
        for i in range(num_points):
            # get the ith row of X
            x_i = X[i]
            # get the ith element of y
            y_i = y[i]
            # get the prediction and loss
            y_hat_i, loss_i = self.forward(x_i, y_i)
            # add lossi to loss
            loss += loss_i
        # divide loss by num_points
        loss = loss / num_points
        # return loss
        return loss

    def train(self, X_tr: np.ndarray, y_tr: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray,
              n_epochs: int) -> Tuple[List[float], List[float]]:
        """
        Train the network using SGD for some epochs.
        :param X_tr: train data
        :param y_tr: train label
        :param X_test: train data
        :param y_test: train label
        :param n_epochs: number of epochs to train for
        :return:
            train_losses: Training losses *after* each training epoch
            test_losses: Test losses *after* each training epoch
        """
        # no need to initailze alpha and beta
        # because they are alreayd initialized in the linear class

        # initialize train_losses and test_losses
        train_losses = []
        test_losses = []

        # loop through n_epochs
        for i in range(n_epochs):
            # shuffle the data
            X_tr_i, y_tr_i = shuffle(X_tr, y_tr, i)
            # loop through each row of X_tr_i
            for r in range(X_tr_i.shape[0]):
                # get the rth row of X_tr_i
                x_r = X_tr_i[r]
                # get the rth row of y_tr_i
                y_r = y_tr_i[r]
                # do forward prop
                y_hat_r, loss_r = self.forward(x_r, y_r)
                # back prop
                self.backward(y_r, y_hat_r)
                # update weights
                self.step()
            # Evaluate training mean cross-entropy loss
            train_loss = self.compute_loss(X_tr, y_tr)
            # Evaluate test mean cross-entropy loss
            test_loss = self.compute_loss(X_test, y_test)
            # append train_loss to train_losses
            train_losses.append(train_loss)
            # append test_loss to test_losses
            test_losses.append(test_loss)
        # return train_losses and test_losses
        return (train_losses, test_losses)

    
    def test(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute the label and error rate.
        :param X: input data
        :param y: label
        :return:
            labels: predicted labels
            error_rate: prediction error rate
        """
        # TODO: make predictions and compute error
        
        # get the number of rows of X
        num_points = X.shape[0]
        # initialize error rate
        error_rate = 0

        # initialize labels as a numpy array of zeros that has the same shape as y
        labels = np.zeros(y.shape)

        # loop through each row of X
        for r in range(num_points):
            # get the ith row of X
            x_r = X[r]
            # get the ith element of y
            y_r = y[r]

            # get the prediction
            y_hat_r, loss_r = self.forward(x_r, y_r)
            # get the index of the maximum value of the prediction
            # np.argmax automatically gets the smallest index
            index = np.argmax(y_hat_r)
            # add the predicted result to labels
            labels[r] = index

            # if the predicted result is not equal to the actual result
            if index != y_r:
                # increment error rate
                error_rate += 1
        
        # do the error rate calculation
        error_rate = error_rate / num_points
        # change every element in labels to an integer
        labels = labels.astype(int)
        # return labels and error rate
        return (labels, error_rate)

            


if __name__ == "__main__":
    args = parser.parse_args()
    # Note: You can access arguments like learning rate with args.learning_rate
    # Generally, you can access each argument using the name that was passed 
    # into parser.add_argument() above (see lines 24-44).

    # Define our labels
    labels = ["a", "e", "g", "i", "l", "n", "o", "r", "t", "u"]

    # Call args2data to get all data + argument values
    # See the docstring of `args2data` for an explanation of 
    # what is being returned.
    (X_tr, y_tr, X_test, y_test, out_tr, out_te, out_metrics,
     n_epochs, n_hid, init_flag, lr) = args2data(args)
    

    # initailize a list of three learning rates
    learning_rates = [0.03, 0.003, 0.0003]

    # loop through the learning rates
    for lr in learning_rates:
        # initialize the neural network
        nn = NN(
            input_size=X_tr.shape[-1],
            hidden_size=n_hid,
            output_size=len(labels),
            weight_init_fn=zero_init if init_flag == 2 else random_init,
            learning_rate=lr
        )

        # train model
        # (this line of code is already written for you)
        train_losses, test_losses = nn.train(X_tr, y_tr, X_test, y_test, n_epochs)

        # draw the plot
        # x-axis is the number of epochs
        x = np.arange(100)
        # y-axis is the training loss
        y_1 = train_losses
        # another y-axis is the validation loss
        z_2 = test_losses
        # plot the training loss
        plt.plot(x, y_1, label="Training Loss")
        # plot the validation loss
        plt.plot(x, z_2, label="Validation Loss")
        # add a legend
        plt.legend()
        # add a title
        plt.title("Cross Entropy Loss with Learning Rate: " + str(lr))
        # add x-axis label
        plt.xlabel("Number of Epochs")
        # add y-axis label
        plt.ylabel("Cross Entropy Loss")
        # save the plot with corresponding learning rate
        plt.savefig("lr_" + str(lr) + ".png")
        # clear the plot
        plt.clf()



 