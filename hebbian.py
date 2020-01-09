# Shanker, Avinash
# 1001-668-570
# 2019-09-06
# Assignment-02-01

import numpy as np
import itertools
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf


def display_images(images):
	# This function displays images on a grid.
	# Farhad Kamangar Sept. 2019
	number_of_images=images.shape[0]
	number_of_rows_for_subplot=int(np.sqrt(number_of_images))
	number_of_columns_for_subplot=int(np.ceil(number_of_images/number_of_rows_for_subplot))
	for k in range(number_of_images):
		plt.subplot(number_of_rows_for_subplot,number_of_columns_for_subplot,k+1)
		plt.imshow(images[k], cmap=plt.get_cmap('gray'))
		# plt.imshow(images[k], cmap=pyplot.get_cmap('gray'))
	plt.show()

def display_numpy_array_as_table(input_array):
	# This function displays a 1d or 2d numpy array (matrix).
	# Farhad Kamangar Sept. 2019
	if input_array.ndim==1:
		num_of_columns,=input_array.shape
		temp_matrix=input_array.reshape((1, num_of_columns))
	elif input_array.ndim>2:
		print("Input matrix dimension is greater than 2. Can not display as table")
		return
	else:
		temp_matrix=input_array
	number_of_rows,num_of_columns = temp_matrix.shape
	plt.figure()
	tb = plt.table(cellText=np.round(temp_matrix,2), loc=(0,0), cellLoc='center')
	for cell in tb.properties()['child_artists']:
	    cell.set_height(1/number_of_rows)
	    cell.set_width(1/num_of_columns)

	ax = plt.gca()
	ax.set_xticks([])
	ax.set_yticks([])
	plt.show()
class Hebbian(object):
    def __init__(self, input_dimensions=2,number_of_classes=4,transfer_function="Hard_limit",seed=None):
        """
        Initialize Perceptron model
        :param input_dimensions: The number of features of the input data, for example (height, weight) would be two features.
        :param number_of_classes: The number of classes.
        :param transfer_function: Transfer function for each neuron. Possible values are:
        "Hard_limit" ,  "Sigmoid", "Linear".
        :param seed: Random number generator seed.
        """
        if seed != None:
            np.random.seed(seed)
        self.input_dimensions = input_dimensions
        self.number_of_classes=number_of_classes
        self.transfer_function=transfer_function
        self._initialize_weights()
    def _initialize_weights(self):
        """
        Initialize the weights, initalize using random numbers.
        Note that number of neurons in the model is equal to the number of classes
        """
        self.weights = np.random.randn(self.number_of_classes,self.input_dimensions+1)
        return self.weights
        raise Warning("You must implement _initialize_weights! This function should initialize (or re-initialize) your model weights. Bias should be included in the weights")

    def initialize_all_weights_to_zeros(self):
        """
        Initialize the weights, initalize using random numbers.
        """
        self.weights = np.zeros((self.number_of_classes,self.input_dimensions+1))
        return self.weights
        raise Warning("You must implement this function! This function should initialize (or re-initialize) your model weights to zeros. Bias should be included in the weights")

    def predict(self, X):
        """
        Make a prediction on an array of inputs
        :param X: Array of input [input_dimensions,n_samples]. Note that the input X does not include a row of ones
        as the first row.
        :return: Array of model outputs [number_of_classes ,n_samples]. This array is a numerical array.
        """
        ##Add_one is using vstack function to stack 1 in the first row.
        ##Dot product of weights with inputs
        ## n = X.W
        Add_Ones = np.vstack((np.ones(X.shape[1]),X))
        model_product=np.dot(self.weights,Add_Ones)
        ##Model_product does the prediction as a Hadlim function
        #to diffrentiate between sigmoid and hardlimit
        if self.transfer_function == "Sigmoid":
            expo = np.exp(np.around(-model_product))
            model_product = 1.0/(1.0 + expo)
        elif self.transfer_function == "Hard_limit":
            model_product[model_product >= 0] = 1
            model_product[model_product < 0 ] = 0
        else:
            model_product == model_product
        return model_product
        raise Warning("You must implement predict. This function should make a prediction on a matrix of inputs")


    def print_weights(self):
        """
        This function prints the weight matrix (Bias is included in the weight matrix).
        """
        ##Printing weight matrix with Bias
        Matrix_Bias = self.weights
        print("****** Weight Matrix With Bias ******\n",Matrix_Bias)
        raise Warning("You must implement print_weights")
        
    def train(self, X, y, batch_size=1,num_epochs=10,  alpha=0.1,gamma=0.9,learning="Delta"):
        """
        Given a batch of data, and the necessary hyperparameters,
        this function adjusts the self.weights using Perceptron learning rule.
        Training should be repeted num_epochs time.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :param batch_size: number of samples in a batch
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :param gamma: Controls the decay
        :param learning: Learning rule. Possible methods are: "Filtered", "Delta", "Unsupervised_hebb"
        :return: None
        """
        input_col = X.shape[1]
        target_row = y.shape[0]
        X = np.vstack((np.ones(input_col),X))
        zero_mat = np.zeros((self.number_of_classes,target_row))
        for x in range(target_row):
            zero_mat[y[x], x] = 1.0
        
        for epochs in range(num_epochs):
            #running the loop in batches
            for i in range(0, input_col, batch_size):
                end = i + batch_size
                if end > X.shape[1]:
                    end = X.shape[1]
                q = X[:, i : end]
                t = zero_mat[:, i : end]
                a = np.dot(self.weights,q)
                if self.transfer_function == 'Sigmoid':
                    for every in a:
                        for x in range(len(every)):
                        	expo = np.exp(-every[x])
                        every[x] = 1.0 / (1.0 + expo)
                elif self.transfer_function == 'Hard_limit':
                    a[a >= 0] = 1
                    a[a < 0 ] = 0
                #Functions to separate the learning rule type
                if learning == "Filtered":
                    self.weights = (self.weights * (1-gamma)) + alpha * (np.dot(t,q.T))
                elif learning == "Delta":
                    self.weights = self.weights + alpha * (np.dot((t-a),q.T))
                elif learning == "Unsupervised_hebb":
                    self.weights = self.weights + alpha * (np.dot(a, q.T))


    def calculate_percent_error(self,X, y):
        """
        Given a batch of data this function calculates percent error.
        For each input sample, if the predicted class output is not the same as the desired class,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return percent_error
        """
        predicted = self.predict(X)
        actual = np.argmax(predicted, axis = 0)
        err = 0.0
        for x in range(predicted.shape[1]):
            if (actual[x] != y[x]):
                err = err + 1
        value = err/X.shape[1]
        return value
            
        
    def calculate_confusion_matrix(self,X,y):
        """
        Given a desired (true) output as one hot and the predicted output as one-hot,
        this method calculates the confusion matrix.
        If the predicted class output is not the same as the desired output,
        then it is considered one error.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return confusion_matrix[number_of_classes,number_of_classes].
        Confusion matrix should be shown as the number of times that
        an image of class n is classified as class m where 1<=n,m<=number_of_classes.
        """
        #creating zero matrix for consuion matrix
        zero_mat = np.zeros((self.number_of_classes,self.number_of_classes))
        predicted = self.predict(X)
        actual = np.argmax(predicted, axis = 0)
        predicted_col = predicted.shape[1]
        for x in range(predicted_col):
            zero_mat[y[x], actual[x]] = 1.0 + zero_mat[y[x], actual[x]]
        return zero_mat
        



if __name__ == "__main__":

    # Read mnist data
    number_of_classes = 10
    number_of_training_samples_to_use = 700
    number_of_test_samples_to_use = 100
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train_vectorized=((X_train.reshape(X_train.shape[0],-1)).T)[:,0:number_of_training_samples_to_use]
    y_train = y_train[0:number_of_training_samples_to_use]
    X_test_vectorized=((X_test.reshape(X_test.shape[0],-1)).T)[:,0:number_of_test_samples_to_use]
    y_test = y_test[0:number_of_test_samples_to_use]
    number_of_images_to_view=16
    test_x=X_train_vectorized[:,0:number_of_images_to_view].T.reshape((number_of_images_to_view,28,28))
    display_images(test_x)
    input_dimensions=X_test_vectorized.shape[0]
    model = Hebbian(input_dimensions=input_dimensions, number_of_classes=number_of_classes,
                    transfer_function="Hard_limit",seed=5)
    model.initialize_all_weights_to_zeros()
    percent_error=[]
    for k in range (10):
        model.train(X_train_vectorized, y_train,batch_size=300, num_epochs=2, alpha=0.1,gamma=0.1,learning="Delta")
        percent_error.append(model.calculate_percent_error(X_test_vectorized,y_test))
    print("******  Percent Error ******\n",percent_error)
    confusion_matrix=model.calculate_confusion_matrix(X_test_vectorized,y_test)
    print(np.array2string(confusion_matrix, separator=","))
