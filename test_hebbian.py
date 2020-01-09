# Shanker, Avinash
# 1001-668-570
# 2019-09-06
# Assignment-02-02

import numpy as np
import pytest
from hebbian import Hebbian
import tensorflow as tf


def test_weight_dimension():
    input_dimensions = 4
    number_of_classes = 9
    model = Hebbian(input_dimensions=input_dimensions, number_of_classes=number_of_classes,
                    transfer_function="Hard_limit")
    assert model.weights.ndim == 2 and \
           model.weights.shape[0] == number_of_classes and \
           model.weights.shape[1] == (input_dimensions + 1)


def test_weight_initialization():
    input_dimensions = 2
    number_of_classes = 5
    model = Hebbian(input_dimensions=2, number_of_classes=number_of_classes,
                    transfer_function="Hard_limit",seed=1)
    assert model.weights.ndim == 2 and model.weights.shape[0] == number_of_classes and model.weights.shape[
        1] == input_dimensions + 1
    weights = np.array([[1.62434536, -0.61175641, -0.52817175],
                        [-1.07296862, 0.86540763, -2.3015387],
                        [1.74481176, -0.7612069, 0.3190391],
                        [-0.24937038, 1.46210794, -2.06014071],
                        [-0.3224172, -0.38405435, 1.13376944]])
    np.testing.assert_allclose(model.weights, weights, rtol=1e-3, atol=1e-3)
    model.initialize_all_weights_to_zeros()
    assert np.array_equal(model.weights, np.zeros((number_of_classes, input_dimensions + 1)))


def test_predict():
    input_dimensions = 2
    number_of_classes = 2
    model = Hebbian(input_dimensions=input_dimensions, number_of_classes=number_of_classes,
                    transfer_function="Hard_limit",seed=1)
    X_train = np.array([[-1.43815556, 0.10089809, -1.25432937, 1.48410426],
                        [-1.81784194, 0.42935033, -1.2806198, 0.06527391]])
    model.initialize_all_weights_to_zeros()
    Y_hat = model.predict(X_train)
    assert (np.array_equal(Y_hat, np.array([[0,0,0,0], [0,0,0,0]]))) or \
           (np.array_equal(Y_hat, np.array([[1,1,1,1], [1,1,1,1]])))
def test_predict_2():
    number_of_classes = 10
    number_of_training_samples_to_use = 3
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train_vectorized=((X_train.reshape(X_train.shape[0],-1)).T)[:,0:number_of_training_samples_to_use]
    y_train = y_train[0:number_of_training_samples_to_use]
    input_dimensions=X_train_vectorized.shape[0]
    model = Hebbian(input_dimensions=input_dimensions, number_of_classes=number_of_classes,
                    transfer_function="Linear",seed=5)
    Y_hat=model.predict(X_train_vectorized)
    np.testing.assert_almost_equal(Y_hat, np.array( \
        [[-4101.41409432, -3820.1709349, -1235.86331202],
         [854.07167203, 4061.22006877, 434.40971256],
         [-552.17756811, -1791.61373625, -2591.16737069],
         [-355.4367891, -3858.75847581, 1320.1141753],
         [1087.86080571, -607.49532925, -460.80234811],
         [-2459.84338339, -1681.30331925, -255.00327678],
         [-460.58803655, -4439.85928602, -1093.82071536],
         [4066.25628304, 4814.90762933, 1955.78972208],
         [564.64444411, -3117.99849963, -419.49244877],
         [-2374.84426405, -2878.08764629, 2979.99404738]]), decimal=2)
    model = Hebbian(input_dimensions=input_dimensions, number_of_classes=number_of_classes,
                    transfer_function="Hard_limit",seed=5)
    Y_hat=model.predict(X_train_vectorized)
    assert (np.allclose(Y_hat, np.array( \
        [[0, 0, 0],
         [1, 1, 1],
         [0, 0, 0],
         [0, 0, 1],
         [1, 0, 0],
         [0, 0, 0],
         [0, 0, 0],
         [1, 1, 1],
         [1, 0, 0],
         [0, 0, 1]]),rtol=1e-3, atol=1e-3))
    model = Hebbian(input_dimensions=input_dimensions, number_of_classes=number_of_classes,
                    transfer_function="Hard_limit", seed=5)
    Y_hat = model.predict(X_train_vectorized)
    assert np.allclose(Y_hat, np.array( \
        [[0.00000000e+000, 0.00000000e+000, 0.00000000e+000],
         [1.00000000e+000, 1.00000000e+000, 1.00000000e+000],
         [1.55714531e-240, 0.00000000e+000, 0.00000000e+000],
         [4.32278692e-155, 0.00000000e+000, 1.00000000e+000],
         [1.00000000e+000, 1.47275575e-264, 7.51766500e-201],
         [0.00000000e+000, 0.00000000e+000, 1.79260263e-111],
         [9.31445172e-201, 0.00000000e+000, 0.00000000e+000],
         [1.00000000e+000, 1.00000000e+000, 1.00000000e+000],
         [1.00000000e+000, 0.00000000e+000, 6.55759060e-183],
         [0.00000000e+000, 0.00000000e+000, 1.00000000e+000]]),rtol=1e-3, atol=1e-3)
def test_confusion_matrix():
    # Read mnist data
    number_of_classes = 10
    number_of_test_samples_to_use = 100
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_test_vectorized = ((X_test.reshape(X_test.shape[0], -1)).T)[:, 0:number_of_test_samples_to_use]
    y_test = y_test[0:number_of_test_samples_to_use]
    input_dimensions = X_test_vectorized.shape[0]
    model = Hebbian(input_dimensions=input_dimensions, number_of_classes=number_of_classes,
                    transfer_function="Sigmoid", seed=5)
    confusion_matrix = model.calculate_confusion_matrix(X_test_vectorized, y_test)
    assert np.array_equal(confusion_matrix, np.array(\
        [ [0.,6.,2.,0.,0.,0.,0.,0.,0.,0.],
        [1., 10.,0.,3.,0.,0.,0.,0.,0.,0.],
        [0.,6.,1.,1.,0.,0.,0.,0.,0.,0.],
         [0.,8.,0.,3.,0.,0.,0.,0.,0.,0.],
         [1.,11.,1.,1.,0.,0.,0.,0.,0.,0.],
         [1.,5.,1.,0.,0.,0.,0.,0.,0.,0.],
         [0.,9.,0.,1.,0.,0.,0.,0.,0.,0.],
         [0.,7.,4.,1.,3.,0.,0.,0.,0.,0.],
         [0.,2.,0.,0.,0.,0.,0.,0.,0.,0.],
         [1.,7.,2.,1.,0.,0.,0.,0.,0.,0.]]))
    model = Hebbian(input_dimensions=input_dimensions, number_of_classes=number_of_classes,
                    transfer_function="Linear", seed=5)
    confusion_matrix = model.calculate_confusion_matrix(X_test_vectorized, y_test)
    assert np.array_equal(confusion_matrix, np.array( \
        [[0., 1., 1., 0., 5., 0., 0., 1., 0., 0.],
         [0., 1., 0., 0., 2., 0., 0., 11., 0., 0.],
         [0., 1., 0., 1., 4., 0., 1., 1., 0., 0.],
         [0., 0., 0., 3., 3., 0., 1., 4., 0., 0.],
         [0., 4., 0., 0., 6., 0., 0., 4., 0., 0.],
         [0., 1., 1., 0., 2., 0., 0., 2., 0., 1.],
         [0., 1., 0., 0., 3., 0., 0., 6., 0., 0.],
         [0., 0., 0., 0., 8., 0., 0., 4., 3., 0.],
         [0., 0., 1., 0., 1., 0., 0., 0., 0., 0.],
         [0., 2., 0., 1., 1., 0., 0., 7., 0., 0.]]))
    model = Hebbian(input_dimensions=input_dimensions, number_of_classes=number_of_classes,
                    transfer_function="Hard_limit", seed=5)
    confusion_matrix = model.calculate_confusion_matrix(X_test_vectorized, y_test)
    assert np.array_equal(confusion_matrix, np.array( \
        [[0., 6., 2., 0., 0., 0., 0., 0., 0., 0.],
         [1., 10., 0., 3., 0., 0., 0., 0., 0., 0.],
         [0., 6., 1., 1., 0., 0., 0., 0., 0., 0.],
         [0., 8., 0., 3., 0., 0., 0., 0., 0., 0.],
         [1., 12., 0., 1., 0., 0., 0., 0., 0., 0.],
         [1., 5., 1., 0., 0., 0., 0., 0., 0., 0.],
         [0., 9., 0., 1., 0., 0., 0., 0., 0., 0.],
         [0., 7., 4., 1., 3., 0., 0., 0., 0., 0.],
         [0., 2., 0., 0., 0., 0., 0., 0., 0., 0.],
         [1., 7., 2., 1., 0., 0., 0., 0., 0., 0.]]))

def test_percent_error():
    number_of_classes = 10
    number_of_test_samples_to_use = 100
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_test_vectorized = ((X_test.reshape(X_test.shape[0], -1)).T)[:, 0:number_of_test_samples_to_use]
    y_test = y_test[0:number_of_test_samples_to_use]
    input_dimensions = X_test_vectorized.shape[0]
    model = Hebbian(input_dimensions=input_dimensions, number_of_classes=number_of_classes,
                    transfer_function="Sigmoid", seed=5)
    percent_error = model.calculate_percent_error(X_test_vectorized, y_test)
    np.testing.assert_almost_equal(percent_error,0.86,decimal=2)
    model = Hebbian(input_dimensions=input_dimensions, number_of_classes=number_of_classes,
                    transfer_function="Linear", seed=15)
    percent_error = model.calculate_percent_error(X_test_vectorized, y_test)
    np.testing.assert_almost_equal(percent_error,0.96,decimal=2)
    model = Hebbian(input_dimensions=input_dimensions, number_of_classes=number_of_classes,
                    transfer_function="Hard_limit", seed=8)
    percent_error = model.calculate_percent_error(X_test_vectorized, y_test)
    np.testing.assert_almost_equal(percent_error,0.91,decimal=2)

def test_training():
    number_of_classes = 10
    number_of_training_samples_to_use = 1000
    number_of_test_samples_to_use = 100
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train_vectorized = ((X_train.reshape(X_train.shape[0], -1)).T)[:, 0:number_of_training_samples_to_use]
    y_train = y_train[0:number_of_training_samples_to_use]
    X_test_vectorized = ((X_test.reshape(X_test.shape[0], -1)).T)[:, 0:number_of_test_samples_to_use]
    y_test = y_test[0:number_of_test_samples_to_use]
    input_dimensions = X_test_vectorized.shape[0]
    model = Hebbian(input_dimensions=input_dimensions, number_of_classes=number_of_classes,
                    transfer_function="Hard_limit", seed=5)
    model.initialize_all_weights_to_zeros()
    percent_error = []
    for k in range(10):
        model.train(X_train_vectorized, y_train, batch_size=300, num_epochs=2, alpha=0.1, gamma=0.1, learning="Delta")
        percent_error.append(model.calculate_percent_error(X_test_vectorized, y_test))
    confusion_matrix = model.calculate_confusion_matrix(X_test_vectorized, y_test)
    assert (np.array_equal(confusion_matrix, np.array( \
        [[8., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [1., 13., 0., 0., 0., 0., 0., 0., 0., 0.],
         [1., 0., 7., 0., 0., 0., 0., 0., 0., 0.],
         [2., 0., 1., 8., 0., 0., 0., 0., 0., 0.],
         [1., 0., 0., 1., 12., 0., 0., 0., 0., 0.],
         [4., 0., 1., 0., 0., 2., 0., 0., 0., 0.],
         [3., 0., 2., 0., 0., 0., 5., 0., 0., 0.],
         [1., 0., 0., 2., 0., 0., 0., 11., 0., 1.],
         [2., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [1., 0., 0., 0., 0., 0., 0., 1., 0., 9.]]))) or \
           (np.array_equal(confusion_matrix, np.array( \
               [[8., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 13., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 0., 6., 0., 0., 0., 1., 0., 0., 0.],
                [2., 0., 1., 8., 0., 0., 0., 0., 0., 0.],
                [2., 0., 0., 1., 11., 0., 0., 0., 0., 0.],
                [4., 0., 1., 0., 0., 2., 0., 0., 0., 0.],
                [4., 0., 1., 0., 0., 0., 5., 0., 0., 0.],
                [2., 0., 0., 1., 0., 0., 0., 12., 0., 0.],
                [1., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                [3., 0., 0., 0., 0., 0., 0., 3., 0., 5.]])))

    assert np.allclose(percent_error,
    np.array([0.74, 0.35, 0.32, 0.3, 0.28, 0.32, 0.25, 0.26, 0.3, 0.25]),rtol=1e-3, atol=1e-3) or \
           np.allclose(percent_error,
                        np.array([0.8 ,0.37,0.36,0.32,0.31,0.31,0.29,0.29,0.24,0.29]), rtol=1e-3, atol=1e-3)
    model = Hebbian(input_dimensions=input_dimensions, number_of_classes=number_of_classes,
                    transfer_function="Linear",seed=5)
    percent_error=[]
    for k in range (10):
        model.train(X_train_vectorized, y_train,batch_size=300, num_epochs=2, alpha=0.1,gamma=0.1,learning="Filtered")
        percent_error.append(model.calculate_percent_error(X_test_vectorized,y_test))
    confusion_matrix=model.calculate_confusion_matrix(X_test_vectorized,y_test)
    assert np.array_equal(confusion_matrix, np.array( \
        [[8., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 11., 0., 1., 0., 0., 0., 0., 2., 0.],
         [2., 0., 4., 1., 0., 0., 0., 1., 0., 0.],
         [2., 0., 1., 8., 0., 0., 0., 0., 0., 0.],
         [1., 0., 0., 0., 11., 0., 0., 1., 0., 1.],
         [5., 0., 0., 1., 0., 0., 0., 1., 0., 0.],
         [3., 0., 1., 0., 0., 0., 6., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 15., 0., 0.],
         [0., 0., 1., 0., 0., 0., 0., 0., 1., 0.],
         [0., 0., 0., 0., 0., 0., 0., 8., 1., 2.]]))
    np.testing.assert_almost_equal(percent_error,
        [0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34], decimal=2)
