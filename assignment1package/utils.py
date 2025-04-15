#import a few libraries. Numpy is named as np and pyplot in matplotlib as plt
import urllib.request
import pandas
import numpy as np
import matplotlib.pyplot as plt
import copy
from mnist import MNIST
import gzip
import os

#set numpy to raise exceptions when encountering numerical errors
np.seterr(all='raise')

#this function is used to convert from integer encoding of labels to one hot encoding
# labels is an 1-D array with the integer labels from 0 to n_labels. 
def one_hot(labels, n_labels):
    return np.squeeze(np.eye(n_labels)[labels.reshape(-1)])

def preprocess_medical_data(data_ex3):
    # Make a list of all the input data column names
    x_columns_ex3 = ['X' + str(i+1) for i in range(178)]

    #randomly get the row that are part of each split of the data
    all_examples_indices_ex3 = np.array(range(len(data_ex3)))
    np.random.seed(1)
    np.random.shuffle(all_examples_indices_ex3)
    train_subjects_ex3 = all_examples_indices_ex3[:int(len(data_ex3)*0.6)]
    val_subjects_ex3 = all_examples_indices_ex3[int(len(data_ex3)*0.6):int(len(data_ex3)*0.75)]
    test_subjects_ex3 = all_examples_indices_ex3[int(len(data_ex3)*0.75):]

    # normalizing input data is very important for a good learning of these kind of networks
    # only the training set should be used to compute statistics so that information from 
    # the validation set or the test set is not included in our learning.
    non_normalized_training_data_ex3 = data_ex3.iloc[train_subjects_ex3][x_columns_ex3].values
    normalization_mean_ex3 = non_normalized_training_data_ex3.mean(axis = 0)
    normalization_std_ex3 = non_normalized_training_data_ex3.std(axis = 0)
    train_data_ex3 = (non_normalized_training_data_ex3 - normalization_mean_ex3)/normalization_std_ex3/2
    val_data_ex3 = (data_ex3.iloc[val_subjects_ex3][x_columns_ex3].values - normalization_mean_ex3)/normalization_std_ex3/2
    test_data_ex3 = (data_ex3.iloc[test_subjects_ex3][x_columns_ex3].values - normalization_mean_ex3)/normalization_std_ex3/2

    #get the labels for all splits and convert them to one-hot encoding
    train_labels_ex3 = data_ex3.iloc[train_subjects_ex3]['y'].values
    val_labels_ex3 = data_ex3.iloc[val_subjects_ex3]['y'].values
    test_labels_ex3 = data_ex3.iloc[test_subjects_ex3]['y'].values
    train_labels_ex3 = one_hot(train_labels_ex3,2)
    val_labels_ex3 = one_hot(val_labels_ex3,2)
    test_labels_ex3 = one_hot(test_labels_ex3,2)
    return train_data_ex3, val_data_ex3, test_data_ex3, train_labels_ex3, val_labels_ex3, test_labels_ex3

def load_and_preprocess_mnist():
    def download_if_not_exists(url, local_filename):
        if not os.path.exists(local_filename):
            print(f"Downloading {local_filename}...")
            urllib.request.urlretrieve(url, local_filename)
        else:
            print(f"{local_filename} already exists. Skipping download.")

    # Downloading and unzipping the data if not already available
    download_if_not_exists('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', './train-images-idx3-ubyte.gz')
    download_if_not_exists('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', './train-labels-idx1-ubyte.gz')
    download_if_not_exists('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', './t10k-images-idx3-ubyte.gz')
    download_if_not_exists('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', './t10k-labels-idx1-ubyte.gz')

    def unzip_gz(filename):
        if not os.path.exists(filename):  # Only unzip if the file is not already unzipped
            with gzip.GzipFile('./' + filename + '.gz') as gz_file:
                with open('./' + filename, 'wb') as out_file:
                    out_file.write(gz_file.read())
        else:
            print(f"{filename} already unzipped. Skipping.")

    unzip_gz('train-images-idx3-ubyte')
    unzip_gz('train-labels-idx1-ubyte')
    unzip_gz('t10k-images-idx3-ubyte')
    unzip_gz('t10k-labels-idx1-ubyte')

    mndata = MNIST('./')

    x_ex4_train, y_ex4_train = mndata.load_training()
    x_ex4_test, y_ex4_test = mndata.load_testing()

    # Preprocessing the data
    x_ex4_train = np.array(x_ex4_train) / 255. - 0.5
    y_ex4_train = one_hot(np.expand_dims(np.array(y_ex4_train), axis=1), 10)
    x_ex4_test = np.array(x_ex4_test) / 255. - 0.5
    y_ex4_test = one_hot(np.expand_dims(np.array(y_ex4_test), axis=1), 10)

    # Randomly selecting a validation set
    np.random.seed(1)
    shuffled_indexes = np.arange(x_ex4_train.shape[0])
    np.random.shuffle(shuffled_indexes)
    x_ex4_val = x_ex4_train[shuffled_indexes, :][:5000, :]
    y_ex4_val = y_ex4_train[shuffled_indexes, :][:5000, :]

    # Reducing the size of the training dataset to make training faster
    x_ex4_train = x_ex4_train[shuffled_indexes, :][5000:7000, :]
    y_ex4_train = y_ex4_train[shuffled_indexes, :][5000:7000, :]
    
    return x_ex4_train, x_ex4_val, x_ex4_test, y_ex4_train, y_ex4_val, y_ex4_test

def load_and_preprocess_mnist_():
    #downloading and unzipping the data
    urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', './train-images-idx3-ubyte.gz')
    urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', './train-labels-idx1-ubyte.gz')
    urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', './t10k-images-idx3-ubyte.gz')
    urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', './t10k-labels-idx1-ubyte.gz')

    def unzip_gz(filename):
        with gzip.GzipFile('./' + filename + '.gz') as gz_file:
            with open('./' + filename , 'wb') as out_file:
                out_file.write(gz_file.read())

    unzip_gz('train-images-idx3-ubyte')
    unzip_gz('train-labels-idx1-ubyte')
    unzip_gz('t10k-images-idx3-ubyte')
    unzip_gz('t10k-labels-idx1-ubyte')

    mndata = MNIST('./')

    x_ex4_train, y_ex4_train = mndata.load_training()
    x_ex4_test, y_ex4_test  = mndata.load_testing()

    #preprocessing the data
    x_ex4_train = np.array(x_ex4_train)/255.-0.5
    y_ex4_train = one_hot(np.expand_dims(np.array(y_ex4_train), axis = 1), 10)
    x_ex4_test = np.array(x_ex4_test)/255.-0.5
    y_ex4_test = one_hot(np.expand_dims(np.array(y_ex4_test), axis = 1), 10)

    #randomly selecting a validation set
    np.random.seed(1)
    shuffled_indexes = (np.arange(x_ex4_train.shape[0]))
    np.random.shuffle(shuffled_indexes)
    x_ex4_val = x_ex4_train[shuffled_indexes, :][:5000, :]
    y_ex4_val = y_ex4_train[shuffled_indexes, :][:5000, :]

    #reducing the size of the training dataset to make training faster
    # and to accentuate the overfitting of the model
    x_ex4_train = x_ex4_train[shuffled_indexes, :][5000:7000, :]
    y_ex4_train = y_ex4_train[shuffled_indexes, :][5000:7000, :]
    
    return x_ex4_train, x_ex4_val, x_ex4_test, y_ex4_train, y_ex4_val, y_ex4_test

###########################################################################################
###########################################################################################
###########################################################################################
##### ************************************************************************* ######

##########################################################################################
##### Validation for Ex. 1.1 #####
def validate_ex11(third_degree_polynomial):
    coeffs_ex11 = np.expand_dims(np.array([5.5,-2.4,0.1,3.3]), axis = 1)
    x = np.expand_dims(np.arange(-1,1,0.1), axis = 1)
    predicted_values = third_degree_polynomial(x, coeffs_ex11)
    
    expected_values_ex11 = np.expand_dims(np.asarray(\
                    [4.7,5.3353,5.7944,6.0971,6.2632,6.3125,6.2648,6.1399,5.9576,5.7377,
                      5.5,5.2643,5.0504,4.8781,4.7672,4.7375,4.8088,5.0009,5.3336,5.8267]), axis=1)

    if np.allclose(predicted_values, expected_values_ex11):
        print('Your third_degree_polynomial function seems to be returning the expected values')
    else:
        print('WARNING: Your third_degree_polynomial function is not returning the expected values.')
##########################################################################################

##########################################################################################
##### Validation for Ex. 1.2 #####
def validate_ex12(fit_func, x, y):
    degree_choice = 4
    weights_ex1_train = fit_func(x, y, 4)

    expected_coeffs_ex12 = np.expand_dims(np.array(\
                            [3.55626657,-4.85071512,13.58932204,-9.0132505,1.75782422]),axis=1)

    if np.allclose(weights_ex1_train, expected_coeffs_ex12):
        print('Your fitting function seems to be returning the expected values')
    else:
        print('WARNING: Your fitting function is not returning the expected values. You should review your code.')
##########################################################################################

##########################################################################################
##### Validation for Ex. 1.5 #####
def validate_ex15(mse, y_val, y_train):
    mse_testing_function = mse(y_val[:20,:],y_train)
    expected_error_ex15 = np.array([0.23942682821699499])

    if np.allclose(np.array([mse_testing_function]), expected_error_ex15):
        print('Your error function seems to be returning the expected values')
    else:
        print('WARNING: Your error function is not returning the expected values. You should review your code.')
##########################################################################################

##########################################################################################
##### Validation for Ex. 2.2 #####
def validate_ex22(two_layer_network_forward, initialize_parameters_ex2, x):
    predicted_outputs_testing_function = two_layer_network_forward(x, initialize_parameters_ex2(1, 20, 1))
    expected_output_ex21 = np.expand_dims(np.array( \
                        [0.30413895,0.28105381,0.21761113,0.236233,0.27359281,0.2995363,0.32282194,0.33807287,
                         0.35145301,0.36860323,0.38583117,0.40305912,0.42173721,0.44258177,0.46342632,
                         0.48427088,0.50511543,0.52595998,0.54680454,0.56764909]), 1)
    if np.allclose(predicted_outputs_testing_function, expected_output_ex21):    
        print('Your forward function seems to be returning the expected values')
    else:
        print('WARNING: Your forward function is not returning the expected values. You should review your code.')
##########################################################################################

##########################################################################################
##### Validation for Ex. 2.3 #####
# to verify that the analytical gradient computation is being done correctly,
# we provide this function that computes the numerical gradient of a network 
# with computations defined by forward_function, parameters defined by the 
# dictionary parameters
def get_numeric_gradient(forward_function, loss_function, inputs, targets, parameters):
    numeric_gradients = {}
    h=0.00001
    for weight_tensor_name in parameters.keys():
        copy_parameters = copy.deepcopy(parameters)
        this_weight = parameters[weight_tensor_name]
        grad = np.zeros_like(this_weight)
        for ix, element in np.ndenumerate(this_weight):
            this_weight_plus = copy.copy(this_weight)
            this_weight_plus[ix] += h
            this_weight_minus = copy.copy(this_weight)
            this_weight_minus[ix] -= h
            copy_parameters[weight_tensor_name] = this_weight_plus
            out_plus = forward_function(inputs, copy_parameters)
            loss_plus = loss_function(out_plus, targets)
            copy_parameters[weight_tensor_name] = this_weight_minus
            out_minus = forward_function(inputs, copy_parameters)
            loss_minus = loss_function(out_minus, targets)
            grad[ix] = (loss_plus - loss_minus) / (2 * h)
        numeric_gradients[weight_tensor_name] = grad
    return numeric_gradients

#this function calculated both numerical and analytical gradients and compare them to see
# if they are almost exactly the same
def test_gradient(forward_function, backward_function, loss_function, inputs, targets, parameters):
    numeric_gradient = get_numeric_gradient(forward_function, loss_function, inputs, targets, parameters)
    analytic_gradient = backward_function(inputs, parameters, targets)
    error_found = False
    for weight_tensor_name in parameters.keys():
        
        relative_error = np.mean(np.abs((numeric_gradient[weight_tensor_name] - analytic_gradient[weight_tensor_name])/(numeric_gradient[weight_tensor_name] + analytic_gradient[weight_tensor_name] + 1e-15)))
        if relative_error > 0.001:
            print(numeric_gradient[weight_tensor_name])
            print(analytic_gradient[weight_tensor_name])
            error_found = True
            break
    if error_found:
        print('WARNING: Analytical gradient for parameter ' + weight_tensor_name + ' is not the same as the numerical one.')
    else:
        print('Analytical and numerical gradients are the same for all parameters')
##########################################################################################

##########################################################################################
##### Validation for Ex. 3.1 #####
def validate_ex31(ce_loss,softmax, two_layer_network_softmax_forward, two_layer_network_softmax_ce_backward,\
             train_data, train_labels, initialize_parameters):
    np.random.seed(1)
    ce_loss_testing_function = ce_loss(softmax(np.random.normal(0, 0.5, [12, 13]) + 3*one_hot(np.random.randint(13, size = 12),
                                       13), axis = 1), one_hot(np.random.randint(13, size = 12), 13))
    if np.allclose(np.array(ce_loss_testing_function), np.array([3.1249936488336])):    
        print('Your loss function seems to be returning the expected values')
    else:
        print('Your loss function is not returning the expected values. You should review your code.')
    
    test_gradient(two_layer_network_softmax_forward, two_layer_network_softmax_ce_backward, ce_loss, train_data, 
                      train_labels, initialize_parameters)
##########################################################################################
