#!/usr/bin/env python
# EXERCISE 3

import os

os.system('pip3 install matplotlib')
os.system('pip3 install numpy')
os.system('pip3 install pandas')
os.system('pip3 install python-mnist')

import urllib.request
import pandas
import numpy as np
import matplotlib.pyplot as plt
import copy
from utils import *
from assignment1_ex2 import initialize_parameters_ex2, run_batch_sgd, two_layer_network_forward, two_layer_network_backward
#test_gradient, preprocess_medical_data, load_and_preprocess_mnist

#needed to plot plots with matplotlib in OSX

#set numpy to raise exceptions when encountering numerical errors
np.seterr(all='raise')

#this function is used to convert from integer encoding of labels to one hot encoding
# labels is an 1-D array with the integer labels from 0 to n_labels. 
def one_hot(labels, n_labels):
    return np.squeeze(np.eye(n_labels)[labels.reshape(-1)])

#Does the transpose of the last two axes of a tensor
def T(input_tensor):
    return np.swapaxes(input_tensor, -1, -2)


x_ex3_train, x_ex3_val, x_ex3_test, y_ex3_train, y_ex3_val, y_ex3_test = load_and_preprocess_mnist()

#sanity check to see that data is as it is supposed to be
#plt.imshow(x_ex3_train[1000,:].reshape(28,28), cmap = 'Greys')
plt.imsave('sample_image.png', x_ex3_train[1000, :].reshape(28, 28), cmap='Greys')

print(y_ex3_train[1000])

# ---------------- exercise 3.1 ----------------
def l2_regularization_backward(inputs, parameters, gt):
    gradients = {}
    for parameter_name in parameters.keys():
        if 'weights' in parameter_name:
            # complete the equation to calculate the l2 regularization loss gradient for weights
            ##your code starts here
            gradients[parameter_name] = parameters[parameter_name]
            ##your code ends here
        elif 'bias' in parameter_name:
            # complete the equation to calculate the l2 regularization loss gradient for bias.
            # Remember, the L2 regularization loss for bias is 0.
            gradients[parameter_name] = 0 * parameters[parameter_name]
            ##your code starts here
            
            ##your code ends here
    return gradients

#a softmax calculation with numerical stability tricks
def softmax(logits, axis):
    # subtracting the maximum logit from all logits for each example and prevents overflow 
    # of the exponential function of the logits and does not change results of the softmax
    # because of properties of division of exponentials
    stabilizing_logits = logits - np.expand_dims(np.max(logits, axis = axis), axis = axis)
    
    # clipping all logits to a minimum of -10 prevents underflow of the exponentials and 
    # only changes the result of the softmax minimally, since we know that one logit has value 0
    # and exp^0>>exp(-10)
    stabilizing_logits = np.clip(stabilizing_logits, -10, None)
    
    #using the softmax classic equation, but with the modified logits to prevent numerical errors
    return np.exp(stabilizing_logits)/np.expand_dims(np.sum(np.exp(stabilizing_logits), axis = axis), axis = axis)

# a forward function combined the two-layer network and the softmax
def two_layer_network_softmax_forward(inputs, parameters):
    logits = two_layer_network_forward(inputs, parameters)
    return softmax(logits, axis = 1)

# a forward function combined the two-layer network and the softmax
def softmax_plus_ce_loss_backward(predicted, gt):
    #the derivative of the output of softmax function followed by a cross-entropy loss
    # with respect to the input is a beautifully simple equation equals to the softmax
    # of the inputs minus the one-hot encoded groundtruth
    return (softmax(predicted, axis = 1) - gt)/predicted.shape[0]

#the calculation of the gradient for the classification network
def two_layer_network_softmax_ce_backward(inputs, parameters, gt):
    return two_layer_network_backward(inputs, parameters, gt, softmax_plus_ce_loss_backward)

# a function to get how many logits predicted the right class when compared to gt
def count_correct_predictions(logits, gt):
    predicted_labels = one_hot(np.argmax(logits, axis = 1), logits.shape[1])
    return np.sum(np.logical_and(predicted_labels,gt))

def two_layer_network_ce_and_l2_regularization_backward(inputs, parameters, gt, regularization_multiplier):
    gradients = {}
    gradients1 = two_layer_network_softmax_ce_backward(inputs, parameters, gt)
    gradients2 = l2_regularization_backward(inputs, parameters, gt)
    for parameter_name in parameters:
        gradients[parameter_name] = gradients1[parameter_name] + regularization_multiplier * gradients2[parameter_name]
    return gradients

# ** END exercise 3.1 **

# ---------------- exercise 3.2 ----------------
n_hidden_nodes = 200

##your code starts here



# Optimized hyperparameters
learning_rate = 0.15    # Slightly increased learning rate
batch_size = 32        # Smaller batch size for more precise gradient updates
n_epochs = 50          # More epochs for training
# More refined lambda values range, focusing on the best-performing region
lambda_values = [0, 0.001, 0.005, 0.01, 0.015, 0.02, 0.03]

# Initialize dictionary to store parameters for different lambda values
parameters_two_layer_classification_ex3_dic = {}

# Initialize dictionary to store validation accuracies
validation_accuracies = {}

# Train with each lambda value
for lambda_value in lambda_values:
    print(f"\nTraining model with weight decay λ = {lambda_value}")
    
    # Initialize parameters (input dimension, hidden nodes, output classes)
    parameters = initialize_parameters_ex2(x_ex3_train.shape[1], n_hidden_nodes, y_ex3_train.shape[1])
    
    # Create a function to run backpropagation with the specific lambda value
    def backward_function(inputs, parameters, gt):
        return two_layer_network_ce_and_l2_regularization_backward(inputs, parameters, gt, lambda_value)
    
    # Track training progress
    best_val_accuracy = 0
    patience_counter = 0
    max_patience = 5  # Increased early stopping patience
    
    # Implement learning rate decay
    current_lr = learning_rate
    
    # Train for multiple epochs
    for epoch in range(n_epochs):
        # Reduce learning rate every 10 epochs
        if epoch > 0 and epoch % 10 == 0:
            current_lr *= 0.8  # Learning rate decay factor
            print(f"Reducing learning rate to {current_lr:.6f}")
        
        # Shuffle the training data
        shuffled_indexes = np.arange(x_ex3_train.shape[0])
        np.random.shuffle(shuffled_indexes)
        shuffled_indexes = np.array_split(shuffled_indexes, x_ex3_train.shape[0]//batch_size)
        
        # Train in batches
        train_loss = 0
        for batch_i in range(len(shuffled_indexes)):
            batch = shuffled_indexes[batch_i]
            input_this_batch = x_ex3_train[batch, :]
            gt_this_batch = y_ex3_train[batch, :]
            
            # Update parameters using the current learning rate
            parameters = run_batch_sgd(backward_function, parameters, current_lr, input_this_batch, gt_this_batch)
        
        # Evaluate model on validation set after each epoch
        val_predictions = two_layer_network_softmax_forward(x_ex3_val, parameters)
        correct = count_correct_predictions(val_predictions, y_ex3_val)
        current_val_accuracy = correct / y_ex3_val.shape[0] * 100
        
        # Print progress
        print(f"Epoch {epoch+1}/{n_epochs}, Validation accuracy: {current_val_accuracy:.2f}%")
        
        # Implement early stopping
        if current_val_accuracy > best_val_accuracy:
            best_val_accuracy = current_val_accuracy
            best_parameters = parameters.copy()  # Save the best parameters
            patience_counter = 0
            
            # If target accuracy is reached, stop training
            if best_val_accuracy >= 90:
                print(f"Target accuracy of {best_val_accuracy:.2f}% reached, stopping training")
                break
        else:
            patience_counter += 1
            
        # Early stopping check
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Use the best parameters found during training
    parameters = best_parameters if 'best_parameters' in locals() else parameters
    
    # Final evaluation on validation set
    val_predictions = two_layer_network_softmax_forward(x_ex3_val, parameters)
    correct = count_correct_predictions(val_predictions, y_ex3_val)
    val_accuracy = correct / y_ex3_val.shape[0] * 100
    
    # Record validation accuracy
    validation_accuracies[lambda_value] = val_accuracy
    
    # Save trained parameters
    parameters_two_layer_classification_ex3_dic[lambda_value] = parameters
    
    print(f"Final: λ = {lambda_value}, Validation accuracy: {val_accuracy:.2f}%")

# Find best lambda value (highest validation accuracy)
best_lambda = max(validation_accuracies, key=validation_accuracies.get)
best_accuracy = validation_accuracies[best_lambda]

print("\nWeight Decay Analysis Results:")
print(f"Best λ value: {best_lambda}, Validation accuracy: {best_accuracy:.2f}%")

# Display validation accuracies for all lambda values
print("\nValidation accuracies for all λ values:")
for lambda_value in sorted(lambda_values):
    print(f"λ = {lambda_value}: {validation_accuracies[lambda_value]:.2f}%")

# Analyze the effect of different lambda values
print("\nAnalysis of weight decay (λ) effect:")
if validation_accuracies[0] < best_accuracy:
    print(f"- Using appropriate L2 regularization (λ = {best_lambda}) improves the model's generalization ability.")
    print(f"- Comparing no regularization (λ = 0) with best regularization (λ = {best_lambda}): accuracy improved by {validation_accuracies[best_lambda] - validation_accuracies[0]:.2f}%")

large_lambdas = [l for l in lambda_values if l > best_lambda]
if large_lambdas and any(validation_accuracies[l] < validation_accuracies[best_lambda] for l in large_lambdas):
    print("- Excessively large λ values can lead to underfitting, reducing generalization performance.")
    worst_large_lambda = max(large_lambdas, key=lambda l: validation_accuracies[best_lambda] - validation_accuracies[l])
    print(f"- For example, λ = {worst_large_lambda} has accuracy {validation_accuracies[best_lambda] - validation_accuracies[worst_large_lambda]:.2f}% lower than the best λ value")

# Whether the target is reached
if best_accuracy >= 90:
    print("\n✓ Successfully reached the 90% validation accuracy target!")
else:
    print(f"\n⚠ Failed to reach 90% validation accuracy target. Best result is {best_accuracy:.2f}%, which is {90 - best_accuracy:.2f}% below the target")

# Plot validation accuracy comparison for different lambda values
plt.figure(figsize=(12, 7))
plt.plot(lambda_values, [validation_accuracies[l] for l in lambda_values], 'o-', linewidth=2, markersize=8)
plt.axhline(y=90, color='r', linestyle='--', label='90% accuracy target')
plt.axvline(x=best_lambda, color='g', linestyle='--', label=f'Best λ = {best_lambda}')
plt.title('Validation Accuracy Comparison for Different Weight Decay Coefficients', fontsize=14)
plt.xlabel('Weight Decay Coefficient (λ)', fontsize=12)
plt.ylabel('Validation Accuracy (%)', fontsize=12)
plt.grid(True)
plt.legend(fontsize=12)
plt.savefig('lambda_comparison_ex3.png')

# Detailed analysis of the best model
print("\nDetailed Analysis of the Best Model:")
print(f"- Number of hidden neurons: {n_hidden_nodes}")
print(f"- Best λ value: {best_lambda}")
print(f"- Validation accuracy: {best_accuracy:.2f}%")
print("- Influencing factors:")
print("  * Larger hidden layer helps increase model capacity")
print("  * Appropriate L2 regularization prevents overfitting")
print("  * Smaller batch size provides more precise gradient updates")
print("  * Learning rate decay helps the model converge to better solutions in later training stages")



##your code ends here


# ** END exercise 3.2 **

# ---------------- exercise 3.3 ----------------
shuffled_indexes = (np.arange(x_ex3_test.shape[0]))
shuffled_indexes = np.array_split(shuffled_indexes,x_ex3_test.shape[0]//batch_size )
corrects = 0
total = 0
for batch_i in range(len(shuffled_indexes)):
    batch = shuffled_indexes[batch_i]
    corrects += count_correct_predictions(two_layer_network_forward(x_ex3_test[batch], parameters_two_layer_classification_ex3_dic[0.001]), y_ex3_test[batch])
    total += len(batch)
print('Test accuracy = ' + str(corrects/float(total)*100) + '%')

# ** END exercise 3.3 **