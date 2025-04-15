import os
import importlib.util
if importlib.util.find_spec("kaggle") is None:
    os.system('pip install -q kaggle')
if importlib.util.find_spec("pydicom") is None:
    os.system('pip install pydicom')
if importlib.util.find_spec("scikit-learn") is None:
    os.system('pip install scikit-learn')

import getpass
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys
from sklearn.metrics import roc_curve, auc
import copy
import torchvision.models as models
import tarfile
import time
from packaging import version
import subprocess

##### Check Torch library requirement #####
my_torch_version = torch.__version__
minimum_torch_version = '1.7'
if version.parse(my_torch_version) < version.parse(minimum_torch_version):
    print('Warning!!! Your Torch version %s does NOT meet the minimum requirement!\
            Please update your Torch library\n' %my_torch_version)

# ## + Create Data Folder:
##### Check what kind of system you are using #####
try:
    hostname = hostname = subprocess.getoutput('hostname')
    if 'lab' in hostname[0] and '.eng.utah.edu' in hostname[0]:
        IN_CADE = True
    else:
        IN_CADE = False
except:
    IN_CADE = False

## Define the folders where datasets will be
machine_being_used = 'cade' if IN_CADE else ('other')
pre_folder = '/scratch/tmp/' if machine_being_used == 'cade' else './'
mnist_dataset_folder = pre_folder + 'deep_learning_datasets_ECE_6960_013/mnist'
xray14_dataset_folder = pre_folder + 'deep_learning_datasets_ECE_6960_013/chestxray14'
pneumonia_dataset_folder = pre_folder + 'deep_learning_datasets_ECE_6960_013/kaggle_pneumonia'

## Create directory if they haven't existed yet 
if not os.path.exists(mnist_dataset_folder):
    os.makedirs(mnist_dataset_folder)    
if machine_being_used != 'cade' and not os.path.exists(mnist_dataset_folder+'/MNIST'):        
    os.makedirs(mnist_dataset_folder+'/MNIST')
if machine_being_used != 'cade' and not os.path.exists(mnist_dataset_folder+'/MNIST/raw'):
    os.makedirs(mnist_dataset_folder+'/MNIST/raw')
if not os.path.exists(xray14_dataset_folder):
    os.makedirs(xray14_dataset_folder)
if not os.path.exists(pneumonia_dataset_folder):
    os.makedirs(pneumonia_dataset_folder)


##### Request a GPU #####
## This function locates an available gpu for usage. In addition, this function reserves a specificed
## memory space exclusively for your account. The memory reservation prevents the decrement in computational
## speed when other users try to allocate memory on the same gpu in the shared systems, i.e., CADE machines. 
## Note: If you use your own system which has a GPU with less than 4GB of memory, remember to change the 
## specified mimimum memory.
def define_gpu_to_use(minimum_memory_mb = 3500):    
    thres_memory = 600 #
    gpu_to_use = None
    try: 
        os.environ['CUDA_VISIBLE_DEVICES']
        print('GPU already assigned before: ' + str(os.environ['CUDA_VISIBLE_DEVICES']))
        return
    except:
        pass
    
    torch.cuda.empty_cache()
    for i in range(16):
        free_memory = subprocess.getoutput(f'nvidia-smi --query-gpu=memory.free -i {i} --format=csv,nounits,noheader')
        if free_memory[0] == 'No devices were found':
            break
        free_memory = int(free_memory[0])
        
        if free_memory>minimum_memory_mb-thres_memory:
            gpu_to_use = i
            break
            
    if gpu_to_use is None:
        print('Could not find any GPU available with the required free memory of ' + str(minimum_memory_mb) \
              + 'MB. Please use a different system for this assignment.')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_to_use)
        print('Chosen GPU: ' + str(gpu_to_use))
        x = torch.rand((256,1024,minimum_memory_mb-thres_memory)).cuda()
        x = torch.rand((1,1)).cuda()        
        del x
        
## Request a gpu and reserve the memory space
define_gpu_to_use()


# ## + Define Utility Functions:

##### Preprocess Image #####
## This function is used to crop the largest 1:1 aspect ratio region of a given image.
## This is useful, especially for medical datasets, since many datasets have images
## with different aspect ratios and this is one way to standardize inputs' size.
class CropBiggestCenteredInscribedSquare(object):
    def __init__(self):
        pass

    def __call__(self, tensor):
        longer_side = min(tensor.size)
        horizontal_padding = (longer_side - tensor.size[0]) / 2
        vertical_padding = (longer_side - tensor.size[1]) / 2
        return tensor.crop(
            (
                -horizontal_padding,
                -vertical_padding,
                tensor.size[0] + horizontal_padding,
                tensor.size[1] + vertical_padding
            )
        )

    def __repr__(self):
        return self.__class__.__name__ + '()'


##### Split a dataset for training, validatation, and testing #####
## This function splits a given dataset into 3 subsets of 60%-20%-20% for train-val-test, respectively.
## This function is used internally in the dataset classes below.
def get_split(array_to_split, split):
    np.random.seed(0)
    np.random.shuffle(array_to_split)
    np.random.seed()
    if split == 'train':
        array_to_split = array_to_split[:int(len(array_to_split)*0.6)]
    elif split == 'val':
        array_to_split = array_to_split[int(len(array_to_split)*0.6):int(len(array_to_split)*0.8)]
    elif split == 'test':
        array_to_split = array_to_split[int(len(array_to_split)*0.8):]
    return array_to_split


##### Compute the number parameters (weights) #####
## This function computes the number of learnable parameters in a Pytorch model
def count_number_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================== Exercise 1 - MNIST Dataset and CNNs (Total of 35 points) ==============================

##### Load MNIST dataset to Pytorch Dataloader #####
## Dowloand MNIST dataset train set from Pytorch (60,000 images total)
mnist_training_data = torchvision.datasets.MNIST(mnist_dataset_folder, train = True, \
                                        transform=transforms.ToTensor(), \
                                        target_transform=None, \
                                        download= True)
print('A summary of MNIST dataset:\n')
print(mnist_training_data,'\n')
## Dowloand MNIST dataset test set from Pytorch (10,000 images total)
mnist_test_data = torchvision.datasets.MNIST(mnist_dataset_folder, train = False, \
                                        transform=transforms.ToTensor(), \
                                        target_transform=None, \
                                        download= True)
print(mnist_test_data)
## Randomly split training images into 80%/20% for training/validation process
train_data_ex1, val_data_ex1 = torch.utils.data.random_split(mnist_training_data, \
                                    [int(0.8*len(mnist_training_data)),\
                                     len(mnist_training_data)-int(0.8*len(mnist_training_data))], \
                                     generator=torch.Generator().manual_seed(1))

assert(len(mnist_test_data) == 10000)
assert(len(train_data_ex1)+len(val_data_ex1) == 60000)

## Load files to Pytorch dataloader for training, validation, and testing
train_loader_ex1 = torch.utils.data.DataLoader(train_data_ex1, batch_size=16, shuffle=True, num_workers=8)
val_loader_ex1 = torch.utils.data.DataLoader(val_data_ex1, batch_size=128, shuffle=True, num_workers=8)
test_loader_ex1 = torch.utils.data.DataLoader(mnist_test_data, batch_size=128, shuffle=False, num_workers=8)



##### Compute accuracy for MNIST dataset #####
def get_accuracy_mnist(model, data_loader):
    ## Toggle model to eval mode
    model.eval()
    
    ## Iterate through the dataset and perform inference for each sample.
    ## Store inference results and target labels for AUC computation 
    with torch.no_grad():
        #run through several batches, does inference for each and store inference results
        # and store both target labels and inferenced scores
        acc = 0.0
        for image, target in data_loader:
            image = image.cuda(); target = target.cuda()
            probs = model(image)
            preds = torch.argmax(probs, 1)
            acc += torch.count_nonzero(preds == target)
                
        return acc/len(data_loader.dataset)
    
# ---------------------------------------- Ex 1.1 - Implement a CNN (15 points) ----------------------------------------
### Your code starts here ###




class model_ex1(torch.nn.Module):
    def __init__(self):
        super(model_ex1, self).__init__()
        
        # Initial zero padding (2 pixels)
        self.pad = torch.nn.ZeroPad2d(2)
        
        # Convolution Layer 1
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, 
                                     padding=0, stride=1, bias=True)
        self.relu1 = torch.nn.ReLU()
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolution Layer 2
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, 
                                     padding=0, stride=1, bias=True)
        self.relu2 = torch.nn.ReLU()
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolution Layer 3
        self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, 
                                     padding=0, stride=1, bias=True)
        self.relu3 = torch.nn.ReLU()
        
        # Fully Connected Layer 1
        self.fc1 = torch.nn.Linear(in_features=120, out_features=84, bias=True)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.relu4 = torch.nn.ReLU()
        
        # Fully Connected Layer 2
        self.fc2 = torch.nn.Linear(in_features=84, out_features=10, bias=True)
    
    def forward(self, x):
        # Debug prints to track tensor shapes
        # print("Input shape:", x.shape)
        
        # Apply initial zero padding
        x = self.pad(x)
        # print("After padding shape:", x.shape)
        
        # Convolution Layer 1 + ReLU + MaxPool
        x = self.conv1(x)
        # print("After conv1 shape:", x.shape)
        x = self.relu1(x)
        x = self.maxpool1(x)
        # print("After maxpool1 shape:", x.shape)
        
        # Convolution Layer 2 + ReLU + MaxPool
        x = self.conv2(x)
        # print("After conv2 shape:", x.shape)
        x = self.relu2(x)
        x = self.maxpool2(x)
        # print("After maxpool2 shape:", x.shape)
        
        # Convolution Layer 3 + ReLU
        x = self.conv3(x)
        # print("After conv3 shape:", x.shape)
        x = self.relu3(x)
        
        # Flatten output for fully connected layers
        # print("Before flatten shape:", x.shape)
        x = x.reshape(x.size(0), -1)  # Use reshape instead of view for safety
        # print("After flatten shape:", x.shape)
        
        # Fully Connected Layer 1 + Dropout + ReLU
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu4(x)
        
        # Fully Connected Layer 2
        x = self.fc2(x)
        
        return x



### Your code ends here ###

# --------------------------------- Ex 1.2 - Implement the training process (20 points) --------------------------------
##### Training Process #####
### Your code starts here ###



import copy

print("Initializing model...")
# Initialize model
try:
    model = model_ex1().cuda()
    print(f"CNN model parameter count: {count_number_parameters(model)}")
    print("Model structure:")
    print(model)
except Exception as e:
    print(f"Error initializing model: {e}")
    raise

# Define loss function and optimizer
print("Setting up loss function and optimizer...")
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training parameters
num_epochs = 20
best_val_acc = 0
best_model_ex1 = None

print("Starting training loop...")
# Training loop
for epoch in range(num_epochs):
    # Training mode
    model.train()
    running_loss = 0.0
    
    # print(f"Epoch {epoch+1}/{num_epochs} - Training...")
    batch_count = 0
    for images, labels in train_loader_ex1:
        # Move data to GPU
        images = images.cuda()
        labels = labels.cuda()
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        try:
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Print progress every 100 batches
            batch_count += 1
            if batch_count % 100 == 0:
                print(f"  Batch {batch_count}, Current loss: {loss.item():.4f}")
        except Exception as e:
            print(f"Error during training batch: {e}")
            raise
    
    # Calculate training loss
    epoch_loss = running_loss / len(train_loader_ex1)
    
    # Calculate validation accuracy
    # print(f"Epoch {epoch+1}/{num_epochs} - Validation...")
    val_acc = get_accuracy_mnist(model, val_loader_ex1)
    
    # Print training progress
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Val Accuracy: {val_acc:.4f}')
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_ex1 = copy.deepcopy(model)
        print(f'Found new best model, validation accuracy: {best_val_acc:.4f}')
    
    # Early stopping - if validation accuracy exceeds 99%, stop training
    if val_acc > 0.99:
        print(f'Validation accuracy reached {val_acc:.4f}, stopping training')
        break

print(f'Training complete! Best validation accuracy: {best_val_acc:.4f}')



### Your code ends here ###


# evaluate the best trained model on the test set.
##### Inference stage for MNIST dataset #####
test_acc = get_accuracy_mnist(best_model_ex1, test_loader_ex1)
print('MNIST Test Accuracy: %.3f%%' %(test_acc*100))
