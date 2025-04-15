#!/usr/bin/env python
# coding: utf-8
"""
Exercise 2: ChestXray14 Dataset

This script loads a subset of the ChestXray14 dataset, creates PyTorch datasets and dataloaders,
and implements two parts:
 - Exercise 2.1: Adopts an ImageNet-pretrained ResNet-18 (modified for 14 labels) for multi-label classification.
 - Exercise 2.2: Implements a custom CNN model for ChestXray14 with at most 500,000 learnable parameters.
Each part trains the model and evaluates it using the AUC metric.
"""

import os
import importlib.util

if importlib.util.find_spec("kaggle") is None:
    print(f'installing kaggle')
    os.system('pip install -q kaggle')
if importlib.util.find_spec("pydicom") is None:
    print(f'installing pydicom')
    os.system('pip install pydicom')
if importlib.util.find_spec("sklearn") is None:
    print(f'installing scikit-learn')
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

hostname = hostname = subprocess.getoutput('hostname')
print(f'host name:{hostname}')

##### Check Torch library requirement #####
my_torch_version = torch.__version__
minimum_torch_version = '1.7'
if version.parse(my_torch_version) < version.parse(minimum_torch_version):
    print('Warning!!! Your Torch version %s does NOT meet the minimum requirement!\
            Please update your Torch library\n' %my_torch_version)

# ------------------------------- Data preparationn ------------------------------- don't need to change this part
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
dataset_root = '/scratch/tmp/' if machine_being_used == 'cade' else '/scratch/general/nfs1/ece6545_spring2025/'
mnist_dataset_folder = os.path.join(dataset_root , 'assignment2_data', 'mnist')
xray14_dataset_folder = os.path.join(dataset_root ,  'assignment2_data', 'chestxray14')
pneumonia_dataset_folder = os.path.join(dataset_root ,  'assignment2_data', 'kaggle_pneumonia_subset')

images_list_file = 'image_names_chestxray14.csv'
label_file = 'Data_Entry_2017_v2020.csv'
data_folder = os.path.join(dataset_root, 'assignment2_data', 'chestxray14')
download_dataset = False

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

# ================================ Exercise 2 - ChestXray14 Dataset (Total of 65 points) ===============================

##### Report download status #####
def report_hook(count_so_far, block_size, total_size):
    current_percentage = (count_so_far * block_size * 100 // total_size)
    previous_percentage = ((count_so_far - 1) * block_size * 100 // total_size)
    if current_percentage != previous_percentage:
        sys.stdout.write('\r' + str((count_so_far * block_size * 100 // total_size)) \
                         + '% of download completed')
        sys.stdout.flush()

##### Download a subset of ChestXray14 dataset #####        
if xray14_dataset_folder != '/scratch/general/nfs1/ece6545_spring2025/assignment2_data/chestxray14':
    os.makedirs(xray14_dataset_folder, exist_ok=True)
    from urllib.request import urlretrieve
    destination_file = xray14_dataset_folder + '/images_4.tar.gz'
    link = 'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz'
    if not os.path.isfile(destination_file):
        urlretrieve(link, destination_file, reporthook = report_hook)

    destination_file = xray14_dataset_folder + '/images_1.tar.gz'
    link = 'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz'
    if not os.path.isfile(destination_file):
        urlretrieve(link, destination_file, reporthook = report_hook)        
    
##### Extract the downloaded file #####
if download_dataset: #xray14_dataset_folder != '/scratch/tmp/deep_learning_datasets_ECE_6960_013/chestxray14':
    destination_file = xray14_dataset_folder + '/images_4.tar.gz'
    tar = tarfile.open(destination_file, "r:gz")
    tar.extractall(path = xray14_dataset_folder)
    tar.close()
    destination_file = xray14_dataset_folder + '/images_1.tar.gz'
    tar = tarfile.open(destination_file, "r:gz")
    tar.extractall(path = xray14_dataset_folder)
    tar.close()

class Chestxray14Dataset(Dataset):
    ##### Initialize the class #####
    def __init__(self, path_dataset_folder, split = 'train'):
        ## Split parameter is used to specify which process the data is used for,
        ## and it can be 'train', 'val', and 'test'
        
        self.path_image_folder = path_dataset_folder + '/images'
        
        ## Get the filenames of all images in the dataset
        all_images_list = pd.read_csv(images_list_file)

        ## Read the labels file which needs to be placed in the same folder as this notebook
        label_data = pd.read_csv(label_file)
        
        ## Merge labels and images information
        examples_to_use = pd.merge(all_images_list, label_data)
        
        ## This is the name of all possible labels in this dataset.
        ## The corresponding label of each sample is an array of 14 elements in which the elements are ordered
        ## in the same way as "self.set_of_finding_labels" and the value of each element represents the 
        ## presence of that condition in the sample. For example, if "cardiomegaly" and "pneumonia" are the two 
        ## conditions presence a given sample, then the corresponding label of that sample is represented 
        ## by an array  - [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        self.set_of_finding_labels = ['Atelectasis', 'Cardiomegaly','Effusion',  'Infiltration', 'Mass',\
                                      'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', \
                                      'Emphysema', 'Fibrosis','Pleural_Thickening', 'Hernia' ]
        
        ## Read labels from the label file
        examples_to_use['Finding Labels'] = examples_to_use['Finding Labels'].str.split(pat = '|')
        examples_to_use['Finding Labels'] = examples_to_use['Finding Labels'].apply(list).\
                                            to_frame(name='Finding Labels')
        for finding_label in self.set_of_finding_labels:
            examples_to_use[finding_label] = examples_to_use.apply(\
                                            lambda x: int(finding_label in x['Finding Labels']), axis=1)
        
        ## Get the list of all patient ids present in the dataset and split into
        ## training, validation and testing by patient id, but not by list of examples
        patient_ids = pd.unique(examples_to_use['Patient ID'])
        patient_ids = pd.DataFrame(get_split(patient_ids, split), columns = ['Patient ID'])
        
        ## Filter the examples to only use the ones that have the chosen patient ids
        examples_to_use = pd.merge(patient_ids,examples_to_use)        
        
        examples_to_use = examples_to_use[['Image Index'] + self.set_of_finding_labels]
        self.image_list = examples_to_use['Image Index'].values
        self.targets = examples_to_use[self.set_of_finding_labels].values
        
        ## Define data augmentation transformations for the input images. In this exercise, we use the following
        ## transformations: square center cropping, resizing to 224x224 (to be similar as ImageNet dataset), 
        ## converting to tensor, normalizing per channel (i.e., R, G, and B) 
        ## with the average and standard deviation of images in the ImageNet dataset        
        self.set_of_transforms = transforms.Compose(
        [CropBiggestCenteredInscribedSquare(),
         transforms.Resize(224),
         transforms.ToTensor(), 
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        
    ##### Retrieve a sample with the corresponding index #####
    ## This function retrieve a sample from the dataset at the specified index 
    ## and returns an image and the corresponding label stored in Pytorch tensors     
    def __getitem__(self, index):
        curr_pil_image = Image.open(self.path_image_folder + '/' + self.image_list[index]).convert('RGB')
        image_to_return = self.set_of_transforms(curr_pil_image)
                
        return image_to_return, torch.FloatTensor(self.targets[index])
    
    ##### Get the length of the dataset #####
    def __len__(self):
        return len(self.image_list)
    
    ##### Access the name of conditions in the labels #####
    def get_labels_name(self):
        return self.set_of_finding_labels


# We will then use the class function above to create structures for each training, validation, and test set.

## Create datasets for this exercise
train_dataset_ex2 = Chestxray14Dataset(xray14_dataset_folder)
val_dataset_ex2 = Chestxray14Dataset(xray14_dataset_folder, split = 'val')
test_dataset_ex2 = Chestxray14Dataset(xray14_dataset_folder, split = 'test')


## Please contact the TA if any of the assertions fails
assert(len(train_dataset_ex2) == 8837)
assert(len(val_dataset_ex2) == 2924)
assert(len(test_dataset_ex2) == 3238)
assert(np.sum(train_dataset_ex2.targets)==5893)
assert(np.sum(train_dataset_ex2.targets[:,7])==404)
assert(np.sum(val_dataset_ex2.targets)==1810)

##### Helper function for display text below #####
def join_str_array_to_labels(str_array,labels):
    return ','.join(['\n{}: {}'.format(labels[index_element], str_array_element) 
                for index_element, str_array_element in enumerate(str_array)])

## Show the statistics of training set
frequencies = np.sum(train_dataset_ex2.targets, axis = 0)/len(train_dataset_ex2)
text_frequencies = ['{:.2f}%'.format(frequency*100) for frequency in frequencies]                    
print('Percentage of positive examples in each class in the training set: ')
print(join_str_array_to_labels(text_frequencies, train_dataset_ex2.get_labels_name()))

## Plot a sample from the training set
print('\n\nShowing one example from the dataset:')
plt.imshow(train_dataset_ex2[1][0].cpu().numpy()[0,:,:], cmap = 'gray')
print(join_str_array_to_labels(train_dataset_ex2[1][1],train_dataset_ex2.get_labels_name()))

##### Calculate AUC metric #####
## This function compute AUC from the given input arrays i.e., predicted value and ground truth arrays
def auroc(logits_predicted, target):
    fpr, tpr, _ = roc_curve(target, logits_predicted)
    return auc(fpr, tpr)

##### Compute AUC of a given dataset #####
## This function takes a model and Pytorch data loader as input. 
## The given model is used to predict the expected label for each sample in the Pytorch data loader. The 
## model output for each sample is an array with 14 elements corresponding with 14 conditions in the 
## ChestXray14 dataset. Then, the AUC is computed for each condition.
def get_score_model_chestxray_binary_model(model, data_loader):
    ## Toggle model to eval mode
    model.eval()
    
    ## Iterate through the dataset and perform inference for each sample.
    ## Store inference results and target labels for AUC computation 
    with torch.no_grad():
        
        logits_predicted = np.zeros([0, 14])
        targets = np.zeros([0, 14])
        ## Iterate through the dataset and perform inference for each sample.
        ## Store inference results and target labels for AUC computation  
        for image, target in data_loader:
            image = image.cuda()
            logit_predicted = model(image)
            logits_predicted = np.concatenate((logits_predicted, logit_predicted.cpu().detach().numpy())\
                                              , axis = 0)
            targets = np.concatenate((targets, target.cpu().detach().numpy()), axis = 0)
            
    ## Return a list of auc values in which each value corresponds to one of the 14 labels
    return [auroc(logits_predicted[:,i], targets[:,i]) for i in range(14)]


# --------------------------- Exercise 2.1 - Adopt an ImageNet Pretrained Model (25 points) ----------------------------

##### Set-up Pytorch dataloader #####
## Your code starts here (Ex 2.1.1) ###

# Set batch sizes
batch_size_train = 32
batch_size_val = 64
batch_size_test = 64

# Create data loaders
train_loader_ex2 = torch.utils.data.DataLoader(train_dataset_ex2, batch_size=batch_size_train, 
                                               shuffle=True, num_workers=8)
val_loader_ex2 = torch.utils.data.DataLoader(val_dataset_ex2, batch_size=batch_size_val, 
                                            shuffle=False, num_workers=8)
test_loader_ex2 = torch.utils.data.DataLoader(test_dataset_ex2, batch_size=batch_size_test, 
                                             shuffle=False, num_workers=8)

# Confirm data loader sizes
print(f"Training set size: {len(train_loader_ex2.dataset)} samples, {len(train_loader_ex2)} batches")
print(f"Validation set size: {len(val_loader_ex2.dataset)} samples, {len(val_loader_ex2)} batches")
print(f"Test set size: {len(test_loader_ex2.dataset)} samples, {len(test_loader_ex2)} batches")

### Your code ends here ###


##### Modify ResNet-18 #####
### Your code starts here (Ex 2.1.2) ###

class ModifiedResNet18(torch.nn.Module):
    def __init__(self, num_classes=14):
        super(ModifiedResNet18, self).__init__()
        # Load pretrained ResNet-18 model
        self.resnet = models.resnet18(pretrained=True)
        
        # Modify the final fully connected layer to output 14 classes
        in_features = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        x = self.resnet(x)
        # Since this is multi-label classification, we don't use softmax activation
        return x

# Initialize the modified ResNet-18 model
model_resnet18 = ModifiedResNet18().cuda()
print(f"Modified ResNet-18 model parameter count: {count_number_parameters(model_resnet18)}")

### Your code ends here ###

##### Define loss function #####
### Your code starts here (Ex 2.1.3) ###

# For multi-label classification problems, use Binary Cross Entropy Loss
criterion = torch.nn.BCEWithLogitsLoss()

# Choose optimizer
optimizer = torch.optim.Adam(model_resnet18.parameters(), lr=0.0001)

### Your code ends here ###

# Ex 2.1.4 is textual. Answer it in your report file.

##### Training Process #####
### Your code starts here (Ex 2.1.5) ###

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    # Initialize best model and best score
    best_model = None
    best_score = 0.0
    
    # Training loop
    for epoch in range(num_epochs):
        # Training mode
        model.train()
        running_loss = 0.0
        
        # Train for one epoch
        for images, labels in train_loader:
            # Move data to GPU
            images = images.cuda()
            labels = labels.cuda()
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Calculate training loss
        epoch_loss = running_loss / len(train_loader)
        
        # Evaluate model performance on validation set
        auc_scores = get_score_model_chestxray_binary_model(model, val_loader)
        mean_auc = np.mean(auc_scores)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Mean AUC: {mean_auc:.4f}')
        
        # Print AUC for each class
        for i, label_name in enumerate(val_dataset_ex2.get_labels_name()):
            print(f'  {label_name}: {auc_scores[i]:.4f}')
        
        # If current model is the best, save it
        if mean_auc > best_score:
            best_score = mean_auc
            best_model = copy.deepcopy(model)
            print(f'Found new best model, validation mean AUC: {best_score:.4f}')
    
    return best_model, best_score

# Train the model
print("Starting ResNet-18 model training...")
best_model_resnet18, best_score_resnet18 = train_model(
    model_resnet18, train_loader_ex2, val_loader_ex2, criterion, optimizer, num_epochs=5)
print(f'Training complete! Best validation mean AUC: {best_score_resnet18:.4f}')

### Your code ends here ###


##### Inference stage for ChestXray14 dataset #####
### Your code starts here (Ex 2.1.6) ###
# Evaluate best model performance on test set
test_auc_scores = get_score_model_chestxray_binary_model(best_model_resnet18, test_loader_ex2)
test_mean_auc = np.mean(test_auc_scores)

print(f'Test set mean AUC: {test_mean_auc:.4f}')

# Find classes where model performs best and worst
best_class_idx = np.argmax(test_auc_scores)
worst_class_idx = np.argmin(test_auc_scores)
class_names = test_dataset_ex2.get_labels_name()

print(f'Best performing class: {class_names[best_class_idx]} (AUC: {test_auc_scores[best_class_idx]:.4f})')
print(f'Worst performing class: {class_names[worst_class_idx]} (AUC: {test_auc_scores[worst_class_idx]:.4f})')

# List AUC for all classes
for i, label_name in enumerate(class_names):
    print(f'  {label_name}: {test_auc_scores[i]:.4f}')

### Your code ends here ###


# -------------------------------- Exercise 2.2 - Implement Your Own Model (40 points) ---------------------------------

##### Implement Your Own Model #####
### Your code starts here (Ex 2.2.1) ###

class CustomCNN(torch.nn.Module):
    def __init__(self, num_classes=14):
        super(CustomCNN, self).__init__()
        
        # First convolutional block with reduced channels and feature map size
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Second convolutional block
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Third convolutional block with residual connection
        self.conv3a = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )
        self.conv3b = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
        )
        self.relu3 = torch.nn.ReLU()
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Adaptive pooling to ensure fixed output size
        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers with reduced size
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(64 * 4 * 4, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # Forward pass through the network
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Residual connection
        residual = x
        x = self.conv3a(x)
        x = self.conv3b(x)
        x = x + residual
        x = self.relu3(x)
        x = self.pool3(x)
        
        # Apply adaptive pooling
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Pass through fully connected layers
        x = self.fc(x)
        
        return x

# Initialize first custom model
model_custom1 = CustomCNN().cuda()
print(f"Custom model 1 parameter count: {count_number_parameters(model_custom1)}")

# Define second custom model - lightweight version with Squeeze-and-Excitation blocks
class LightweightCNN(torch.nn.Module):
    def __init__(self, num_classes=14):
        super(LightweightCNN, self).__init__()
        
        # First convolutional block with minimal channels
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(24),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Second convolutional block with depthwise separable convolution
        self.depthwise = torch.nn.Sequential(
            torch.nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, groups=24),
            torch.nn.BatchNorm2d(24),
            torch.nn.ReLU()
        )
        self.pointwise = torch.nn.Sequential(
            torch.nn.Conv2d(24, 48, kernel_size=1),
            torch.nn.BatchNorm2d(48),
            torch.nn.ReLU()
        )
        
        # Squeeze and Excitation block
        self.se1 = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(48, 24, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(24, 48, kernel_size=1),
            torch.nn.Sigmoid()
        )
        
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third convolutional block
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        
        # Squeeze and Excitation block 2
        self.se2 = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(64, 32, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=1),
            torch.nn.Sigmoid()
        )
        
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fourth convolutional block - adding another layer for more capacity
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(96),
            torch.nn.ReLU()
        )
        
        # Squeeze and Excitation block 3
        self.se3 = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(96, 48, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(48, 96, kernel_size=1),
            torch.nn.Sigmoid()
        )
        
        # Adaptive pooling to fix dimension issues
        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(96 * 4 * 4, 192),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),  # Slightly reduced dropout
            torch.nn.Linear(192, num_classes)
        )
        
    def forward(self, x):
        # Forward pass
        x = self.conv1(x)
        
        # Depthwise separable convolution
        x = self.depthwise(x)
        x = self.pointwise(x)
        
        # Apply SE block
        se_weight = self.se1(x)
        x = x * se_weight
        x = self.pool2(x)
        
        # Third convolution block
        x = self.conv3(x)
        
        # Apply SE block 2
        se_weight = self.se2(x)
        x = x * se_weight
        x = self.pool3(x)
        
        # Fourth convolution block
        x = self.conv4(x)
        
        # Apply SE block 3
        se_weight = self.se3(x)
        x = x * se_weight
        
        # Final pooling
        x = self.adaptive_pool(x)
        
        # Flatten and FC layers
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Initialize second custom model
model_custom2 = LightweightCNN().cuda()
print(f"Custom model 2 parameter count: {count_number_parameters(model_custom2)}")

# Define third custom model - deeper version
class DeepCNN(torch.nn.Module):
    def __init__(self, num_classes=14):
        super(DeepCNN, self).__init__()
        
        # Feature extraction with fewer channels
        self.features = torch.nn.Sequential(
            # First block
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second block
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third block
            torch.nn.Conv2d(32, 48, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(48),
            torch.nn.ReLU(),
            torch.nn.Conv2d(48, 48, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(48),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth block
            torch.nn.Conv2d(48, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Adaptive pooling for consistent output size
        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((4, 4))
        
        # Classifier with reduced dimensions
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(64 * 4 * 4, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Pass through feature extraction layers
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        # Pass through classifier
        x = self.classifier(x)
        return x

# Initialize third custom model
model_custom3 = DeepCNN().cuda()
print(f"Custom model 3 parameter count: {count_number_parameters(model_custom3)}")

# Check if all models meet parameter requirement
models_to_check = [
    ("CustomCNN", CustomCNN()),
    ("LightweightCNN", LightweightCNN()),
    ("DeepCNN", DeepCNN())
]

for name, model in models_to_check:
    param_count = count_number_parameters(model)
    if param_count > 500000:
        print(f"WARNING: {name} has {param_count} parameters, exceeding the 500,000 limit!")
    else:
        print(f"{name} has {param_count} parameters, meets the requirement.")

# Data augmentation transforms for training
train_transform = transforms.Compose([
    CropBiggestCenteredInscribedSquare(),
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Learning rate scheduler
class WarmupCosineLRScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, initial_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
            
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

# Extended training function with more epochs and better scheduler
def train_model_extended(model, train_loader, val_loader, criterion, optimizer, initial_lr, num_epochs=8, patience=3):
    # Initialize best model and best score
    best_model = None
    best_score = 0.0
    no_improve_count = 0
    
    # Initialize scheduler with warmup
    scheduler = WarmupCosineLRScheduler(
        optimizer=optimizer,
        warmup_epochs=1,
        total_epochs=num_epochs,
        initial_lr=initial_lr
    )
    
    # Training loop
    for epoch in range(num_epochs):
        # Update learning rate using scheduler
        current_lr = scheduler.step(epoch)
        
        # Training mode
        model.train()
        running_loss = 0.0
        
        # Train for one epoch
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Progress indicator
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)} ({batch_idx/len(train_loader)*100:.1f}%)")
            
            # Move data to GPU
            images = images.cuda()
            labels = labels.cuda()
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
        
        # Calculate training loss
        epoch_loss = running_loss / len(train_loader)
        
        # Evaluate model performance on validation set
        print(f"Evaluating epoch {epoch+1}/{num_epochs} on validation set...")
        auc_scores = get_score_model_chestxray_binary_model(model, val_loader)
        mean_auc = np.mean(auc_scores)
        
        print(f'Epoch {epoch+1}/{num_epochs}, LR: {current_lr:.6f}, Loss: {epoch_loss:.4f}, Val Mean AUC: {mean_auc:.4f}')
        
        # Print AUC for each class
        for i, label_name in enumerate(val_dataset_ex2.get_labels_name()):
            print(f'  {label_name}: {auc_scores[i]:.4f}')
        
        # Early stopping logic
        if mean_auc > best_score:
            best_score = mean_auc
            best_model = copy.deepcopy(model)
            no_improve_count = 0
            print(f'Found new best model, validation mean AUC: {best_score:.4f}')
            
            # If we've reached the target AUC, we can stop early
            if best_score >= 0.67:
                print(f'Reached target AUC of 0.67, continuing training for potential further improvement')
        else:
            no_improve_count += 1
            print(f'No improvement for {no_improve_count} epochs. Best AUC so far: {best_score:.4f}')
            
            if no_improve_count >= patience and best_score >= 0.67:
                print(f'Early stopping as no improvement for {patience} epochs and already reached target 0.67 AUC')
                break
    
    return best_model, best_score

# Train custom models with hyperparameter tuning
def train_custom_models():
    # Define different learning rates to try
    learning_rates = [0.0008, 0.0005]  # Slightly lower learning rates
    
    # Define different batch sizes to try
    batch_sizes = [16, 24]  # Adjusted batch sizes
    
    # Keep track of best model, score and configuration
    best_model = None
    best_score = 0.0
    best_config = None
    
    # Try different model and hyperparameter combinations
    models = [model_custom1, model_custom2, model_custom3]
    model_names = ["CustomCNN", "LightweightCNN", "DeepCNN"]
    
    # Training loop with progress reporting
    print(f"Starting grid search with {len(models)} models, {len(learning_rates)} learning rates, {len(batch_sizes)} batch sizes")
    start_time = time.time()
    
    total_combinations = len(models) * len(learning_rates) * len(batch_sizes)
    combination_count = 0
    
    # First try the promising LightweightCNN model which almost reached the target
    model_idx = 1  # LightweightCNN index
    for lr_idx, lr in enumerate(learning_rates):
        for bs_idx, bs in enumerate(batch_sizes):
            # Update progress counter
            combination_count += 1
            progress_pct = combination_count / total_combinations * 100
            
            elapsed = time.time() - start_time
            est_remaining = elapsed / combination_count * (total_combinations - combination_count) if combination_count > 0 else 0
            
            print(f"\n[{combination_count}/{total_combinations}] ({progress_pct:.1f}%) - Est. remaining: {est_remaining/60:.1f} min")
            print(f"Training model: {model_names[model_idx]}, Learning rate: {lr}, Batch size: {bs}")
            
            # Initialize model
            current_model = LightweightCNN().cuda()
            
            # Set optimizer with weight decay for regularization
            optimizer = torch.optim.AdamW(current_model.parameters(), lr=lr, weight_decay=5e-5)
            
            # Create new data loader with augmentation
            train_dataset_augmented = Chestxray14Dataset(xray14_dataset_folder)
            train_dataset_augmented.set_of_transforms = train_transform  # Apply data augmentation
            
            train_loader = torch.utils.data.DataLoader(train_dataset_augmented, batch_size=bs, 
                                                      shuffle=True, num_workers=4)  # Reduced workers
            
            # Train model with extended function and more epochs
            trained_model, score = train_model_extended(current_model, train_loader, val_loader_ex2, 
                                                      criterion, optimizer, initial_lr=lr, num_epochs=8, patience=3)
            
            # If this configuration achieves target AUC, save it and break the loop
            if score >= 0.67:
                best_score = score
                best_model = copy.deepcopy(trained_model)
                best_config = {
                    "model": model_names[model_idx],
                    "learning_rate": lr,
                    "batch_size": bs
                }
                print(f"Found model that exceeds target 0.67 AUC! Config: {best_config}, Val mean AUC: {best_score:.4f}")
                return best_model, best_score, best_config
            
            # Update best model if current is better
            if score > best_score:
                best_score = score
                best_model = copy.deepcopy(trained_model)
                best_config = {
                    "model": model_names[model_idx],
                    "learning_rate": lr,
                    "batch_size": bs
                }
                print(f"Found new best model! Config: {best_config}, Val mean AUC: {best_score:.4f}")
    
    # If LightweightCNN didn't reach the target, try the other models
    for model_idx in [0, 2]:  # CustomCNN and DeepCNN
        for lr_idx, lr in enumerate(learning_rates):
            for bs_idx, bs in enumerate(batch_sizes):
                # Update progress counter
                combination_count += 1
                progress_pct = combination_count / total_combinations * 100
                
                elapsed = time.time() - start_time
                est_remaining = elapsed / combination_count * (total_combinations - combination_count) if combination_count > 0 else 0
                
                print(f"\n[{combination_count}/{total_combinations}] ({progress_pct:.1f}%) - Est. remaining: {est_remaining/60:.1f} min")
                print(f"Training model: {model_names[model_idx]}, Learning rate: {lr}, Batch size: {bs}")
                
                # Reinitialize model
                if model_idx == 0:
                    current_model = CustomCNN().cuda()
                else:
                    current_model = DeepCNN().cuda()
                
                # Set optimizer with weight decay for regularization
                optimizer = torch.optim.AdamW(current_model.parameters(), lr=lr, weight_decay=5e-5)
                
                # Create new data loader with augmentation
                train_dataset_augmented = Chestxray14Dataset(xray14_dataset_folder)
                train_dataset_augmented.set_of_transforms = train_transform  # Apply data augmentation
                
                train_loader = torch.utils.data.DataLoader(train_dataset_augmented, batch_size=bs, 
                                                         shuffle=True, num_workers=4)  # Reduced workers
                
                # Train model with extended function and more epochs
                trained_model, score = train_model_extended(current_model, train_loader, val_loader_ex2, 
                                                         criterion, optimizer, initial_lr=lr, num_epochs=8, patience=3)
                
                # If this configuration achieves target AUC, save it
                if score >= 0.67:
                    best_score = score
                    best_model = copy.deepcopy(trained_model)
                    best_config = {
                        "model": model_names[model_idx],
                        "learning_rate": lr,
                        "batch_size": bs
                    }
                    print(f"Found model that exceeds target 0.67 AUC! Config: {best_config}, Val mean AUC: {best_score:.4f}")
                    return best_model, best_score, best_config
                
                # Update best model if current is better
                if score > best_score:
                    best_score = score
                    best_model = copy.deepcopy(trained_model)
                    best_config = {
                        "model": model_names[model_idx],
                        "learning_rate": lr,
                        "batch_size": bs
                    }
                    print(f"Found new best model! Config: {best_config}, Val mean AUC: {best_score:.4f}")
    
    total_time = time.time() - start_time
    print(f"Grid search completed in {total_time/60:.1f} minutes")
    return best_model, best_score, best_config

# Train custom models
print("Starting custom model training...")
best_custom_model, best_custom_score, best_config = train_custom_models()
print(f"Best custom model training complete! Val mean AUC: {best_custom_score:.4f}")
print(f"Best configuration: {best_config}")
    
### Your code ends here ###

##### Verify the number of learnable parameters requirement #####
## ***** Please change the parameter inside the "count_number_parameters to the name of the model you want to test *****
### Your code starts here  (Ex 2.2.2) ###

# Verify parameter count of best model
if best_config["model"] == "CustomCNN":
    parameter_model = CustomCNN().cuda()
elif best_config["model"] == "LightweightCNN":
    parameter_model = LightweightCNN().cuda()
else:
    parameter_model = DeepCNN().cuda()

# Check if parameter count meets requirements (must be under 500,000)
parameter_count = count_number_parameters(parameter_model)
if parameter_count > 500000:
    print(f'Warning! Your model exceeds the learnable parameters requirement!')
    print(f'Current parameter count: {parameter_count}')
else:
    print(f'Model parameter count: {parameter_count}, meets the requirement (< 500,000).')
    
### Your code ends here ###

# Let's test your best model on the test set!


##### Perform inference with your best model #####
### Your code starts here  (Ex 2.2.3) ### 

# Evaluate best custom model on test set
custom_test_auc_scores = get_score_model_chestxray_binary_model(best_custom_model, test_loader_ex2)
custom_test_mean_auc = np.mean(custom_test_auc_scores)

print(f'Custom model test set mean AUC: {custom_test_mean_auc:.4f}')

# Find classes where model performs best and worst
best_class_idx = np.argmax(custom_test_auc_scores)
worst_class_idx = np.argmin(custom_test_auc_scores)
class_names = test_dataset_ex2.get_labels_name()

print(f'Best performing class: {class_names[best_class_idx]} (AUC: {custom_test_auc_scores[best_class_idx]:.4f})')
print(f'Worst performing class: {class_names[worst_class_idx]} (AUC: {custom_test_auc_scores[worst_class_idx]:.4f})')

# List AUC for all classes
for i, label_name in enumerate(class_names):
    print(f'  {label_name}: {custom_test_auc_scores[i]:.4f}')

# Compare with ResNet-18 model if available
if 'test_mean_auc' in locals() and 'test_auc_scores' in locals():
    print("\nModel comparison:")
    print(f"ResNet-18 mean AUC: {test_mean_auc:.4f}")
    print(f"Custom model mean AUC: {custom_test_mean_auc:.4f}")
    print(f"Difference: {test_mean_auc - custom_test_mean_auc:.4f}")
    
    # Calculate per-class performance difference
    print("\nPer-class performance comparison:")
    for i, label_name in enumerate(class_names):
        resnet_score = test_auc_scores[i]
        custom_score = custom_test_auc_scores[i]
        diff = resnet_score - custom_score
        print(f"  {label_name}: ResNet: {resnet_score:.4f}, Custom: {custom_score:.4f}, Diff: {diff:.4f}")
else:
    print("\nNo ResNet-18 model results available for comparison.")

### Your code ends here ###