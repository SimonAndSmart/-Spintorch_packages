#!/usr/bin/env python
# coding: utf-8

# # SpinTorch (Local/Json version6.52) 
# >- require Creat_json code version: **v6.51.3** 
# >- supported saved model version: **v6.40** 
# >- supported result analysis code version: **v1.3** 
# 
# v6.32
# - fixed loss value representation during testing.
# 
# v6.33
# - changed 'weighting_cut_off' into 'dist_cut_off' so now we define the radius of the probe directlly.
# - added ability to make mutiple accuracy test
# - better way to chose equal probe measure method
# 
# v6.34
# - bug fix
# 
# v6.35
# - add max intensity classify for accuracy test back
# 
# v6.36
# - devided the one sum-intensity value into a set of 2ns period intensity sum
# - added a extra layer over regression training so it will pick the region gives the best acc on training data.
# - detailed acc result for each choice of time pixle saved into txt files
# 
# v6.37
# - set model.m_history to empty every epoch to save memory
# - analysis start from lasttest epoch in 'test_epochs' list
# 
# v6.38
# - added function to sum the time-intensity data in a weighted way 
# - added an itrating function to find the time window which gives the lowest loss
# 
# v6.40
# - re-desgine the input signal generator allow supoposition.
# - added a separate loss curve for the minium loss by selecting the time window.
# - fixed plot output error when epoch=1.
# 
# v6.41
# - add an extra test at the minimum training loss epoch number.
# - fixed signal generator with overtime problem
# 
# v6.42
# - add ability to separate data set to a few fractions across many 'epoch' so able to save model and show result in the same way (temporary method)
# 
# v6.43
# - include 'direct' modulate method
# - add a auto remove training model when code ends to save place
# 
# v6.44
# - heart-beat data treatment(not included here)
# 
# v6.45
# - added 'balanced_reduction_loss' loss function, focus on udertrained data mainly.
# 
# v6.46
# - add ability for individual duration for certain frequency in wave generator.
# - adapted the signal generation code with the new separating method
# 
# v6.47
# - separate the loss function used for training and used for determine the pixel weight
# 
# v6.48
# - add a new version of the 'balanced_reduction_loss' loss function witch use ratio insdead of absolute values
# 
# v6.49
# - change the load_data ability into directlly load the feature and label tensor
# - recude output plots number
# 
# v6.50
# - use the try function to temporarily avoid errors when plotting
# - use try function to avoid broken .pt model
# 
# v6.51
# - add an parameter 'square_after_sum' that allow the probe addup value without squaring the value before adding up if set to False.
# 
# v6.52
# - 'square_after_sum' now apply to testing as well when True.
# - add the support of a few PyTorch build-in loss function
# 
# Latter
# - fix ROC curve

# # Set all parameters

# In[5]:


import numpy as np
import argparse
import pickle

parser = argparse.ArgumentParser(description='Load parameters from a JSON file.')# Create an ArgumentParser object
parser.add_argument('--json_file', required=True, help='Path to JSON file')# Add a command-line argument for the JSON file path
args = parser.parse_args()# Parse the command-line arguments


# Load data
with open(args.json_file, 'rb') as f:
    params = pickle.load(f)

print(params)# Print the loaded parameters
locals().update(params)

# if loss_function != 'default':
#     loss_function = eval(params['loss_function'])()


# In[6]:


'''Define Environment'''
import os
import torch
import torch.optim as optim
from torch import tensor
import sklearn.metrics as sk_metric
from sklearn.linear_model import LogisticRegression
from math import ceil
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import pickle
import torch.nn as nn


if os.getcwd().endswith('Spintorch_packages'):
    print("You have already done this step, Skipped")
else:
    your_path = os.getcwd()  #you can change this if you want to save the model and plots at somewhere else
    print("pwd:",your_path)

if os.path.isdir(your_path+'/Spintorch_packages'):
    os.chdir(your_path+'/Spintorch_packages')
    os.system('git pull')

else:
    os.chdir(your_path)
    os.system("git clone https://github.com/SimonAndSmart/Spintorch_packages.git --depth 1 --branch main --single-branch --no-tags")
    print('Done!')
    os.chdir('Spintorch_packages')

import spintorch
from spintorch.utils import tic, toc, stat_cuda
from spintorch.plot import wave_integrated, wave_snapshot

import warnings
warnings.filterwarnings("ignore", message=".*Casting complex values to real.*")

if torch.cuda.is_available():
    dev = torch.device('cuda')  # 'cuda' or 'cpu'
else:
    dev = torch.device('cpu')
print('\nRunning on', dev)


# In[7]:


'''Define Data (generated)'''
from torch.utils.data import TensorDataset , DataLoader

def Clone_AddNoise(inputs_list, outputs_list, noise_std=0.05, clone=1, seed=None):
    if seed is not None:# Set the random seed for reproducibility
        torch.manual_seed(seed)

    cloned_inputs_list = inputs_list.repeat(clone, 1, 1)  # Clone the input list by the given factor
    cloned_outputs_list = outputs_list.repeat(clone)    # Repeat the elements in the output list by the given factor

    # Add noise to each tensor in the list
    noisy_inputs_list = [t + torch.randn_like(t) * noise_std for t in cloned_inputs_list]
    noisy_inputs_list  = torch.stack(noisy_inputs_list, dim=0)

    return noisy_inputs_list, cloned_outputs_list

##---

def generate_wave(f, time_duration, dt, timesteps):
    t = torch.linspace(0, (timesteps-1)*dt, timesteps)
    return torch.sin(2 * np.pi * f * t).unsqueeze(0).unsqueeze(-1)

def validate_inputs(frequencies, timesteps, dt, normalize_superposition):
    assert isinstance(frequencies, list), "frequencies must be a list"
    assert all(isinstance(f, (list, tuple)) for f in frequencies), "frequencies must be a list of lists or tuples"
    assert isinstance(timesteps, int), "timesteps must be an integer"
    assert dt > 0, "dt must be a positive number"
    assert normalize_superposition in [None, 'normal', 'sqrt'], "normalize_superposition must be None, 'normal' or 'sqrt'"

def generate_inputs(frequencies, default_time_duration, timesteps, dt, normalize_superposition=None):
    validate_inputs(frequencies, timesteps, dt, normalize_superposition)

    tensor_list = []
    for signal_freqs in frequencies:  
        signal_tensor_list = []
        total_timesteps_so_far = 0
        for i, freqs in enumerate(signal_freqs):  
            # Default duration for each frequency
            time_duration = default_time_duration

            # Check if the frequency is a tuple and extract values
            if isinstance(freqs, tuple):
                # Check if the second element is a string (indicating a custom duration)
                if isinstance(freqs[1], str):
                    f, time_duration_str = freqs  # Frequency with custom duration
                    time_duration = int(time_duration_str)  # Convert the duration string to an integer
                else:
                    f = freqs  # Tuple of frequencies for superposition
            else:
                f = freqs  # Single frequency

            # Calculate remaining timesteps
            if total_timesteps_so_far + time_duration > timesteps:
                remaining_timesteps = timesteps - total_timesteps_so_far
                timesteps_for_wave = remaining_timesteps
            else:
                timesteps_for_wave = time_duration

            # Generate the wave
            if isinstance(f, tuple):
                waves = sum(generate_wave(freq, time_duration, dt, timesteps_for_wave) for freq in f)
                if normalize_superposition == 'normal':
                    waves /= len(f)
                elif normalize_superposition == 'sqrt':
                    waves /= np.sqrt(len(f))
            else:
                waves = generate_wave(f, time_duration, dt, timesteps_for_wave)

            signal_tensor_list.append(waves)
            total_timesteps_so_far += timesteps_for_wave

            if total_timesteps_so_far >= timesteps:
                break

        # If total timesteps not reached, extend the last frequency
        if total_timesteps_so_far < timesteps:
            remaining_timesteps = timesteps - total_timesteps_so_far
            waves = generate_wave(f, time_duration, dt, remaining_timesteps)
            signal_tensor_list.append(waves)

        signal_tensor = torch.cat(signal_tensor_list, dim=1)
        tensor_list.append(signal_tensor)
    return tensor_list


# In[8]:


'''Define Data (from datasets)'''

from torch.utils.data import DataLoader, TensorDataset
from typing import Optional
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import pandas as pd

def prepare_data(features: torch.Tensor, labels: torch.Tensor, train_ratio: float, val_ratio: float, 
                 batch_size: int, distribute_data_in_multi_epochs: Optional[int] = None):
    
    # Split the data
    # First split the data into a training set and a temporary set (combination of validation and test sets)
    train_features, temp_features, train_labels, temp_labels = train_test_split(features, labels, train_size=train_ratio, stratify=labels, random_state=42)
    # Then split the temporary set into validation and test sets
    val_features, test_features, val_labels, test_labels = train_test_split(temp_features, temp_labels, train_size=val_ratio/(1-train_ratio), stratify=temp_labels, random_state=42)

    
    # Prepare data loaders
    train_loader = DataLoader(TensorDataset(train_features,train_labels), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(TensorDataset(val_features,val_labels), batch_size=1)
    test_loader = DataLoader(TensorDataset(test_features,test_labels), batch_size=1)
    
    # Split training data loader into smaller loaders if distribute_data_in_multi_epochs is specified
    if distribute_data_in_multi_epochs is not None and distribute_data_in_multi_epochs > 1:
        train_loader = split_data_loader(train_loader, distribute_data_in_multi_epochs)
    
    return train_loader, val_loader, test_loader

def split_data_loader(data_loader: DataLoader, num_splits: int):
    # Get the entire dataset from the data loader
    data = list(data_loader)

    # Concatenate all the batches to get the full dataset
    full_data = torch.cat([batch for batch, _ in data]), torch.cat([labels for _, labels in data])

    # Compute the size of each split
    split_size = full_data[0].size(0) // num_splits

    # Split the data and create new data loaders
    split_data_loaders = []
    for i in range(num_splits):
        start_index = i * split_size
        end_index = start_index + split_size if i < num_splits - 1 else full_data[0].size(0)
        split_data = TensorDataset(full_data[0][start_index:end_index], full_data[1][start_index:end_index])
        split_data_loaders.append(DataLoader(split_data, batch_size=data_loader.batch_size))
    
    return split_data_loaders


# In[9]:


def add_index_to_data(df): #need fix
    df['index'] = 0
    df.loc[df['label'] == 0, 'index'] = range(1, sum(df['label'] == 0) + 1)
    df.loc[df['label'] == 1, 'index'] = range(1, sum(df['label'] == 1) + 1)
    return df
def index_treatment(df,method='drop'):
    if method=='drop':
        df=df.drop(['index'], axis=1)
    else:
        print('NEED CODE HERE!')
    return df


# In[10]:


def direct_input_generation(df,timesteps):
    frequencies=df.loc[:, df.columns!='label'].values.tolist()
    features_list=[f_list+list(np.zeros(timesteps-len(f_list))) for f_list in frequencies]
    label_list = df.loc[:, df.columns=='label'].values.tolist()
    return features_list,label_list


# In[11]:


def separate_feature_label(df,time_duration,timesteps,dt,normalize_superposition):

    frequencies_ratio=df.loc[:, df.columns!='label'].values.tolist()
    frequencies=[[f_band_start+f_bandwidth*r for r in ratio]+[0] for ratio in frequencies_ratio]

    features_list=generate_inputs(frequencies, time_duration, timesteps, dt, normalize_superposition)
    label_list = df.loc[:, df.columns=='label'].values.tolist()
    return features_list,label_list


# In[12]:


# Correcting the function to load and process data
def load_and_process_data(filename, train_ratio=0.5, val_ratio=0.2, batch_size=2, distribute_data_in_multi_epochs=4,modulate_method='direct'):
    # Reload the csv file with correct column names
    column_names = [f'time{i}' for i in range(1, 61)] + ['label']
    df = pd.read_csv(filename, header=None, names=column_names)

    # Encode the labels
    df['label'] = df['label'].map({'R': 0, 'M': 1})
    # Add the original index
    df_i = add_index_to_data(df)
    df = index_treatment(df_i)
    # Shuffle the data
    df = shuffle(df, random_state=42)
    if modulate_method=='direct':
        feature_list,label_list = direct_input_generation(df,timesteps)
        features= torch.tensor(feature_list).unsqueeze(2).to(dev)
    elif modulate_method=='FM' :  
        feature_list,label_list = separate_feature_label(df,time_duration,timesteps,dt,normalize_superposition)
        features = torch.cat(feature_list, dim=0).to(dev)
    


    # Convert the data into tensor
    
    features = features * Bt
    labels = torch.LongTensor(label_list).squeeze(1).to(dev)

    # Prepare the data loaders
    train_loader, val_loader, test_loader = prepare_data(features, labels, train_ratio, val_ratio, batch_size, distribute_data_in_multi_epochs)

    return train_loader, val_loader, test_loader


# In[13]:


if dataset_path=='': #directlly generate data
    input_list = generate_inputs(frequencies, time_duration, timesteps, dt, normalize_superposition)

    # concatenate all signal tensors along the signal axis
    INPUTS_list = torch.cat(input_list, dim=0).to(dev)
    INPUTS_list = INPUTS_list * Bt
    OUTPUTS_list = torch.LongTensor(desired_out_list).to(dev)

    #----#

    # increase data size by adding noise
    INPUTS_list_c,OUTPUTS_list_c = Clone_AddNoise(INPUTS_list,OUTPUTS_list,noise_std=noise_std,clone=clone,seed=seed)
    # data loader not working at the time 
    train_loader_list, val_loader, test_loader = prepare_data(INPUTS_list_c, OUTPUTS_list_c, train_ratio, val_ratio, batch_size, distribute_data_in_multi_epochs)
    train_loader, _, _ = prepare_data(INPUTS_list_c, OUTPUTS_list_c, train_ratio, val_ratio, batch_size, distribute_data_in_multi_epochs=None)
    if distribute_data_in_multi_epochs != None and distribute_data_in_multi_epochs>1:
        print(f"len(train_loader_list)={len(train_loader_list)}, len(train_loader_list[0])={len(train_loader_list[0])}, len(val_loader)={len(val_loader)}, len(test_loader)={len(test_loader)}")
    else:
        print(f"len(train_loader)={len(train_loader_list)}, len(val_loader)={len(val_loader)}, len(test_loader)={len(test_loader)}")
 
    print('Defined data (generated), Done')


# In[14]:


def load_features_labels(path):
    dataset = torch.load(path)
    data_tensor = dataset['data']
    labels_tensor = dataset['labels']
    return data_tensor,labels_tensor


# In[15]:


if dataset_path !='' and 'sonar.csv' in dataset_path: #shit mountain here!
    train_loader_list, val_loader, test_loader = load_and_process_data(dataset_path, train_ratio=train_ratio, val_ratio=val_ratio, batch_size=batch_size, distribute_data_in_multi_epochs=distribute_data_in_multi_epochs,modulate_method=modulate_method)
    train_loader, _ , _ = load_and_process_data(dataset_path, train_ratio=train_ratio, val_ratio=val_ratio, batch_size=batch_size, distribute_data_in_multi_epochs=None,modulate_method=modulate_method)
    print(f"len(train_loader_list)={len(train_loader_list)}, len(train_loader_list[0])={len(train_loader_list[0])}, len(val_loader)={len(val_loader)}, len(test_loader)={len(test_loader)}")
    print(f'Defined data(from data set: [{dataset_path}] ), Done')
elif dataset_path !='' :
    features,labels=load_features_labels(dataset_path)
    features=features*Bt
    train_loader_list, val_loader, test_loader = prepare_data(features, labels, train_ratio, val_ratio, batch_size, distribute_data_in_multi_epochs)
    train_loader, _ , _ = prepare_data(features,labels, train_ratio, val_ratio, batch_size=batch_size, distribute_data_in_multi_epochs=None)
    print(f"len(train_loader_list)={len(train_loader_list)}, len(train_loader_list[0])={len(train_loader_list[0])}, len(val_loader)={len(val_loader)}, len(test_loader)={len(test_loader)}")
    print(f'Defined data(from data set: [{dataset_path}] ), Done')


# In[16]:


'''Define Probes'''

probes = []
for y_coordinate in range(damping_width+1,ny-damping_width-1):
    for x_coordinate in range(nx-damping_width-detection_width-2,nx-damping_width-2): # '-2' for detection range 2 steps before the damping area
        probes.append(spintorch.WaveProbe(x_coordinate,y_coordinate)) 


probe_center_list = []
for p in range(Np):
    probe_center_list.append([nx-damping_width-(detection_width+1)//2-2, int((ny-damping_width)*(p+1)/(Np+1))+damping_width//2])
print('Define probes, Done')


# In[17]:


'''Define Geometry'''
if geometry_type == 1 :
    # Ms_CoPt = 723e3 # saturation magnetization of the nanomagnets (A/m)
    # r0, dr, dm, z_off = 15, 4, 2, 10  # starting pos, period, magnet size, z distance
    # rx, ry = int((nx-2*r0)/dr), int((ny-2*r0)/dr+1)   #been defined in JSON files already
    
    rho = torch.zeros((rx,ry))  # Design parameter array
    geom = spintorch.WaveGeometryArray(rho, (nx, ny), (dx, dy, dz), Ms, B0, r0, dr, dm, z_off, rx, ry, Ms_CoPt)
elif geometry_type == 2 :
    B1 = 50e-3      # training field multiplier (T)
    geom = spintorch.WaveGeometryFreeForm((nx, ny), (dx, dy, dz), B0, B1, Ms)
else:
    geom = spintorch.WaveGeometryMs((nx, ny), (dx, dy, dz), Ms, B0)
######
src = spintorch.WaveLineSource(sig_start_bot,0,sig_start_top, ny-1, dim=2)


'''Define model'''
model = spintorch.MMSolver(geom, dt, [src], probes,damping_width=damping_width)

#------------------ input more necessary information into the model
model.probe_coordinate['p_center']=probe_center_list


if probe_reading_method_Train == 'equal_probes':
    model.probe_coordinate['Train'] = True
else:
    model.probe_coordinate['Train'] = False
    
if probe_reading_method_Test == 'equal_probes':
    model.probe_coordinate['Test'] = True
else:
    model.probe_coordinate['Test'] = False


all_labels = []
for _, labels in train_loader:
    all_labels.extend(labels.tolist())
model.all_labels=list(set(all_labels+list(range(Np))))
#------------------

model.to(dev)   # sending model to GPU/CPU
model.retain_history = True

'''Define optimizer and lossfunction'''
optimizer_class = getattr(optim, optimizer_name) # find the optimizer class in 'torch.optim' library
optimizer = optimizer_class(model.parameters(), lr=learn_rate, **optimizer_params)
print('Model define, Done')


# In[76]:


'''Define Loss function'''

def dist_to_probe_center(probe,probe_center,situation,label_ignore=False,separate_xy=False):
    '''Calculate the distance from current probe index to the given probe center'''
    x_pos=int(probe.x)
    y_pos=int(probe.y)

    x_dist=abs(probe_center[0]-x_pos)
    y_dist=abs(probe_center[1]-y_pos)
    if separate_xy:  #for the case if x and y distance is weighted differentlly
        return x_dist,y_dist
    elif not(label_ignore):
        dist=np.sqrt(x_dist**2+y_dist**2)
    else:
        if x_pos > probe_center[0]: #for ignored signals, only activate the pickups at the back
            dist=np.sqrt(x_dist**2+y_dist**2)
        else:
            if situation == 'Train':
                dist=0  #for ignored signals, dist=0 means zero weight
            else:
                dist=float('inf')
    return dist

def gaussian_weight_fn(dist,dist_cut_off,situation,label_ignore=False,equal_weight=False):
    'convert distance into weighting, result not normalized, parameters all defined above'
    
    if label_ignore == True and situation == 'Test':
        equal_weight=True

    if equal_weight==True:
        if dist < dist_cut_off[0]:
            return 1
        else:
            return 0

    if not(label_ignore):
        if dist < dist_cut_off[0]:
            c=weighting_intersect[0]
            sigma=weighting_sigma[0]
            
            result = c + np.exp(-(dist) ** 2 / (2 * sigma ** 2)) 
            return max(result,0)
        else:
            return 0
    else:
        if dist > dist_cut_off[1]:
            c=weighting_intersect[1]
            sigma=weighting_sigma[1]
            
            result = c - np.exp(-(dist) ** 2 / (2 * sigma ** 2)) 
            return max(result,0)
        else:
            return 0

def prob_weighting(probe,model,desired_output,dist_cut_off,situation,dist_fn=gaussian_weight_fn,equal_weight=False):
    'Returns the weighting of a probe with respect to the current label'
    if desired_output==-1:
        dist_list=[]
        for probe_center in model.probe_coordinate['p_center']:
            dist_list.append(dist_to_probe_center(probe,probe_center,situation=situation,label_ignore=True))
        distance = min(dist_list)
        return dist_fn(distance,dist_cut_off,situation,label_ignore=True,equal_weight=equal_weight)
    else:
        distance=dist_to_probe_center(probe,model.probe_coordinate['p_center'][desired_output],situation=situation)
        return dist_fn(distance,dist_cut_off,situation,equal_weight=equal_weight)


# In[19]:


def merge_vertual_probes(model, model_out_tensor, labels, situation='Train', square_after_sum=False):
    """
    The `merge_vertual_probes` function is designed to merge multiple channels of a tensor (`model_out_tensor`)
    based on weighted sums, where the weights are determined by the `prob_weighting()` function for each label.

    Parameters:
    - model: An object containing attributes like probes, mask, all_labels, and probe_coordinate.
    - model_out_tensor: A tensor with model output data which will undergo merging based on provided labels.
    - labels: A list of labels for merging.
    - situation: A string indicating the situation (default 'Train') to fetch the correct `equal_weight` value.
    - square_after_sum: A boolean flag. If `True` (default), the function squares the final tensor before returning.
                    If `False`, it squares the `model_out_tensor` first, computes the weighted sum, and then 
                    applies a square root to the result.

    Workflow:
    - Extract probes based on model's mask.
    - Filter out the '-1' label if not present in `labels`.
    - Initialize an output tensor `probes_out_tensor`.
    - For each label in `all_labels`, compute weights using the `prob_weighting()` function.
    - Depending on the `square_after_sum` flag:
      * If `True`: Compute the weighted sum directly.
      * If `False`: Square the `model_out_tensor` first, compute the weighted sum, and then apply a square root.
    - If `square_after_sum` is `True`, square the entire `probes_out_tensor`, otherwise return it as is.
    """
    
    # Extract probes based on model's mask
    if model.mask is not None:
        probes_masked = list(np.array(model.probes)[model.mask])
    else:
        probes_masked = list(model.probes)

    # Filter out the '-1' label if not in provided labels
    all_labels = model.all_labels
    if -1 not in labels:
        all_labels = [x for x in all_labels if x != -1]
    probes_out_tensor = torch.zeros(list(model_out_tensor.shape[:-1]) + [len(all_labels)])


    if situation == 'Train':
        dist_cut_off=dist_cut_off_train
    elif situation == 'Test':
        dist_cut_off=dist_cut_off_test
    else:
        raise ValueError(f'situation: [{situation}] not supported')

    equal_weight = model.probe_coordinate[situation]
    for label in all_labels:
        # Compute and normalize weights for each probe
        weight_tensor = torch.tensor([prob_weighting(prob, model, label,dist_cut_off,situation=situation, equal_weight=equal_weight) for prob in probes_masked])
        weight_tensor = weight_tensor / weight_tensor.sum() 
        weight_tensor = weight_tensor.view(1, 1, weight_tensor.size(0))

        # Modify the tensor based on the square_after_sum flag
        if not square_after_sum:
            squared_tensor = model_out_tensor.pow(2)
            probe_out_tensor = (squared_tensor * weight_tensor).sum(dim=-1, keepdim=True)
        else:
            probe_out_tensor = (model_out_tensor * weight_tensor).sum(dim=-1, keepdim=True)

        probes_out_tensor[:, :, label] = probe_out_tensor[:, :, 0]

    return probes_out_tensor.pow(2) if square_after_sum else probes_out_tensor


# In[20]:


def batch_loss(preds,labels,show=True):
    """
    Default loss function
    """
    if merge_ignore_with_last_probe==True and -1 in labels: # in this way the ignore method will also consider the last probe as a target direction
        preds= torch.cat([preds[:, :-2], (preds[:, -2] + preds[:, -1]).unsqueeze(1)], dim=-1)
    loss = []
    for batch_i in range(len(labels)):
        target_value = preds[batch_i][int(labels[batch_i])]
        target_loss = preds[batch_i].unsqueeze(0).sum(dim=1)/target_value-1
        loss.append(((target_loss.sum()/target_loss.size()[0]).log10()).view(1))
    loss=torch.cat(loss)
    [print("LOSS:",loss)if show else None]
    return loss 


# In[21]:


def balanced_reduction_loss(preds, labels, max_avg_ratio=0.5,exceed_ratio=0.02, show=True):
    """
    Calculates a custom loss for each batch that penalizes low target predictions and encourages high target predictions,
    but only up to the point where the target prediction exceeds both the mean and the max of the non-target predictions.
    A hyperparameter controls the relative importance of exceeding the mean and max of non-target predictions.
    
    Parameters:
        preds (Tensor): The predicted values.
        labels (Tensor): The actual labels.
        max_avg_ratio (float, optional): Determines the relative importance of exceeding the mean (1 - max_avg_ratio) 
        and max (max_avg_ratio) of non-target predictions. Default is 0.5.
        show (bool, optional): If true, print the losses for each batch and additional details. Default is True.
        
    Returns:
        loss (Tensor): The calculated loss for each batch.
    """
    loss = []
    preds_log = torch.log10(preds)

    for batch_i in range(len(labels)):
        target_index = int(labels[batch_i])
        target_value = preds_log[batch_i][target_index]
        non_target_values = torch.cat([preds_log[batch_i][:target_index], preds_log[batch_i][target_index+1:]])
        non_target_mean = non_target_values.mean()
        non_target_max = non_target_values.max()
        
        # Penalize the loss if the target value is lower than the mean or max of the non-target values
        mean_diff_loss = torch.maximum(non_target_mean*(1+exceed_ratio) - target_value, torch.zeros_like(target_value))
        max_diff_loss = torch.maximum(non_target_max*(1+exceed_ratio) - target_value, torch.zeros_like(target_value))

        # Encourage the target value to be higher, but with diminishing returns as it increases
        target_encouragement = 1 / target_value

        # Weigh the mean and max diff losses based on the max_avg_ratio parameter
        total_loss = max_avg_ratio * max_diff_loss + (1 - max_avg_ratio) * mean_diff_loss + target_encouragement
        loss.append(total_loss.view(1))
        if show:
            print(f"Batch {batch_i} - Mean Diff Loss: {mean_diff_loss.item()}, Max Diff Loss: {max_diff_loss.item()}, Target Encouragement: {target_encouragement.item()}, Total Loss: {total_loss.item()}, Target Value: {target_value.item()}, Non-Target Mean: {non_target_mean.item()}, Non-Target Max: {non_target_max.item()}")
    loss = torch.cat(loss)
    return loss


# In[22]:


import torch

def balanced_reduction_loss_ratio(preds, labels, max_avg_ratio=0.5,exceed_ratio=0.02, show=True):
    """
    Calculates a custom loss for each batch that encourages high target predictions and penalizes low non-target predictions.
    The predictions are first transformed using the logarithm.
    The loss is then computed as a weighted sum of the ratios of non-target mean and max to target.
    
    Parameters:
        preds (Tensor): The predicted values.
        labels (Tensor): The actual labels.
        max_avg_ratio (float, optional): Relative importance of the max and average non-target values. Default is 0.5.
        show (bool, optional): If true, print the losses for each batch. Default is True.
        
    Returns:
        loss (Tensor): The calculated loss for each batch.
    """
    loss = []
    for batch_i in range(len(labels)):
        preds_log = torch.log10(preds[batch_i])  # Apply log transformation to handle large values
        target_index = int(labels[batch_i])
        target_value = preds_log[target_index]
        non_target_values = torch.cat([preds_log[:target_index], preds_log[target_index+1:]])
        non_target_mean = non_target_values.mean()
        non_target_max = non_target_values.max()

        # Calculate the ratio of non-target mean and max to target
        mean_ratio = non_target_mean*(1+exceed_ratio) / target_value -1
        max_ratio = non_target_max*(1+exceed_ratio) / target_value -1

        # Compute the loss as a weighted sum of the mean and max ratios
        mean_diff_loss = torch.maximum(mean_ratio, torch.zeros_like(target_value))
        max_diff_loss = torch.maximum(max_ratio, torch.zeros_like(target_value))

        # Encourage the target value to be higher, but with diminishing returns as it increases
        target_encouragement = 1 / target_value

        # Weigh the mean and max diff losses based on the max_avg_ratio parameter
        total_loss = max_avg_ratio * max_diff_loss + (1 - max_avg_ratio) * mean_diff_loss + target_encouragement
        loss.append(total_loss.view(1))
        if show:
            print(f"Batch {batch_i} - Mean Diff Loss: {mean_diff_loss.item()}, Max Diff Loss: {max_diff_loss.item()}, Target Encouragement: {target_encouragement.item()}, Total Loss: {total_loss.item()}, Target Value: {target_value.item()}, Non-Target Mean: {non_target_mean.item()}, Non-Target Max: {non_target_max.item()}")
    loss = torch.cat(loss)
    return loss


# In[23]:


# def target_maximization_loss(preds, labels, target_scale=0.01, non_target_scale=0.000001, show=True):
#     """
#     Calculates a custom loss for each batch that encourages high target predictions and discourages high non-target predictions.
#     The target predictions are encouraged via an exponential function, controlled by the 'target_scale' hyperparameter.
#     Both the target and non-target losses are scaled to ensure neither dominates.
    
#     Parameters:
#         preds (Tensor): The predicted values.
#         labels (Tensor): The actual labels.
#         target_scale (float, optional): Controls how quickly the loss decreases as the target prediction increases. Default is 0.01.
#         non_target_scale (float, optional): Scales the non-target loss to match the magnitude of the target loss. Default is 0.000001.
#         show (bool, optional): If true, print the losses for each batch. Default is True.
        
#     Returns:
#         loss (Tensor): The calculated loss for each batch.
#     """
#     loss = []
#     for batch_i in range(len(labels)):
#         target_index = int(labels[batch_i])
#         target_value = preds[batch_i][target_index]
#         non_target_values = torch.cat([preds[batch_i][:target_index], preds[batch_i][target_index+1:]])
#         non_target_loss = non_target_scale * non_target_values.mean()
#         target_loss = torch.exp(-target_scale * target_value)
#         total_loss = non_target_loss + target_loss
#         loss.append(total_loss.view(1))
#         if show:
#             print(f"Batch {batch_i} - Target Loss: {target_loss.item()}, Non-Target Loss: {non_target_loss.item()}, Total Loss: {total_loss.item()}")
#     loss = torch.cat(loss)
#     return loss


# In[24]:


import inspect
import torch.nn.functional as F

loss_functions = {
    'default': batch_loss,
    'balanced_reduction_loss': balanced_reduction_loss,
    'balanced_reduction_loss_ratio': balanced_reduction_loss_ratio,

    # PyTorch built-in loss functions with 'none' reduction
    'cross_entropy': nn.CrossEntropyLoss(reduction='none'),  # Implicitly applies softmax
    'nll_loss': nn.NLLLoss(reduction='none'),  # Assumes log softmax applied to predictions
    'multi_margin': nn.MultiMarginLoss(reduction='none'),
}


def log(x):
    return torch.log(x)

def apply_softmax(preds):
    return F.softmax(preds, dim=1)

def log_sum_exp_trick(tensor):
    """
    Apply the Log-Sum-Exp trick to a 2D tensor.
    Assumes that the tensor's shape is [batch_size, num_features].
    """
    # Get the maximum value for each row in the tensor
    a, _ = tensor.max(dim=1, keepdim=True)
    
    # Compute the log-sum-exp by subtracting the max (for numerical stability)
    lse = a + (tensor - a).exp().sum(dim=1, keepdim=True).log()
    
    # We'll squeeze the result to get rid of the unnecessary dimension
    return lse.squeeze()



def choose_function(func_name, x, y, loss_fn_params, output_transform='log'):
    # Apply transformations to raw outputs
    if output_transform == 'log':
        x = torch.log(x)
    elif output_transform == 'lse':
        x = log_sum_exp_trick(x)
    elif output_transform == 'softmax':
        x = apply_softmax(x)
    elif output_transform == 'none':
        pass
    else:
        print(f"Error: Unknown output transformation {output_transform}")
        return None

    func = loss_functions.get(func_name)
    if func is None:
        return print(f"Error: No function matched for the name {func_name}")

    # If it's a built-in PyTorch loss function, just call it directly
    if isinstance(func, nn.Module):
        losses = func(x, y)
        
        # If losses is already 1D, return directly
        if losses.dim() == 1:
            return losses
        # Else, aggregate across extra dimensions
        else:
            aggregated_losses = losses.mean(dim=tuple(range(1, losses.dim())))
            return aggregated_losses
    else:
        params = inspect.signature(func).parameters
        unknown_params = set(loss_fn_params.keys()) - set(params.keys())
        if unknown_params:
            print(f"Error: Unknown parameter(s) {unknown_params} for function {func_name}")
            return None
        return func(x, y, **loss_fn_params)


# In[25]:


merge_timesteps=int(probe_time_resolution//dt)
pixels_no = timesteps//merge_timesteps - silent_pixel
print('number of considered pixel =',pixels_no)
pixel_weights = tensor(np.ones(int(pixels_no)))
print('initialize the training_pixel_weight:\n',pixel_weights)


# In[78]:


print("probes before:",len(probes))
probes_active = probes.copy()
for i in range(len(probes_active)):
    weight=0
    for label in model.all_labels :
        weight+=abs(prob_weighting(probes_active[i],model,label,dist_cut_off_train,situation='Train')) 
        weight+=abs(prob_weighting(probes_active[i],model,label,dist_cut_off_test,situation='Test')) 
    if weight==0:
        probes_active[i]=None
probes_active = [item for item in probes_active if item is not None]
print("probes after :",len(probes_active))
model.probes=torch.nn.ModuleList(probes_active)


# In[26]:


print("probes before:",len(probes))
probes_active = probes.copy()
for i in range(len(probes_active)):
    weight=0
    for label in model.all_labels :
        weight+=abs(prob_weighting(probes_active[i],model,label,dist_cut_off_train,situation='Train')) 
        weight+=abs(prob_weighting(probes_active[i],model,label,dist_cut_off_test,situation='Test')) 
    if weight==0:
        probes_active[i]=None
probes_active = [item for item in probes_active if item is not None]
print("probes after :",len(probes_active))
model.probes=torch.nn.ModuleList(probes_active)


# In[27]:


'''Define directry and load model'''

plotdir = your_path + '/Spintorch_packages/plots/' + basedir + '/'
if not os.path.isdir(plotdir):
    os.makedirs(plotdir)
savedir = your_path + '/Spintorch_packages/models/' + basedir + '/'
if not os.path.isdir(savedir):
    os.makedirs(savedir) 

'''Load checkpoint'''

#--------------------------------------# auto continue
from pathlib import Path

savedir_path = Path(savedir)

files = savedir_path.glob("model_e*.pt")
max_number = -1

for file in files:
    number_str = file.stem.split("_e")[1]
    number = int(number_str)
    if number > max_number:
        max_number = number
epoch = max_number 
print('Start with epoch =',epoch)

#--------------------------------------#

epoch_init = epoch
if epoch_init>=0:
    while epoch_init >= 0:
        try:
        
            checkpoint = torch.load(savedir + 'model_e%d.pt' % (epoch_init))
            epoch = checkpoint['epoch']
            pixel_weights = checkpoint['pixel_weights']
            loss_dict = checkpoint['loss_dict']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            break  # If the file loads successfully, break the loop
        
        except EOFError:
            print(f"Error loading checkpoint for epoch {epoch_init}, trying epoch {epoch_init - 1}")
            epoch_init -= 1
if epoch_init == -1:
    active_labels=[]
    for _ , label in train_loader:
        active_labels+=label.tolist()
    active_labels=list(set(active_labels))
    loss_dict = {f'{key}': [] for key in active_labels}
    loss_dict['min_loss'] = []
    print('new loss dict is defined:\n',loss_dict)


# In[28]:


def show_signal_using(data_loader, plot_dir, title='training', close=False):
    data_size = min(16, len(data_loader.dataset))  # limit to the first 25

    for n_per_row in range(5, 1, -1):
        if data_size >= n_per_row ** 2:
            break

    rows = ceil(data_size / n_per_row)
    cols = n_per_row

    fig, ax = plt.subplots(rows, cols, sharey=True, figsize=(5+6*cols,4*rows),dpi=200)
    fig.suptitle(f'This is how all the {title} data looks like:', y=0.962+0.0042*rows,fontsize=14+2*cols)
    
    ax = ax.ravel()  # this makes indexing easier

    for i, (inputs, outputs) in enumerate(data_loader,1):
        if i > 16:  # add this line to break the loop after 16 iterations
            break
        ax[i-1].plot(inputs.squeeze().numpy(),label=f'Desired output:{int(outputs.item())}')  # modify index to start at 0
        ax[i-1].legend(loc='upper right')  # modify index to start at 0
        
    plt.tight_layout()
    plt.savefig(plot_dir+'aa_Input_signals_shape.png')
    
    if close:
        plt.close()


# In[29]:


dataset = train_loader.dataset  # Get the dataset from existing DataLoader
single_train_loader = torch.utils.data.DataLoader(dataset, batch_size=1)  # Create a new DataLoader with batch size of 1
show_signal_using(single_train_loader, plotdir, title='training', close=False)


# In[30]:


#note the model detail
#Write the details of the data, parameters and geometry used for this model
open (f'{plotdir}/aa_model_detail.txt', 'w').write(comments_for_this_training_and_testing)
open (f'{savedir}/aa_model_detail.txt', 'w').write(comments_for_this_training_and_testing)

param_names =  ["basedir", "dx", "dy", "dz","nx","ny","Ms","B0","geometry_type","Np","Ms_CoPt", "r0", "dr", "dm", "z_off", "rx", "ry", 
                "damping_width", "sig_start_top", "sig_start_bot","dt", "timesteps", 
                "optimizer_name", "optimizer_params", "loss_function", "loss_fn_params","square_after_sum","apply_softmax", "learn_rate", "epoch_max","pixel_weighting_step","pixel_weighting_cap",
                "Bt", "time_duration", "frequencies", "normalize_superposition","modulate_method", "desired_out_list", 
                "noise_std","clone","seed","batch_size","noise_std_test","clone_test","seed_test","test_epochs",
                "validation_ratio","probe_time_resolution","silent_pixel","dataset_path","train_ratio","val_ratio",
                "distribute_data_in_multi_epochs","f_band_start","f_bandwidth",
                "detection_width","probe_reading_method_Train","probe_reading_method_Test","weighting_intersect","weighting_sigma","dist_cut_off_train","dist_cut_off_test",
                "merge_ignore_with_last_probe","comments_for_this_training_and_testing"]
                

# create a new dictionary with only the defined parameters
parameters = {key: value for key, value in locals().items() if key in param_names and value is not None}
# save the defined parameters to a CSV file
df = pd.DataFrame([parameters])
df = df.transpose()
df.columns = ['value']
df.index.name = 'parameter'
df.to_csv(f'{plotdir}/aa_model_parameters.csv')
df.to_csv(f'{savedir}/aa_model_parameters.csv')


# In[31]:


dist_list=np.linspace(0,10,5000)
weight_list_label_train=[gaussian_weight_fn(dist,dist_cut_off_train,'Train',equal_weight=model.probe_coordinate['Train']) for dist in dist_list]
weight_list_label_test=[gaussian_weight_fn(dist,dist_cut_off_test,'Test',equal_weight=model.probe_coordinate['Test']) for dist in dist_list]
weight_list_ignore_train=[gaussian_weight_fn(dist,dist_cut_off_train,'Train',label_ignore=True) for dist in dist_list]

fig,ax=plt.subplots(figsize=(12,5),dpi=120)

ax.plot(dist_list,weight_list_label_train,'-',label='normal(Train)',alpha=0.9)
ax.plot(dist_list,weight_list_label_test,'--',label='normal(Test)',alpha=0.9)
if -1 in model.all_labels:
    ax.plot(dist_list,weight_list_ignore_train,'-',label='ignore(Train)',alpha=0.7)
    
ax.legend()
ax.set_title('current weighting function used')
ax.set_xlabel('distance (cell)')
ax.set_ylabel('weighting')
plt.savefig(plotdir + f'aa_probe_weighting_during_training.png')


# In[32]:


'''Speed up method'''
def calculate_mask(labels,model,situation='Train',show=False):
    mask = []
    active_probe=0
    all_labels=model.all_labels
    equal_weight=model.probe_coordinate[situation]
    if -1 not in labels:# skip the process for label ignore signals
        all_labels = [x for x in model.all_labels if x != -1] 
    

    for probe in model.probes:
        weight = 0
        for label in all_labels: # all labels for current batch
            if situation == 'Train':
                weight += abs(prob_weighting(probe,model,label,dist_cut_off_train,equal_weight=equal_weight,situation='Train'))
            elif situation == 'Test':
                weight += abs(prob_weighting(probe,model,label,dist_cut_off_test,equal_weight=equal_weight,situation='Test'))
        if weight == 0:
            mask.append(False)
        else:
            mask.append(True)
            active_probe+=1
    if show:
        print('mask contain labels:',all_labels)
        print(f"({active_probe:d}/{len(model.probes):d}) probes is active in this batch")
    return np.array(mask)


# In[33]:


def loss_dict_append(loss_dict, new_loss_list, labels, exclude_keys=['min_loss']):
    "'loss_dict' is a dictionary,'new_loss_list' is a 1d tensor,'labels' is a 1d tensor"
    new_dict = {key: [] for key in loss_dict.keys() if key not in exclude_keys}
    
    for loss, label in zip(new_loss_list, labels):
        label_key = f'{int(label.item())}'
        if label_key in new_dict:  # Make sure the label_key is not excluded
            new_dict[label_key] += [loss.item()]

    for key, value in new_dict.items():
        if value:  # Check if the value is not empty
            loss_dict[key] += [sum(value) / len(value)]
        else:
            loss_dict[key] += [None]
            
    return loss_dict


# In[34]:


def change_pixel_weighting(pixel_weights,start_pos,window_size,pixel_weighting_step,pixel_weighting_cap,show=False):
    with torch.no_grad():
        pixel_weights_local=pixel_weights.clone()
        changes = pixel_weighting_step*len(pixel_weights_local)/window_size
        [print('change_pixel_weighting():,old weights=',pixel_weights_local)if show==True else None]
        pixel_weights_local -= pixel_weighting_step
        pixel_weights_local[start_pos:start_pos+window_size] += changes
        [print('change_pixel_weighting():,new weights=',pixel_weights_local)if show==True else None]
        return pixel_weights_local.clamp_(min=pixel_weighting_cap[0], max=pixel_weighting_cap[1])


def obtain_pixel_weighting(merged_probes_reading,labels,pixel_weighting_step,pixel_weights,pixel_weighting_cap,func_name=loss_function['pixel_w_loss_fn'],loss_fn_params=loss_fn_params['pixel_w_loss_fn'],show=False):
    with torch.no_grad():
        print("obtain_pixel_weighting() : starting with pixel_weights=",pixel_weights)

        # initialize the 'min_loss' with the full pixel case
        full_pixel = merged_probes_reading.shape[1]
        probes_reading_full=merged_probes_reading.sum(dim=1)
        min_loss= torch.mean(choose_function(func_name,probes_reading_full,labels,{**loss_fn_params, 'show': show}))
        new_weight = pixel_weights # if full pixel case is the largest, then keep weighting un-changed
        
        for window_size in range(full_pixel, 0, -1): # from full to 1 pixel
            for start_pos in range(full_pixel - window_size, -1, -1): # from last possible place to bigin

                probes_reading_parts=merged_probes_reading[:,start_pos:start_pos+window_size,:].sum(dim=1)
                pixel_loss = torch.mean(choose_function(func_name,probes_reading_parts,labels,{**loss_fn_params, 'show': False})) 
                [print([start_pos,window_size,full_pixel],'loss=',pixel_loss)if show==True else None]

                if pixel_loss < min_loss:
                    min_loss = pixel_loss
                    new_weight=change_pixel_weighting(pixel_weights,start_pos,window_size,pixel_weighting_step,pixel_weighting_cap)

        if pixel_weighting_step ==0:
            return pixel_weights,min_loss
        else:
            return new_weight,min_loss


# In[35]:


def partly_sum_outputs(probe_out_tensor,time_resolution,dt):

    merge_no=int(time_resolution//dt)
    label_count=probe_out_tensor.shape[-1]
    print(f"{merge_no} time steps for every merge,total of {probe_out_tensor.shape[1]} steps")
    remainder = probe_out_tensor.size(1) % merge_no
    probe_out_tensor = probe_out_tensor[:, remainder:, :]    # Discard the first 'remainder' elements in the second dimension
    merge_tensor = probe_out_tensor.reshape(len(probe_out_tensor), -1, merge_no, label_count).sum(2)    # Reshape and sum
    return merge_tensor


# In[36]:


if os.path.isfile(f'{plotdir}/aa_log-pixel_weights_during_train.txt'):
    with open(f'{plotdir}/aa_log-pixel_weights_during_train.txt', 'r') as file:
        pixel_weights_tracking = file.read()
    print('pixel_weights_during_train log exist, continue with it')
else:
    pixel_weights_tracking = ''

import time

if os.path.isfile(f'{plotdir}/aa_log-training_time.txt'):
    with open(f'{plotdir}/aa_log-training_time.txt', 'r') as file:
        training_time_tracking = file.read()
    print('training_time log exist, continue with it')
else:
    training_time_tracking = ''


# In[37]:


tic()

for epoch in range(epoch_init+1, epoch_max):

    epoch_start_time = time.time()

    stored_loss=[tensor([]),tensor([])]
    stored_min_loss=[]
    model.m_history=[] #prevent memory leak
    
    if distribute_data_in_multi_epochs != None and distribute_data_in_multi_epochs>1:
        print(f"Now, using train_loader number:{int(epoch%distribute_data_in_multi_epochs)}")
        train_loader_used=train_loader_list[int(epoch%distribute_data_in_multi_epochs)]
    else:
        train_loader_used=train_loader_list

    for batch, data in enumerate(train_loader_used, 0):

        INPUTS,labels = data
        mask_for_this_batch=calculate_mask(labels,model,show=True)
        model.set_mask(mask_for_this_batch)

        optimizer.zero_grad()
        model_out_tensor = model(INPUTS)
        #--- --- --- ---

        probe_out_tensor = merge_vertual_probes(model,model_out_tensor,labels,situation='Train',square_after_sum=square_after_sum)
        pixel_probes_reading = partly_sum_outputs(probe_out_tensor, probe_time_resolution, dt)
        pixel_probes_reading=pixel_probes_reading[:,silent_pixel:,:] # throw away the starting no signal time period

        pixel_weights,min_loss=obtain_pixel_weighting(pixel_probes_reading,labels,pixel_weighting_step,pixel_weights,pixel_weighting_cap,func_name=loss_function['pixel_w_loss_fn'],loss_fn_params=loss_fn_params['pixel_w_loss_fn'])
        stored_min_loss.append(min_loss)
        
        # save the weighting log as file
        print(f"Train: pixel_weights={[round(x.item(),3) for x in pixel_weights]}")
        pixel_weights_tracking+=f"epoch={epoch},batch={batch},pixel_weights={[round(x.item(),3) for x in pixel_weights]}\n"
        open (f'{plotdir}/aa_log-pixel_weights_during_train.txt', 'w').write(pixel_weights_tracking)

        # apply the weighting and calculate the loss
        weighted_pixel_probes_reading = torch.mul(pixel_probes_reading, pixel_weights.view(1, len(pixel_weights), 1)) 
        weighted_loss = torch.mean(choose_function(loss_function['train_loss_fn'],weighted_pixel_probes_reading.sum(dim=1),labels,loss_fn_params['train_loss_fn']))
        print(f"Train: weighted_loss={weighted_loss}")

        #--- --- --- ---

        with torch.no_grad():
            Test_reading = merge_vertual_probes(model,model_out_tensor,labels,situation='Test',square_after_sum=square_after_sum).sum(dim=1)
            Test_loss_list= choose_function(loss_function['train_loss_fn'],Test_reading,labels,loss_fn_params['train_loss_fn'])
            stored_loss[0] = torch.cat((stored_loss[0], Test_loss_list), dim=0)
            stored_loss[1] = torch.cat((stored_loss[1], labels), dim=0)

            if epoch%5==4 or epoch==0:
                merged_probes_reading = pixel_probes_reading.sum(dim=1)
                spintorch.plot.plot_multiple_outputs(merged_probes_reading,labels,f"Epoch{epoch}-batch{batch}", plotdir)
 
         #--- --- --- ---

        weighted_loss.backward()
        optimizer.step()
        print(f"epoch {epoch} batch {batch} finished:  -- Loss (not weighted): {Test_loss_list.mean(dim=0).item():.4f} , weighted (used for training):  {weighted_loss.item():.4f}")
    
    toc()  

    loss_dict = loss_dict_append(loss_dict,stored_loss[0],stored_loss[1], exclude_keys=['min_loss'])
    
    print('stored_min_loss=',stored_min_loss)
    print('loss_dict[\'min_loss\']=', loss_dict['min_loss'])

    loss_dict['min_loss'] += [np.mean(stored_min_loss)]
    avg_loss = [sum(val for val in (loss_dict[key][i] for key in loss_dict) if val is not None) / len(loss_dict.keys()) for i in range(len(loss_dict['0']))]# average across all labels


    min_loss_dict={'normal_loss':avg_loss,'min_loss':loss_dict['min_loss']}

    spintorch.plot.plot_loss(avg_loss, plotdir+'aa')
    spintorch.plot.plot_loss_dict(min_loss_dict, plotdir+'aa_min', exclude_keys=[])
    spintorch.plot.plot_loss_dict(loss_dict, plotdir+'aa', exclude_keys=['min_loss'])

    '''Save model checkpoint'''
    torch.save({
                'epoch': epoch,
                'pixel_weights':pixel_weights,
                'loss_dict': loss_dict,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, savedir + f'model_e{epoch:d}.pt')
    with open(f'{plotdir}aa_log-loss_dict.pkl', 'wb') as f:
        pickle.dump(loss_dict, f)
    '''Plot spin-wave propagation'''
    if epoch%5==4 or epoch==0:
        with torch.no_grad():
            spintorch.plot.geometry(model, epoch=epoch, plotdir=plotdir,probe_center_list=probe_center_list)
            mz = torch.stack(model.m_history, 1)[0,:,2,]-model.m0[0,2,].unsqueeze(0).cpu()
            try:
                wave_snapshot(model, mz[timesteps-1], (plotdir+'snapshot_time%d_epoch%d.png' % (timesteps,epoch)),r"$m_z$")
                wave_snapshot(model, mz[int(timesteps/2)-1], (plotdir+'snapshot_time%d_epoch%d.png' % (int(timesteps/2),epoch)),r"$m_z$")
                wave_integrated(model, mz, (plotdir+'integrated_epoch%d.png' % (epoch)),probe_center_list=probe_center_list)
            except Exception as e:
                print(f"An error occurred: {e}")
    training_time_tracking+=f"{time.time()-epoch_start_time:.2f}\n"
    open (f'{plotdir}/aa_log-training_time.txt', 'w').write(training_time_tracking)


# ## Model accuracy test

# In[ ]:


'''Define usefull ploting function'''

def intensity_time_plot(model_out_list, OUTPUTS, case, loss_list, plotdir, show=False):
    # improved
    data_size = [len(model_out_list), len(model_out_list[0][0])] # batch size , probe number
    tot_size = data_size[0] * data_size[1] # how many images in total will be plotted

    if data_size[0] == data_size[1]: # if output has a shape n*n just plot n in a row  
        n_per_row = data_size[0]
    else:
        for n_per_row in range(10, 1, -1):
            if tot_size >= n_per_row ** 2:
                break # if there are x**2 or more images, plot with x in a row (up to 10)

    fig, ax = plt.subplots(ceil(tot_size/n_per_row), n_per_row, figsize=(n_per_row*5, 2.5*ceil(tot_size/n_per_row)+1), sharey=True, sharex=True)
    fig.suptitle(f'Test case: {case} ', fontsize=18)

    ax = ax.flatten() # flattening the axes for easier handling

    for i in range(tot_size): #plot all the output vs time plots
        ax[i].set_title(f"No. {int(i/data_size[1])} in batch, probe {int(i%data_size[1])}")
        if OUTPUTS[int(i/data_size[1])].item() == i%data_size[1]: # if plotting the desired prob output, print the loss in legend
            ax[i].plot(model_out_list[int(i/data_size[1])][:,int(i%data_size[1])].detach().numpy(), label=f'Loss:{loss_list[int(i/data_size[1])]:.3f}')
            ax[i].legend(loc='upper left')
        else:
            ax[i].plot(model_out_list[int(i/data_size[1])][:,int(i%data_size[1])].detach().numpy())
    
    plt.tight_layout()
    plt.savefig(plotdir + f'prob_result_ver_time_case{case}.png')
    
    if show == True: # show the plots in terminal only when been asked
        plt.show()
    else:
        plt.close(fig)


# In[ ]:


'''Define prediction method'''

class Diff_Regression:
    """example use: model = Diff_Regression(num_steps=500, score_func=f1_score)"""
    def __init__(self, num_steps=1000, score_func=sk_metric.accuracy_score):
        self.best_threshold = None
        self.num_steps = num_steps
        self.score_func = score_func

    def fit(self, validation_preds, validation_labels):
        validation_preds_diff = [x[0] - x[1] for x in validation_preds]  # computing the difference

        thresholds = np.linspace(min(validation_preds_diff), max(validation_preds_diff), num=self.num_steps)
        best_threshold, best_score = None, 0

        for threshold in thresholds:
            predicted_labels = [1 if x - threshold > 0 else 0 for x in validation_preds_diff]
            score = self.score_func(validation_labels, predicted_labels)

            if score > best_score:
                best_threshold, best_score = threshold, score

        self.best_threshold = best_threshold
        return self

    def predict(self, X):
        if self.best_threshold is None:
            raise ValueError("Model not fitted yet")

        X_diff = [x[0] - x[1] for x in X]  # computing the difference
        return np.array([1 if x - self.best_threshold > 0 else 0 for x in X_diff])


# In[ ]:


import torch

class MaxValue_Predictor:
    """A simple model that predicts the label corresponding to the index of the maximum value in a given list"""
    def __init__(self):
        self.value_index_label_map = {}

    def fit(self, preds, labels):
        if len(preds) != len(labels):
            raise ValueError("Length of preds and labels must be the same.")
        
        # Create temporary storage to count label frequency for each index
        index_label_count = {}

        # Initialize dictionary for all possible indices
        for i in range(preds.shape[1]):
            index_label_count[i] = {}

        # Iterate over each data point
        for index, value_list in enumerate(preds):
            max_value_index = torch.argmax(value_list).item()  # Get the index of max value

            # Count label frequency
            label = int(labels[index].item())
            index_label_count[max_value_index][label] = index_label_count[max_value_index].get(label, 0) + 1
         
        print("Before mapping: ", index_label_count)

        # Now create the mapping between value index and most frequent label
        for index, label_count in index_label_count.items():
            # If there's data for this index, get the most frequent label; otherwise default to the index itself
            self.value_index_label_map[index] = max(label_count, key=label_count.get) if label_count else index

        print("After mapping: ", self.value_index_label_map)

        return self

    def predict(self, preds):
        predictions = []
        print("Max index:")
        for value_list in preds:
            max_value_index = torch.argmax(value_list).item()  # Get the index of max value
            print(max_value_index,end='')
            
            # Get label corresponding to max_value_index
            prediction = self.value_index_label_map.get(max_value_index, None)
            
            if prediction is None:
                raise ValueError(f"No label found for index {max_value_index}. Ensure model has been fitted with appropriate data.")
            
            predictions.append(prediction)
        print('')
        return predictions


# In[ ]:


def logistic_regression_train(X_train, y_train):
    # Create and train the logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def diff_d0_regression_train(X_train, y_train):
    model = Diff_Regression()
    model.fit(X_train, y_train)
    return model

def train_the_model(model_type,X_train, y_train):
    model = model_type
    model.fit(X_train, y_train)
    return model


def regression_model_predict(model, X_test):
    # Make predictions
    predictions = model.predict(X_test)
    return predictions


# In[ ]:


def acc_and_confusion_mx(pred_labels, test_labels):
    accuracy = sk_metric.accuracy_score(test_labels, pred_labels)
    cm = sk_metric.confusion_matrix(test_labels, pred_labels)
    return accuracy, cm


# def draw_roc_curve(pred_label, test_label,plt_dir):

#     fpr, tpr, thresholds = sk_metric.roc_curve(test_label, pred_label)
#     roc_auc = sk_metric.auc(fpr, tpr)

#     plt.figure(figsize=(5,5),dpi=100)
#     plt.plot(fpr, tpr, color='darkorange', lw=2.5, label='ROC curve (AUC = %0.2f)' % roc_auc)
#     plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.0])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic')
#     plt.legend(loc="lower right")
#     plt.savefig(plt_dir)
#     plt.close()


# In[ ]:


print("probes before:",len(probes_active))
probes_test_active = probes_active.copy()
for i in range(len(probes_test_active)):
    weight=0
    for label in model.all_labels :
        weight+=abs(prob_weighting(probes_test_active[i],model,label,dist_cut_off_train,situation='Test'))
    if weight==0:
        probes_test_active[i]=None
probes_test_active = [item for item in probes_test_active if item is not None]
print("probes after :",len(probes_test_active))
model.probes=torch.nn.ModuleList(probes_test_active)


# ### Define testing functions

# In[ ]:


def load_tained_model(test_epoch):

    checkpoint = torch.load(savedir + 'model_e%d.pt' % (test_epoch))
    epoch = checkpoint['epoch']
    pixel_weights = checkpoint['pixel_weights']
    loss_dict = checkpoint['loss_dict']
    model.probes=torch.nn.ModuleList(probes_active)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print("probes before:",len(probes_active))
    probes_test_active = probes_active.copy()
    for i in range(len(probes_test_active)):
        weight=0
        for label in model.all_labels :
            weight+=abs(prob_weighting(probes_test_active[i],model,label,dist_cut_off_test,situation='Test'))
        if weight==0:
            probes_test_active[i]=None
    probes_test_active = [item for item in probes_test_active if item is not None]
    print("probes after :",len(probes_test_active))
    model.probes=torch.nn.ModuleList(probes_test_active)

    # creat dirictiry
    basedir_t = f"model_test_e{epoch}/"
    basedir_t = basedir+'/'+basedir_t + '/'
    plotdir_t = 'plots/' + basedir_t
    if not os.path.isdir(plotdir_t): # if folder dose not exist
        os.makedirs(plotdir_t)         # creat one
    return epoch,model,optimizer,plotdir_t


# In[ ]:


def plots_for_every_training_signal(model,single_train_loader,epoch,plotdir_t):
    model.m_history=[] #prevent memory leak
    model.retain_history =True
    with torch.no_grad():
        spintorch.plot.geometry(model, epoch=epoch, plotdir=plotdir_t,probe_center_list=model.probe_coordinate['p_center'])
        for case_index, data in enumerate(single_train_loader, 0):
            INPUTS,label = data
            mask_for_this_batch=calculate_mask(label,model,situation='Test',show=True)
            model.set_mask(mask_for_this_batch)

            model_out_tensor = model(INPUTS)
            if square_after_sum == False:
                probe_out_tensor_sq_false=merge_vertual_probes(model,model_out_tensor,labels,square_after_sum=False)
            probe_out_tensor=merge_vertual_probes(model,model_out_tensor,labels,square_after_sum=True)
            probe_reading = probe_out_tensor.sum(dim=1)

            test_loss = choose_function(loss_function['train_loss_fn'],probe_reading,label,loss_fn_params['train_loss_fn'])

            '''Plot spin-wave propagation'''
            #spintorch.plot.plot_output(probe_reading[0],int(label),f"{epoch}-case{case_index}", plotdir_t)
            intensity_time_plot(probe_out_tensor,label,case_index,test_loss,plotdir_t)# plot and save the prob result signal verses time 
            if square_after_sum == False:
                    intensity_time_plot(probe_out_tensor_sq_false,label,case_index,test_loss,plotdir_t+'_no_square')
            mz = torch.stack(model.m_history, 1)[0,:,2,]-model.m0[0,2,].unsqueeze(0).cpu()
            if case_index < 4:
                for time_fraction in range(1,6):
                    time=int(timesteps*(time_fraction/5))-1
                    wave_snapshot(model, mz[time], (plotdir_t+f'snapshot_time_epoch{epoch:d}_case{case_index:d}_t={time}.png'),r"$m_z$")
            wave_integrated(model, mz, (plotdir_t+f'integrated_epoch{epoch:d}_case{case_index:d}.png'),probe_center_list=probe_center_list)


# In[ ]:


#collect training data

def collect_testing_data(test_loader, model, time_resolution=None):
    model.retain_history =False
    with torch.no_grad():
        test_preds = torch.Tensor() # Empty tensor for predictions
        test_labels = torch.Tensor() # Empty tensor for labels
        for case_index, data in enumerate(test_loader, 0):
            INPUTS, label = data
            mask_for_this_batch = calculate_mask(label, model, situation='Test', show=False)
            model.set_mask(mask_for_this_batch)

            model_out_tensor = model(INPUTS)
            probe_out_tensor = merge_vertual_probes(model, model_out_tensor, labels,square_after_sum=square_after_sum)
            if time_resolution:
                probe_reading = partly_sum_outputs(probe_out_tensor, time_resolution, dt)
            else:
                probe_reading = probe_out_tensor.sum(dim=1).unsqueeze(0)

            # Reshape data and concatenate to existing tensors
            test_preds = torch.cat((test_preds, probe_reading), 0)
            test_labels = torch.cat((test_labels, label), 0)

    if -1 in test_labels and torch.isnan(test_preds).any():
        test_preds = test_preds[:,:, :-1]
        test_labels = test_labels + 1 # move everything up by 1 so ignore is '0' wanted sig1 is '1' ...

    return test_preds, test_labels


# In[ ]:


##testing
def acc_ROC_pixel(accuracy,confusion_mx,plotdir_t,method_used,pixel_used):

    result_text = f"""With method:{method_used} \nAccuracy: {accuracy*100:.2f}% \nConfusion Matrix: \n{confusion_mx}
    \n starting_pixel:{pixel_used[0]},window_size:{pixel_used[1]},total_pixel:{pixel_used[2]}
    """

    open (f'{plotdir_t}/model_accuracy_{method_used}.txt', 'w').write(result_text)
    print(result_text)
    # need fix
    #draw_roc_curve(test_preds, test_labels,plotdir_t+f'ROC_for_test_data_{method_used}.png')
    


def loop_test_itrator(validation_preds,test_preds,validation_labels,test_labels,model_using,plotdir_t,method_used):
    print(f'\n Below is the train and test for: \n {method_used} \n ---------------------------------')
    training_acc_history=''
    max_acc=0
    full_pixel=validation_preds.shape[1]
    for window_size in range(full_pixel, 0, -1): # from full to 1 pixel
        for start_pos in range(full_pixel - window_size, -1, -1): # from last possible place to bigin
            
            #train the model with validation data
            validation_preds_part=validation_preds[:,start_pos:start_pos+window_size,:].sum(dim=1)
            regressor = train_the_model(model_using,validation_preds_part, validation_labels)
            #acc test for the trained model 
            pred_labels_train = regression_model_predict(regressor, validation_preds_part)
            accuracy_validation, _ =acc_and_confusion_mx(pred_labels_train,validation_labels) #to prevent over fitting, we can use different validation data here
            
            #acc test using the model on test data
            test_preds_part=test_preds[:,start_pos:start_pos+window_size,:].sum(dim=1)
            pred_labels_test = regression_model_predict(regressor,test_preds_part)
            accuracy_test,confusion_mx_test=acc_and_confusion_mx(pred_labels_test,test_labels)

            #show and store the acc for each case
            pixel_used=[start_pos,window_size,full_pixel]
            result_history=f"validation acc:{accuracy_validation*100:.2f}%, test acc:{accuracy_test*100:.2f}%, pixel:{pixel_used}"
            print(result_history)
            training_acc_history+=result_history+'\n'

            if accuracy_validation>max_acc:
                max_acc=accuracy_validation
                print('used!')
                #save the data
                acc_ROC_pixel(accuracy_test,confusion_mx_test,plotdir_t,method_used,pixel_used)
            
            if window_size==full_pixel: #do another test for full pixel
                acc_ROC_pixel(accuracy_test,confusion_mx_test,plotdir_t,method_used+'_full_pixel',pixel_used)

    out_text = f"""With method:{method_used} the accuracy got for {len(validation_labels)} training data each has been devided into {full_pixel} pixel range is: \nacc,start pixel,pixel size,total available pixel \n{str(training_acc_history)}"""
    open (f'{plotdir_t}/model_acc_for_diff_pixel_{method_used}.txt', 'w').write(out_text)


# ### perform the test

# In[ ]:


'''accuracy test (with different methods)'''
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader

if distribute_data_in_multi_epochs != None and distribute_data_in_multi_epochs>1:
    dataset_0 = train_loader_list[0].dataset  # Get the dataset from existing DataLoader
    dataset_1 = train_loader_list[1].dataset  
    dataset_01 = ConcatDataset([dataset_0, dataset_1])
else:
    dataset_01 = train_loader_list.dataset

single_train_loader = torch.utils.data.DataLoader(dataset_01, batch_size=1)  # Create a new DataLoader with batch size of 1

print('validation size :',len(list(val_loader)),'testing size :',len(list(test_loader)))


for sample_epoch in reversed(sorted(test_epochs)):
    print('sampleing epoch:', sample_epoch)

    epoch,model,optimizer,plotdir_t=load_tained_model(sample_epoch)
    plotdir_t_path = Path(plotdir_t)
    #--auto skip
    files = list(plotdir_t_path.glob("model_accuracy*.txt"))
    print("No. of model_accuracy files exists:",len(files))
    if len(files) == 6 and Np ==2: #-- could be different!
        print('skiped')
        continue
    elif len(files) == 4 and Np !=2:
        print('skiped')
        continue

    if len(test_epochs) <=4 or sample_epoch==max(test_epochs):
        plots_for_every_training_signal(model,single_train_loader,sample_epoch,plotdir_t) #do simulation and make plots for every training signal

    val_preds,val_labels=collect_testing_data(val_loader,model,time_resolution=probe_time_resolution) #do the simulation for each sample signal
    val_preds=val_preds[:,silent_pixel:,:] # throw away the starting no signal time period
    test_preds,test_labels=collect_testing_data(test_loader,model,time_resolution=probe_time_resolution) #do the simulation for each sample signal
    test_preds=test_preds[:,silent_pixel:,:] # throw away the starting no signal time period

    # Train the MaxValue regressor model with training signals result
    loop_test_itrator(val_preds,test_preds,val_labels,test_labels,MaxValue_Predictor(),plotdir_t,'max_value_regressor')
    if Np ==2:
        # Train the diff regression model with training signals result
        loop_test_itrator(val_preds,test_preds,val_labels,test_labels,Diff_Regression(),plotdir_t,'diff_regressor')
    # Train the logistic regression model with training signals result
    loop_test_itrator(val_preds,test_preds,val_labels,test_labels,LogisticRegression(),plotdir_t,'logistic_regressor')


# In[ ]:


def load_min_loss_model(test_epoch,min_type):

    checkpoint = torch.load(savedir + 'model_e%d.pt' % (test_epoch))
    epoch = checkpoint['epoch']
    pixel_weights = checkpoint['pixel_weights']
    loss_dict = checkpoint['loss_dict']
    model.probes=torch.nn.ModuleList(probes_active)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print("probes before:",len(probes_active))
    probes_test_active = probes_active.copy()
    for i in range(len(probes_test_active)):
        weight=0
        for label in model.all_labels :
            weight+=abs(prob_weighting(probes_test_active[i],model,label,dist_cut_off_test,situation='Test'))
        if weight==0:
            probes_test_active[i]=None
    probes_test_active = [item for item in probes_test_active if item is not None]
    print("probes after :",len(probes_test_active))
    model.probes=torch.nn.ModuleList(probes_test_active)

    # creat dirictiry
    basedir_t = f"model_min-{min_type}_test_e{epoch}/"
    basedir_t = basedir+'/'+basedir_t + '/'
    plotdir_t = 'plots/' + basedir_t
    if not os.path.isdir(plotdir_t): # if folder dose not exist
        os.makedirs(plotdir_t)         # creat one
    return epoch,model,optimizer,plotdir_t


# In[ ]:


def find_min_epoch(loss_list):
    avg_loss = loss_list
    min_value = min(avg_loss)
    min_index = avg_loss.index(min_value)

    print("Minimum loss:", min_value)
    print("at epoch:", min_index)
    return min_index


# In[ ]:


import shutil
# an extra test at the epoch where loss is minimized.
for min_type in ['avg_loss','min_loss']:
    if min_type=='avg_loss':
        avg_loss = [sum(loss_dict[key][i] for key in loss_dict if loss_dict[key][i] is not None) / len(loss_dict.keys()) for i in range(len(loss_dict['0']))] # average across all labels
        sample_epoch = find_min_epoch(avg_loss)
    else:
        min_loss = loss_dict['min_loss']
        sample_epoch = find_min_epoch(min_loss)
        if find_min_epoch(avg_loss) == find_min_epoch(min_loss):
            print('epoch of avg_loss and min_loss are the same, skiped')
            shutil.copytree(load_min_loss_model(sample_epoch,'avg_loss')[3], load_min_loss_model(sample_epoch,'min_loss')[3], dirs_exist_ok=True)
            break

    print('sampleing epoch:', sample_epoch,'where [ training',min_type,'] is minimm')

    epoch,model,optimizer,plotdir_t=load_min_loss_model(sample_epoch,min_type)
    plotdir_t_path = Path(plotdir_t)
    #--auto skip
    files = list(plotdir_t_path.glob("model_accuracy*.txt"))
    print("No. of model_accuracy files exists:",len(files))
    if len(files) == 6 and Np ==2: #-- could be different!
        print('skiped')
        continue
    elif len(files) == 4 and Np !=2:
        print('skiped')
        continue

    plots_for_every_training_signal(model,single_train_loader,sample_epoch,plotdir_t) #do simulation and make plots for every training signal

    val_preds,val_labels=collect_testing_data(val_loader,model,time_resolution=probe_time_resolution) #do the simulation for each sample signal
    val_preds=val_preds[:,silent_pixel:,:] # throw away the starting no signal time period
    test_preds,test_labels=collect_testing_data(test_loader,model,time_resolution=probe_time_resolution) #do the simulation for each sample signal
    test_preds=test_preds[:,silent_pixel:,:] # throw away the starting no signal time period

    # Train the MaxValue regressor model with training signals result
    loop_test_itrator(val_preds,test_preds,val_labels,test_labels,MaxValue_Predictor(),plotdir_t,'max_value_regressor')
    if Np ==2:
        # Train the diff regression model with training signals result
        loop_test_itrator(val_preds,test_preds,val_labels,test_labels,Diff_Regression(),plotdir_t,'diff_regressor')
    # Train the logistic regression model with training signals result
    loop_test_itrator(val_preds,test_preds,val_labels,test_labels,LogisticRegression(),plotdir_t,'logistic_regressor')


# In[ ]:


# finally remove all the models on the way to save place
savedir_path = Path(savedir)
files = list(savedir_path.glob("model_e*.pt"))
test_epochs = set([str(epoch) for epoch in test_epochs])  # convert to string for easy comparison

file_dict = {}
for file in files:
    number_str = file.stem.split("_e")[1]
    file_dict[number_str] = file

if not file_dict:  # if the directory doesn't have any model files
    print("No model files in the directory.")
    exit()

# Get the last existing model
max_epoch = max(file_dict.keys(), key=int)
print(f"Max epoch: {max_epoch}")

# Delete all model files except the ones in test_epochs and the last one
if len(test_epochs)<6:
    for epoch, file in file_dict.items():
        if epoch not in test_epochs and epoch != max_epoch:
            os.remove(file)
            print(f"Removed {file}")
else:
    for epoch, file in file_dict.items():
        if epoch != int(max_epoch)//2 or epoch != int(max_epoch):
            os.remove(file)
            print(f"Removed {file}")

