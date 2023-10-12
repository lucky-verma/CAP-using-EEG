"""
pytorch data loader to load CAP Sleep Database (Physionet) into pytorch tensors 
"""

import os
import numpy as np
import torch
import torch.utils.data as data
import pyedflib
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, lfilter, freqz
from scipy.signal import resample

# define a function to read the edf file
def read_edf_file(file_path):
    f = pyedflib.EdfReader(file_path)
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    sigbufs = np.zeros((n, f.getNSamples()[0]))
    for i in np.arange(n):
        sigbufs[i, :] = f.readSignal(i)
    return sigbufs, signal_labels

# define a function to read the xml file
def read_xml_file(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    annotations = []
    for child in root:
        if child.tag == 'ScoredEvents':
            for grandchild in child:
                if grandchild.tag == 'ScoredEvent':
                    annotations.append(grandchild.attrib)
    return annotations

# define a function to read the annotations
def read_annotations(annotations):
    stages = []
    for i in np.arange(len(annotations)):
        if annotations[i]['EventType'] == 'Stages|Stages':
            stages.append(annotations[i])
    return stages

# define a function to get the sleep stages
def get_sleep_stages(stages):
    sleep_stages = []
    for i in np.arange(len(stages)):
        sleep_stages.append(stages[i]['EventConcept'])
    return sleep_stages

# create a class to load the dataset
class CAPSleepDataset(data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.edf')]
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        edf_filename = os.path.join(self.data_dir, self.file_list[idx])
        xml_filename = os.path.join(self.data_dir, self.file_list[idx].replace('.edf', '.xml'))
        
        # Read EDF file
        edf_reader = pyedflib.EdfReader(edf_filename)
        signals = []
        for i in range(edf_reader.signals_in_file):
            signal = edf_reader.readSignal(i)
            signals.append(signal)
        signals = np.array(signals)
        
        # Read XML annotations
        xml_tree = ET.parse(xml_filename)
        root = xml_tree.getroot()
        
        # Parse and process annotations as needed
        # For example, you can extract labels and convert them to tensors
        
        # Example: Extract labels (assuming binary classification)
        labels = root.find('.//SleepStage').text
        labels = [int(label) for label in labels.split()]  # Convert to a list of integers
        labels = torch.tensor(labels, dtype=torch.float32)
        
        # Apply any data transformations if provided
        if self.transform:
            signals = self.transform(signals)
        
        return signals, labels
