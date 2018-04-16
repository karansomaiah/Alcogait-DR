# importing necessary functions
import numpy as np
import csv 
import sys
import os
import h5py
import pandas as pd
import simplejson as json
from scipy import stats
import gc

## contains mostly helper functions to get the dataset in the appropriate format
class data_reader:
    def __init__(self, dataset):
        self.data, self.id2label = self.readOpportunity()
        self.save_data(dataset)
        
    def save_data(self,dataset):
        if not os.path.exists('opportunity.h5'):
            f = h5py.File('opportunity.h5')
            for key in self.data:
                f.create_group(key)
                for field in self.data[key]:
                    f[key].create_dataset(field, data=self.data[key][field])
            f.close()
            print('Done saving h5 file')
            
    def readOpportunity(self):
        
        # splitting train and test
        files = {
            'train': ['S1-ADL1.dat','S1-ADL3.dat', 'S1-ADL4.dat', 'S1-ADL5.dat', 'S1-Drill.dat', 
                      'S2-ADL1.dat', 'S2-ADL2.dat', 'S2-ADL5.dat', 'S2-Drill.dat', 'S3-ADL1.dat', 
                      'S3-ADL2.dat', 'S3-ADL5.dat', 'S3-Drill.dat', 'S4-ADL1.dat', 'S4-ADL2.dat', 
                      'S4-ADL3.dat', 'S4-ADL4.dat', 'S4-ADL5.dat', 'S4-Drill.dat'],
            'test': ['S2-ADL3.dat', 'S2-ADL4.dat','S3-ADL3.dat', 'S3-ADL4.dat']
        }
        
        # mapping the labels
        label_map = [
            (0,      'Other'), (406516, 'Open Door 1'), (406517, 'Open Door 2'), (404516, 'Close Door 1'),
            (404517, 'Close Door 2'), (406520, 'Open Fridge'), (404520, 'Close Fridge'), (406505, 'Open Dishwasher'),
            (404505, 'Close Dishwasher'), (406519, 'Open Drawer 1'), (404519, 'Close Drawer 1'), (406511, 'Open Drawer 2'),
            (404511, 'Close Drawer 2'), (406508, 'Open Drawer 3'), (404508, 'Close Drawer 3'), (408512, 'Clean Table'),
            (407521, 'Drink from Cup'), (405506, 'Toggle Switch')
        ]
        
        # dictionary for label-id to getting integer labels
        labelToId = {str(x[0]): i for i, x in enumerate(label_map)}
        idToLabel = [x[1] for x in label_map]
        
        # column numbers to use
        cols = [
            37, 38, 39, 40, 41, 42, 43, 44, 45, 50, 51, 52, 53, 54, 55, 56, 57, 58,63, 64, 65, 66, 67, 68, 
            69, 70, 71, 76, 77, 78, 79, 80, 81, 82, 83, 84, 89, 90, 91, 92, 93, 94, 95, 96, 97, 102, 103, 
            104, 105, 106, 107, 108,109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
            124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 249
            ]
        
        # calling the read files dataset
        data = {dataset: self.readOpportunityFiles(files[dataset], cols, labelToId)
                for dataset in ('train', 'test')}

        return data, idToLabel
    
    def readOpportunityFiles(self, filelist, cols, labelToId):
        data = []
        labels = []
        for i, filename in enumerate(filelist):
            print('Reading file %d of %d' % (i+1, len(filelist)))
            with open('data/opportunity/OpportunityUCIDataset/dataset/%s' % filename, 'r') as f:
                reader = csv.reader(f, delimiter=' ')
                for line in reader:
                    elem = []
                    for ind in cols:
                        elem.append(line[ind])
                    if sum([x == 'NaN' for x in elem]) == 0:
                        data.append([float(x) / 1000 for x in elem[:-1]])
                        labels.append(labelToId[elem[-1]])

        return {'inputs': np.asarray(data, dtype='float32'), 'targets': np.asarray(labels, dtype='float32')+1}
    

def windowz(data, size):
    start = 0
    while start < len(data):
        yield start, start + size
        start += (size / 2)


def segment_opp(x_train,y_train,window_size):
    segments = np.zeros(((len(x_train)//(window_size//2))-1,window_size,77))
    labels = np.zeros(((len(y_train)//(window_size//2))-1))
    i_segment = 0
    i_label = 0
    for (start,end) in windowz(x_train,window_size):
        start = int(start); end = int(end)
        if(len(x_train[start:end]) == window_size):
            m = stats.mode(y_train[start:end])
            segments[i_segment] = x_train[start:end]
            labels[i_label] = m[0]
            i_label+=1
            i_segment+=1
    return segments, labels 


## STEPS:
# 1. Call the data_reader 'class' to read the data and structure it
# 2. Segment the signals into pieces of 24 and stack them together
# 3. Done.