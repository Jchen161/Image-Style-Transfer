import importlib
#import torch.utils.data
from data.base_dataset import BaseDataset


def create_dataset(opt):
    data_loader=CustomeDatasetDataLoader(opt)
    dataset=data_loader.load_dataset()
    return dataset

class CustomDatasetDataLoader():
    
    def __init__(self,opt):
        dataset_filename='data.base_dataset'
        datasetlib=importlib.import_module(dataset_filename)



