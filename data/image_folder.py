import torch.utils.data as data

from PIL import Image
import os

IMAG_EXTENSIONS=['.jpg','.JPG','.png','PNG']

def make_dataset(dir,max_dataset_size=float("inf")):
    images=[]
    assert os.path.isdir(dir)

    for root,_,fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path=os.path.join(root,fname)
            images.append(path)
    return images[:min(max_dataset_size,len(images))]