import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import random
import os
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import uuid
Path_anc = "/home/usio/Documents/StupidFaceRecognitionVS/data/anc"
Path_pos = "/home/usio/Documents/StupidFaceRecognitionVS/data/pos"
Path_neg = "/home/usio/Documents/StupidFaceRecognitionVS/data/neg"


def preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (100,100), interpolation=cv2.INTER_LINEAR)
    #img = torch.from_numpy(img).permute(2, 0, 1).float() # C H W
    img = torch.from_numpy(img).float() # C H W
    img = img / 255.0
    return img

#prep_img = preprocess("/home/usio/Documents/StupidFaceRecognitionVS/data/anc/dab94c4a-1bae-11f0-ba1d-089df4f1b4ae.jpg")


class ImgDataset(Dataset):
    def __init__(self, paths:list[str]):
        self.anc_paths:list[str] = paths
        self.pos_:list[str] = [os.path.join(Path_pos, img) for img in os.listdir(Path_pos)]
        self.neg_:list[str] = [os.path.join(Path_neg, img) for img in os.listdir(Path_neg)]
        self.cache:dict[str, torch.Tensor] = {}

    def __len__(self):
        return len(self.anc_paths)
    
    def __getitem__(self, index):
        anc_path = self.anc_paths[index]
        compare_path = ""
        choose_ = random.randint(0,1)
        if choose_ == 0:
            compare_path = self.neg_[random.randint(0,len(self.neg_)-1)]
        if choose_ == 1:
            compare_path = self.pos_[random.randint(0,len(self.pos_)-1)]
        
        if anc_path not in self.cache:
            self.cache[anc_path] = preprocess(anc_path)
        if compare_path not in self.cache:
            self.cache[compare_path] = preprocess(compare_path)
        
        return self.cache[anc_path], self.cache[compare_path], choose_
    


if __name__ == "__main__":
    print("running dataset")
    dataset = ImgDataset([os.path.join(Path_anc, img) for img in os.listdir(Path_anc)])
    DataLoader = DataLoader(dataset, 12)
    for i,b in enumerate(DataLoader):
        a,c,t = b
        
    

