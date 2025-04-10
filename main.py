import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import random
import os
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

try:
    Path_anc = "/home/usio/Documents/StupidFaceRecognitionVS/StupidFaceRecognition/data/anc"
    Path_pos = "/home/usio/Documents/StupidFaceRecognitionVS/StupidFaceRecognition/data/pos"
    Path_neg = "/home/usio/Documents/StupidFaceRecognitionVS/StupidFaceRecognition/data/neg"

    os.makedirs(Path_anc)
    os.makedirs(Path_pos)
    os.makedirs(Path_neg)
except Exception as e:
    print(e)

lfw_path = "/home/usio/Documents/StupidFaceRecognitionVS/StupidFaceRecognition/lfw-deepfunneled/lfw-deepfunneled"
for dir_ in os.listdir(lfw_path):
    print(dir_)
    for img in os.listdir(os.path.join(lfw_path, dir_)):
        prev_path = os.path.join(lfw_path, dir_, img)
        new_path = os.path.join(Path_neg, img)
        os.replace(prev_path, new_path)