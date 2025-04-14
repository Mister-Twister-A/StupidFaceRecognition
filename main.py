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

matplotlib.use('TkAgg') 
Path_anc = "/home/usio/Documents/StupidFaceRecognitionVS/data/anc"
Path_pos = "/home/usio/Documents/StupidFaceRecognitionVS/data/pos"
Path_neg = "/home/usio/Documents/StupidFaceRecognitionVS/data/neg"

lfw_path = "/home/usio/Documents/StupidFaceRecognitionVS/StupidFaceRecognition/lfw-deepfunneled/lfw-deepfunneled"
for dir_ in os.listdir(lfw_path):
    if dir_[0] == '.':
        continue
    for img in os.listdir(os.path.join(lfw_path, dir_)):
        prev_path = os.path.join(lfw_path, dir_, img)
        new_path = os.path.join(Path_neg, img)
        os.replace(prev_path, new_path)

cap = cv2.VideoCapture(0)
last_f = None
while cap.isOpened():
    ret, f = cap.read()
    last_f = f[200:200+250, 200:200+250, :]
    cv2.imshow("lol", last_f)
    if cv2.waitKey(1) & 0XFF == ord('a'):
        i_path = os.path.join(Path_anc, f"{uuid.uuid1()}.jpg")
        cv2.imwrite(i_path, last_f)
    if cv2.waitKey(1) & 0XFF == ord('p'):
        i_path = os.path.join(Path_pos, f"{uuid.uuid1()}.jpg")
        cv2.imwrite(i_path, last_f)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

cap.release()
# lol
cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)
print(last_f.shape)
plt.imshow(last_f)
plt.show()
print("lol")
