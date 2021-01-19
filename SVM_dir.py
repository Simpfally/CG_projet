import cv2
import numpy as np
import matplotlib.pyplot as plt
import os, os.path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import argparse

# did not prevent the no-driver warning on my laptop for some reason
device = torch.device("cpu")
os.environ["CUDA_VISIBLE_DEVICES"]=""

postraindir = "./train/pedestrians_pos/"
negtraindir = "./train/pedestrians_neg/"
postestdir = "./test/pedestrians_pos/"
negtestdir = "./test/pedestrians_neg/"
sizehog = 3780

npos = len(os.listdir(postraindir))
nneg = len(os.listdir(negtraindir))
ndesc = npos + nneg

hog = cv2.HOGDescriptor()
hogdescs = np.zeros((ndesc, sizehog+1))
labels = np.zeros((ndesc,1))

##preparing data

idx = 0
for filename in os.listdir(postraindir) :
  img = cv2.imread(os.path.join(postraindir,filename), cv2.IMREAD_GRAYSCALE)

  h = np.transpose(hog.compute(img))
  v = np.ones((1,1))
  hogdescs[idx,:] = np.concatenate((h, v),1);
  labels[idx,0] = 1;
  idx = idx + 1

for filename in os.listdir(negtraindir) :
  img = cv2.imread(os.path.join(negtraindir,filename), cv2.IMREAD_GRAYSCALE)

  h = np.transpose(hog.compute(img))
  v = np.ones((1,1))
  hogdescs[idx,:] = np.concatenate((h, v),1);
  labels[idx, 0] = -1
  idx = idx + 1


thogdescs = np.zeros((ndesc, sizehog+1))
tlabels = np.zeros((ndesc,1))

##preparing data

idx = 0
for filename in os.listdir(postraindir) :
  img = cv2.imread(os.path.join(postraindir,filename), cv2.IMREAD_GRAYSCALE)

  h = np.transpose(hog.compute(img))
  v = np.ones((1,1))
  thogdescs[idx,:] = np.concatenate((h, v),1);
  tlabels[idx,0] = 1;
  idx = idx + 1

for filename in os.listdir(negtraindir) :
  img = cv2.imread(os.path.join(negtraindir,filename), cv2.IMREAD_GRAYSCALE)

  h = np.transpose(hog.compute(img))
  v = np.ones((1,1))
  thogdescs[idx,:] = np.concatenate((h, v),1);
  tlabels[idx, 0] = -1
  idx = idx + 1


## Write the SVM training code below

#uncomment below to plot a training error called errtrain
#plt.plot(errtrain)
#plt.title("Training error")
#plt.show()

## Write the test code below



class SVM(nn.Module):
    def __init__(self):
        super().__init__() 
        self.lin1 = nn.Linear(3781, 1)
        
    def forward(self, x):
        fwd = self.lin1(x)  # Forward pass
        return fwd


parser = argparse.ArgumentParser()
parser.add_argument("-c", type=float, default=0.01)
parser.add_argument("-f", type=str)
args = parser.parse_args()

if args.f is not None:
    with open(args.f, 'w') as f:
        f.write("")

def append(args, vals):
    if args.f is None:
        return
    vals = [str(x) for x in vals]
    with open(args.f, 'a') as f:
        f.write(",".join(vals) + "\n")


#print(hogdescs.shape) # 1808 col : 3781 values
#print(labels.shape) # 1808 col : 1 value (1 or -1)
X = torch.FloatTensor(hogdescs)  
Y = torch.FloatTensor(labels)
n = len(Y)
tX = torch.FloatTensor(thogdescs)  
tY = torch.FloatTensor(tlabels)
W = torch.zeros(3781)
max_epochs = 100
for epoch in range(1, max_epochs+1):
    shuffled = torch.randperm(n) #give random permutations of indexes
    correct = 0
    # forward&backward&optimize with training data
    for i in range(0, n):
        x = X[shuffled[i]]
        y = Y[shuffled[i]]


        output = W.matmul(x)
        if y * output <= 1:
            W = (1-args.c) * W + args.c * y * x
        else:
            W = (1-args.c) * W


        if float(output) * float(y) > 0:
            correct += 1
        
    # forward with testing data
    tcorrect = 0
    for i in range(0, n):
        x = tX[i]
        y = tY[i]
        output = W.matmul(x)
        if float(output) * float(y) > 0:
            tcorrect += 1

    print(f"Epoch: {epoch} / {max_epochs} {correct/n} {tcorrect/n}")
    append(args, [epoch, correct/n, tcorrect/n])
