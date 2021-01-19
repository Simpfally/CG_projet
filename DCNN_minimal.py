import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils import data
from os import listdir
from os import makedirs
from os.path import join
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import time


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

def load_img(filepath):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img = Image.open(filepath)#.convert('YCbCr')
    img = transform(img)

    if img.size(0)<3:
        print("bug")
        print(filepath)
    
    return img

#dataset loader
class DatasetFromFolder(data.Dataset):
    def __init__(self, datadir):
        super(DatasetFromFolder, self).__init__()
        posdir = join(datadir, 'pedestrians_pos')
        filenames1 = [join(posdir, x) for x in listdir(posdir)]
        negdir = join(datadir, 'pedestrians_neg')
        filenames0 = [join(negdir, x) for x in listdir(negdir)]
        self.filenames = filenames1+filenames0

    def __getitem__(self,index):
        name = self.filenames[index]
        input = load_img(name)

        target = torch.zeros([1], dtype=torch.float)
        if name.find("neg") >=0:
            target[0] = 0 #no pedestrians
        elif name.find("pos") >=0:
            target[0] = 1 #pedestrian

        return input,target


    def __len__(self):
        return len(self.filenames)


##Classifier

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier ,self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5,5))
        #self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5))
        self.lin1 = torch.nn.Linear(in_features=11160, out_features=1)
        #self.lin3 = torch.nn.Linear(in_features=60, out_features=1)


    def forward(self,x):
        x = torch.nn.functional.relu(torch.nn.functional.max_pool2d(self.conv1(x), kernel_size=2))
        #x = torch.nn.functional.relu(torch.nn.functional.max_pool2d(self.conv2(x), kernel_size=2))
        x = x.reshape(x.size(0), -1)
        #x = torch.nn.functional.relu(self.lin1(x))
        #x = torch.nn.functional.relu(self.lin2(x))
        x = self.lin1(x)
        x = torch.sigmoid(x)

        return x


def train():
    starttime = time.time()
    d_learning_rate = 1e-3
    sgd_momentum = 0.9

    num_epochs = 100
    num_w = 16
    batch_s = 32

    print("Training with :")
    print("Num epochs:",num_epochs)
    print("Num workers:",num_w)
    print("Batchsize:",batch_s)
    print("Learning rate:", d_learning_rate)
    print("Momentum:", sgd_momentum)


    #loading classifier
    net = Classifier()

    #Loss and optimizer
    # we'll optimize the binary cross entropy
    # torch.nn.BCELoss
    # using the stochastic gradient descent
    # torch.optim.SGD
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=d_learning_rate, momentum=sgd_momentum)

    #loading data
    trainset = DatasetFromFolder("train")
    trainloader = DataLoader(trainset, num_workers=num_w, batch_size=batch_s, shuffle=True)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward 
            out = net(inputs)

            loss = criterion(out, labels)
            loss.backward()
            
            #optimize
            optimizer.step()
    
            # print statistics
            running_loss += loss.detach().numpy()
            #print(running_loss)
            #print('[%d, %5d] loss: %.7f' %
            #      (epoch + 1, i + 1, running_loss / 2000))
            #running_loss = 0.0
        lossy = running_loss/len(trainloader)
        #if epoch % 10 == 9:
        acc = test(net,num_w,batch_s)
        time_elapsed = time.time() - starttime
        print("Epoch:",epoch+1,"/",num_epochs, lossy, acc, time_elapsed)


        append(args, [epoch+1,lossy,acc, time_elapsed])
    
    time_elapsed = time.time() - starttime
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    torch.save(net.state_dict(), "classifier.pth")


def test(net, num_w, batch_s):

    starttime = time.time()

    #print("Testing with :")
    #print("Num workers:",num_w)
    #print("Batchsize:",batch_s)

    #setting the classifier in test mode
    net.eval()

    #loading data
    testset = DatasetFromFolder("test")
    testloader = DataLoader(testset, num_workers=num_w, batch_size=batch_s, shuffle=True)
    ngoodclassif = 0
    ntest = 0
    for i, data in enumerate(testloader, 0):
        # get the inputs
        inputs, labels = data

        #forward in the net
        out = net(inputs)

        #compute the number of well classified data and the total number of tests
        for p, l in zip(out, labels):
            if abs(p-l) < 0.5:
                ngoodclassif += 1
            ntest += 1
    
    ratio = ngoodclassif/ntest
    #print('Good classification ratio: %.2f %%' %(ratio*100))
    return ratio


def main():
    train()

if __name__ == '__main__':
    main()
