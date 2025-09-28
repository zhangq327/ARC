#code for Figure 7

import math
import queue

import torch
import torchvision.datasets
from  torchvision import transforms
from sklearn.datasets import  make_gaussian_quantiles
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
import torch.optim as optim
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader,Dataset
from sklearn.manifold import TSNE
from resnet50 import ResNet50

import  PIL
import random
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
np.random.seed(1)
random.seed(1)
torch.backends.cudnn.deterministic = True
LARGE_NUMBER = 1e9
norm=1.
eps=100

def dfs(i,graph,visit):
    print(i)
    visit[i]=1
    for j in range(10000):
        if graph[i][j] ==1 and visit[j] ==0:
            dfs(j,graph,visit)


def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


N=128
mean1 = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

val =torchvision.datasets.CIFAR10(root = '.',download=True,train=False,transform=transforms.Compose([transforms.RandomResizedCrop(32,
                    scale=(0.08, 1),
                   # scale=(0.99,1.0),
                    interpolation=PIL.Image.BICUBIC,
                ),transforms.ToTensor(),transforms.Normalize(mean1, std)]))
train_sets=[torchvision.datasets.CIFAR10(root = '.',download=True,train=False,transform=transforms.Compose([transforms.RandomResizedCrop(32,
                    scale=(0.08, 1.0),
                    #scale=(0.99,1.0),
                    interpolation=PIL.Image.BICUBIC,
                ),#transforms.RandomApply([transforms.ColorJitter(brightness=0.8,contrast=0.8,saturation=0.8,hue=0.2)],p=0),
             transforms.ToTensor(),transforms.Normalize(mean1, std)])) for i in range(10)]
validation_loader=torch.utils.data.DataLoader(val,batch_size=N,shuffle=False)
validation_loaders = [torch.utils.data.DataLoader(train_sets[i],batch_size=N,shuffle=False) for i in range(10)]
model = ResNet50()
length_validation = len(val)
optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.9,weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
model = model.to("cuda:0")

loaded_state_dict = torch.load('resnet50.pth',map_location='cuda:0')
model.load_state_dict(loaded_state_dict)

my_accuracy=0
correct=0
length_validation=len(val)
with torch.no_grad():
    for x, y in validation_loader:
        device='cuda:0'
        x, y = x.to(device), y.to(device)
        model.eval()
        yhat = model(x)
        yhat = yhat.reshape(-1, 10)
        loss = criterion(yhat, y)
        _, yhat2 = torch.max(yhat.data, 1)
        correct += (yhat2 == y).sum().item()

my_accuracy = 100 * correct / length_validation

print(my_accuracy)


fig = plt.figure()
ax = plt.subplot(111)
count=0
x=[]
label=[]
for x,y in validation_loader:
    print(count)
    count+=1
    x,y=x.to("cuda:0"),y.to("cuda:0")
    x = model(x).detach().cpu()
    x= x.view(len(x),-1)
    if count==1:
        result=x.cpu().numpy()
        label = y.cpu().numpy()
    else:
        result=np.append(result,x.cpu().numpy(),axis=0)
        label = np.append(label,y.cpu().numpy())
    #print(label)
    #print(len(result))
tsne = TSNE(n_components=2)
result = tsne.fit_transform(result)

choose=[4,1,2,3,4,5,6,7,8,9,10]
N=9*128
feature =[]
for i in range(10):
    count=0
    for x,y in validation_loaders[i]:
        count+=1
        if count ==10:
            break
        x=x.to('cuda:0')
        x=model(x).detach().cpu()
        x = x.view(len(x), -1)
        if feature==[]:
            feature = x.cpu().numpy()
        else:
            feature=np.append(feature,x.cpu().numpy(),axis=0)
        print(len(feature))
N=300
for t in range(10):
    dis = choose[t]
    for tt in range(7,8,1):
        thereshold=0.98
        print(t,tt,":")
        graph=np.zeros((N,N))
        feature = torch.Tensor(feature)
        feature = F.normalize(feature,p=2,dim=1)
        dis = feature@feature.t()
        num1=0
        sta=0
        for k in range(1):
            for i in range(N):
                for j in range(N):
                    if i==j:
                        continue
                    #print(i,j)
                    tmp=[dis[i+l*1152][j+ll*1152] for l in range(10) for ll in range(10)]

                    #print(min(tmp))
                    tmp=np.max(tmp)
                    if tmp > thereshold and label[i]==label[j]:
                        graph[i][j]=1
                        print(tmp)
                        sta+=tmp
                        num1+=1
                    else:
                        graph[i][j]=0
                    if label[i]!=label[j]:
                        if random.randint(0,10)<=11:
                            graph[i][j]=0
            print(num1)
            #x=x.view(N,-1)
            for j in range(0):
                graph1=np.matmul(graph,graph)
                graph1=np.minimum(graph1,1)
                graph=np.maximum(graph1,graph)
            graph2= np.zeros((N, N))
            thereshold=0.98
            num2=0
            for i in range(N):
                for j in range(N):
                    if i==j:
                        continue
                    print(i,j)
                    tmp=[dis[i+l*1152][j+ll*1152] for l in range(10) for ll in range(10)]
                    #print(min(tmp))
                    tmp=np.max(tmp)
                    if tmp > thereshold:
                        if label[i]==label[j]:
                            num1+=0
                        else:
                            num2+=1
                            graph2[i][j]=1

            N2=N
            fig = plt.figure()
            G = nx.Graph()
            point = [i for i in range(N2)]
            G.add_nodes_from(point)
            graph=graph[0:N2,0:N2]
            G = nx.Graph(graph)
            position = result[0:N2]
            colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(10)]
            colors = [colors[label[i]] for i in range(N2)]
            nx.draw_networkx_nodes(G, position, nodelist=point, node_color=colors, node_size=10)
            nx.draw_networkx_edges(G,position,edge_color='black')
            G = nx.Graph(graph2)
            nx.draw_networkx_edges(G, position,edge_color='red')
            plot1=plt.plot([],[],c="r")
            plot2 = plt.plot([],[],c="black")
            plt.legend(plot1+plot2,["# Inter-classes edges = "+str(num2),"# Intra-classes edges = "+str(num1)])
            plt.savefig('cifar_val3.pdf',)
            my_min = 0
            exit()


