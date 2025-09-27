import torch
import torch.nn.functional as F
import models
from numba import jit


def feature_store(train_loader, device, model):
    total = torch.tensor([]).to(device)
    for idx, batch in enumerate(train_loader):
        feature = model.model(batch[0].to(device))
        feature = F.normalize(feature, dim=1)
        total = torch.cat((total, feature.detach()), dim=0)
    return total

#total is a tensor which stores the features of data points, 
#views means the number of augmented views of the same raw samples, 
def ACR_calculate(total_features, views, batch_size, device):
    pos=0
    iters = len(total_features) // (batch_size*views)
    for j in range(iters):
        init_index = j * batch_size * views
        f1 = total_features[init_index: init_index + batch_size]
        dis = torch.cdist(f1,total_features)
        _, neighbour_indices = torch.sort(dis)
        #whether nearest neighbour is from the same raw sample
        nearest_neighbour = neighbour_indices[:, 1:2]
        nearest_neighbour = nearest_neighbour - torch.tensor([[i] for i in range(batch_size)]).to(device)
        nearest_neighbour = torch.where(nearest_neighbour % batch_size==0, 0, 1) 
        pos += torch.sum(nearest_neighbour).item()
    return pos/(iters*batch_size)


def ARC_calculate(args, init_model, final_model, device):

    # Data loading code
    init_model.prepare_data()
    train_loader, val_loader= init_model.dataloaders(iters=96000)

    #store features of init model
    total_features = feature_store(train_loader, device, init_model)
    ACR_init = ACR_calculate(total_features, args.views, args.batch_size, device)

    # store features of the final model
    total_features = feature_store(train_loader, device, final_model)
    ACR_final = ACR_calculate(total_features, args.views, args.batch_size,device)
    ARC = (1-float(ACR_final))/(1-float(ACR_init))
    print('ACR_init: '+ str(ACR_init))
    print('ACR_final: ' + str(ACR_final))
    print('ARC: ' + str(ARC))



@jit()
def metric_calculate(dis,metric1='mean',metric2='mean',views=2,bs=256):
    sim = np.ones((bs,len(dis[0])//views))
    sim = sim*100
    for j in range(bs):
        for t in range(len(dis[0])//(views*bs)-1):
            for k in range(t*bs*views,t*bs*views+bs):
                if j==k:
                    if metric1=='mean':
                        sim[j][k//(views*bs)*bs+k%bs]=np.mean([dis[j,k+l*256] for l  in range(1,views)])
                    elif metric1=='min':
                        sim[j][k//(views*bs)*bs+k%bs]=np.min([dis[j,k+l*256] for l  in range(1,views)])
                    elif metric1=='max':
                        sim[j][k//(views*bs)*bs+k%bs]=np.max([dis[j,k+l*256] for l  in range(1,views)])
                    
                else:
                    if metric2=='mean':
                        sim[j][k//(views*bs)*bs+k%bs]=np.mean([dis[j,k+l*256] for l  in range(1,views)])
                    elif metric2=='min':
                        sim[j][k//(views*bs)*bs+k%bs]=np.min([dis[j,k+l*256] for l  in range(1,views)])
                    elif metric2=='max':
                        sim[j][k//(views*bs)*bs+k%bs]=np.max([dis[j,k+l*256] for l  in range(1,views)])
    return sim



#total is a tensor which stores the features of data points, 
#views means the number of augmented views of the same raw samples, 
def GACR_calculate(total_features, views, batch_size, device, metric1, metric2):
    pos=0
    iters = len(total_features) // (batch_size*views)
    for j in range(iters):
        init_index = j * batch_size * views
        f1 = total_features[init_index: init_index + batch_size]
        dis = torch.cdist(f1,total_features)
        dis = dis.detach().cpu().numpy()
        sim = metric_calculate(dis, metric1, metric2, views, batch_size)
        sim = torch.tensor(sim)     
        _, neighbour_indices = torch.sort(sim)
        #whether nearest neighbour is from the same raw sample
        nearest_neighbour = neighbour_indices[:, 1:2]
        nearest_neighbour = nearest_neighbour - torch.tensor([[i] for i in range(batch_size)]).to(device)
        nearest_neighbour = torch.where(nearest_neighbour % batch_size==0, 0, 1) 
        pos += torch.sum(nearest_neighbour).item()
    return pos/(iters*batch_size)



def GARC_calculate(args, init_model, final_model, device, metric1='mean',metric2='mean'):

    # Data loading code
    init_model.prepare_data()
    train_loader, val_loader= init_model.dataloaders(iters=96000)

    #store features of init model
    total_features = feature_store(train_loader, device, init_model)
    ACR_init = GACR_calculate(total_features, args.views, args.batch_size, device, metric1, metric2)

    # store features of the final model
    total_features = feature_store(train_loader, device, final_model)
    ACR_final = GACR_calculate(total_features, args.views, args.batch_size,device, metric1, metric2)
    ARC = (1-float(ACR_final))/(1-float(ACR_init))
    print('ACR_init: '+ str(ACR_init))
    print('ACR_final: ' + str(ACR_final))
    print('ARC: ' + str(ARC))



