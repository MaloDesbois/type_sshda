import tsaug 
import numpy as np
import torch
from model import sshda
import torch.nn as nn
import torch.optim as optim
from numpy import load
import torch.nn.functional as F
from torch.autograd import Function
import torch.nn.init as init
from utils import data_loading
from utils import EarlyStopping
from sklearn.metrics import f1_score
from utils import dropout,identité
import matplotlib.pyplot as plt
import time
from utils import my_transformation

L2018=np.load('/home/malo/Stage/Data/data modifiées 11 classes/l2018_modif.npz',allow_pickle=True)
L2019=np.load('/home/malo/Stage/Data/data modifiées 11 classes/l2019_modif.npz',allow_pickle=True)
L2020=np.load('/home/malo/Stage/Data/data modifiées 11 classes/l2020_modif.npz',allow_pickle=True)
R2018=np.load('/home/malo/Stage/Data/data modifiées 11 classes/r2018_modif.npz',allow_pickle=True)
R2019=np.load('/home/malo/Stage/Data/data modifiées 11 classes/r2019_modif.npz',allow_pickle=True)
R2020=np.load('/home/malo/Stage/Data/data modifiées 11 classes/r2020_modif.npz',allow_pickle=True)
T2018=np.load('/home/malo/Stage/Data/data modifiées 11 classes/t2018_modif.npz',allow_pickle=True)
T2019=np.load('/home/malo/Stage/Data/data modifiées 11 classes/t2019_modif.npz',allow_pickle=True)
T2020=np.load('/home/malo/Stage/Data/data modifiées 11 classes/t2020_modif.npz',allow_pickle=True)

n_epochs=10
data =[[L2018],[T2019]] # [[L2018,L2019,L2020,R2018,R2019,R2020],[T2019]] 
train_dataloader, train_dataloader1, train_dataloader2, test_dataloader,dates,data_shape = data_loading(data,nbr_s=400,nbr_t=800)
data_shape=(data_shape[0],data_shape[2],data_shape[1])

config={'emb_size':64,'num_heads':8,'Data_shape':data_shape,'Fix_pos_encode':'tAPE','Rel_pos_encode':'eRPE','dropout':0.2,'dim_ff':64}
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = sshda(config,11,2).to(device)

learning_rate = 0.0001
loss_fn = nn.CrossEntropyLoss()
s=0

optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
#transformation = tsaug.AddNoise(scale=0.01)
#transformation = tsaug.Quantize(n_levels=20)
#transformation = tsaug.TimeWarp(n_speed_change=5, max_speed_ratio=3)
#transformation = dropout(p=0.8)
#transformation = identité
nom_transformation ="add_noise"
valid_f1 = 0.
liste_transformation = [tsaug.AddNoise(scale=0.01), tsaug.Quantize(n_levels=20), tsaug.TimeWarp(n_speed_change=5, max_speed_ratio=3), dropout(p=0.8)]
transformation = my_transformation(0.3,liste_transformation,device)
for n in range(n_epochs):
    print(f'éqpoue {n+1}')
    tot_pred = []
    tot_labels = []
    start = time.time()
    if n < 5 :
        
        for xm_batch, y_batch_, dom_batch in train_dataloader:
            x_batch,m_batch = xm_batch[:,:,:2],xm_batch[:,:,2] # m_batch correspond aux mask du batch
            
            y_batch, y_batch_info = y_batch_[:,0], y_batch_[:,1]
            x_batch = x_batch.to(device)
            m_batch = m_batch.to(device)
            y_batch = y_batch.to(device)
            dom_batch = dom_batch.to(device)
            optimizer.zero_grad()
            pred_lab, pred_dom, emb_lab, emb_dom = model(x_batch, m_batch)
            emb_lab = nn.functional.normalize(emb_lab)
            emb_dom = nn.functional.normalize(emb_dom)
            loss_ortho = torch.mean(torch.sum(emb_lab*emb_dom,dim=1))
            
            loss_lab = loss_fn(pred_lab, y_batch)
            loss_dom = loss_fn(pred_dom,dom_batch)
            
            loss =  loss_lab + torch.abs(loss_ortho) + loss_dom
            
            loss.backward()
            optimizer.step()
            
            pred_npy = np.argmax(pred_lab.cpu().detach().numpy(), axis=1)
            tot_pred.append( pred_npy )
            tot_labels.append( y_batch.cpu().detach().numpy())
            
    elif 5<n<10 :
        
        tot_pred_pl = []
        tot_labels_pl = []
        for xm_batch, y_batch_, dom_batch in train_dataloader1:
            x_batch,mask_batch = xm_batch[:,:,:2],xm_batch[:,:,2] # m_batch correspond aux mask du batch
            y_batch, y_batch_info = y_batch_[:,0], y_batch_[:,1]
            x_batch = x_batch.to(device)
            mask_batch = mask_batch.to(device)
            y_batch = y_batch.to(device)
            dom_batch = dom_batch.to(device)
            
            i_l = [k for k in range(len(y_batch)) if y_batch[k] != -1 ]
            i_ul = [k for k in range(len(y_batch)) if y_batch[k] == -1 ]
            xl_batch,ml_batch, doml_batch = x_batch[i_l],mask_batch[i_l],dom_batch[i_l]
            yl_batch = y_batch[i_l].clone().detach()
            
            model.eval()
            pred_lab, pred_dom, emb_lab, emb_dom = model(x_batch, mask_batch)
            
            yul_batch = [ torch.argmax(pred_lab[k]) if max(pred_lab[k])>0.95 else torch.tensor(-1) for k in i_ul]  # pseudo label pour les données non labelisée 
            #yul_batch_info = y_batch_info[i_ul]                                                                 # contient les vrais labels des données considérées
                                                                                                                # non labélisées
            ind_loss_pl = [k for k in range(len(yul_batch)) if yul_batch[k] != torch.tensor(-1) ]                 # indices des pseudo-labels conservés
            yul_batch = torch.tensor(yul_batch).to(device)
            yul_batch = yul_batch.to(torch.int64)
            tot_pred_pl.append(yul_batch[ind_loss_pl].cpu().detach().numpy())                    # on ajoute dans cette liste les pseudo-labels de confiance
            #tot_labels_pl.append(yul_batch_info[ind_loss_pl].cpu().detach().numpy())              # on ajoute dans la liste les vrais labels correspondant aux pseudo-labels              
            model.train()
            optimizer.zero_grad()
            xul_batch, mul_batch, domul_batch = x_batch[i_ul], mask_batch[i_ul], dom_batch[i_ul]
            xul_batch, mul_batch,domul_batch = xul_batch.to(device), mul_batch.to(device), domul_batch.to(device)
            xul_batch,mul_batch = transformation.augment(xul_batch,mul_batch)    # on applique la transformation de données sur les données non labélisées
            #xul_batch=np.array(xul_batch.cpu())
            #xul_batch = transformation.augment(xul_batch)
            
            
            xul_batch = torch.tensor(xul_batch).to(device)
            x_batch =   torch.cat((xl_batch,xul_batch),axis=0)
            y_batch = torch.cat((yl_batch,yul_batch),axis=0)
            mask_batch = torch.cat((ml_batch,mul_batch),axis=0) 
            dom_batch = torch.cat((doml_batch,domul_batch),axis=0) 
            ind_loss = [k for k in range(len(y_batch)) if y_batch[k] != torch.tensor(-1) ] # ici on ne conserve que les éléments pour lesquels on a un label 
                                                                            #ou un pseudo label de confiance
            
            pred_lab, pred_dom, emb_lab, emb_dom = model(x_batch, mask_batch)
            if s <3 :
                print(pred_lab[:1])
                s+=1
            emb_lab = nn.functional.normalize(emb_lab)
            emb_dom = nn.functional.normalize(emb_dom)
            loss_ortho = torch.mean(torch.sum(emb_lab*emb_dom,dim=1))
            loss_lab = loss_fn(pred_lab[ind_loss],y_batch[ind_loss])
            loss_dom = loss_fn(pred_dom,dom_batch)
            
            loss = torch.abs(loss_ortho) + loss_lab + loss_dom
            loss.backward()
            optimizer.step()
            pred_npy = np.argmax(pred_lab.cpu().detach().numpy(), axis=1)
            tot_pred.append( pred_npy )
            tot_labels.append( y_batch.cpu().detach().numpy())
            
    else :
        
        tot_pred_pl = []
        tot_labels_pl = []
        for xm_batch, y_batch_, dom_batch in train_dataloader1:
            x_batch,mask_batch = xm_batch[:,:,:2],xm_batch[:,:,2] # m_batch correspond aux mask du batch
            y_batch, y_batch_info = y_batch_[:,0], y_batch_[:,1]
            x_batch = x_batch.to(device)
            mask_batch = mask_batch.to(device)
            y_batch = y_batch.to(device)
            dom_batch = dom_batch.to(device)
            
            i_l = [k for k in range(len(y_batch)) if y_batch[k] != -1 ]
            i_ul = [k for k in range(len(y_batch)) if y_batch[k] == -1 ]
            xl_batch,ml_batch, doml_batch = x_batch[i_l],mask_batch[i_l],dom_batch[i_l]
            yl_batch = y_batch[i_l].clone().detach()
            
            model.eval()
            pred_lab, pred_dom, emb_lab, emb_dom = model(x_batch, mask_batch)
            
            yul_batch = [ torch.argmax(pred_lab[k]) if max(pred_lab[k])>0.95 else torch.tensor(-1) for k in i_ul]  # pseudo label pour les données non labelisée 
            #yul_batch_info = y_batch_info[i_ul]                                                                 # contient les vrais labels des données considérées
                                                                                                                # non labélisées
            ind_loss_pl = [k for k in range(len(yul_batch)) if yul_batch[k] != torch.tensor(-1) ]                 # indices des pseudo-labels conservés
            yul_batch = torch.tensor(yul_batch).to(device)
            yul_batch = yul_batch.to(torch.int64)
            tot_pred_pl.append(yul_batch[ind_loss_pl].cpu().detach().numpy())                    # on ajoute dans cette liste les pseudo-labels de confiance
            #tot_labels_pl.append(yul_batch_info[ind_loss_pl].cpu().detach().numpy())              # on ajoute dans la liste les vrais labels correspondant aux pseudo-labels              
            model.train()
            optimizer.zero_grad()
            xul_batch, mul_batch, domul_batch = x_batch[i_ul], mask_batch[i_ul], dom_batch[i_ul]
            xul_batch, mul_batch,domul_batch = xul_batch.to(device), mul_batch.to(device), domul_batch.to(device)
            xul_batch,mul_batch = transformation.augment(xul_batch,mul_batch)    # on applique la transformation de données sur les données non labélisées
            #xul_batch=np.array(xul_batch.cpu())
            #xul_batch = transformation.augment(xul_batch)
            
            
            xul_batch = torch.tensor(xul_batch).to(device)
            x_batch =   torch.cat((xl_batch,xul_batch),axis=0)
            y_batch = torch.cat((yl_batch,yul_batch),axis=0)
            mask_batch = torch.cat((ml_batch,mul_batch),axis=0) 
            dom_batch = torch.cat((doml_batch,domul_batch),axis=0) 
            ind_loss = [k for k in range(len(y_batch)) if y_batch[k] != torch.tensor(-1) ] # ici on ne conserve que les éléments pour lesquels on a un label 
                                                                            #ou un pseudo label de confiance
            
            pred_lab, pred_dom, emb_lab, emb_dom = model(x_batch, mask_batch)
            if s <3 :
                print(pred_lab[:1])
                s+=1
            emb_lab = nn.functional.normalize(emb_lab)
            emb_dom = nn.functional.normalize(emb_dom)
            loss_ortho = torch.mean(torch.sum(emb_lab*emb_dom,dim=1))
            loss_lab = loss_fn(pred_lab[ind_loss],y_batch[ind_loss])
            loss_dom = loss_fn(pred_dom,dom_batch)
            
            loss = torch.abs(loss_ortho) + loss_lab + loss_dom
            loss.backward()
            optimizer.step()
            
            pred_npy = np.argmax(pred_lab.cpu().detach().numpy(), axis=1)
            tot_pred.append( pred_npy )
            tot_labels.append( y_batch.cpu().detach().numpy())
            
            
    print(time.time()-start)    
    tot_pred = np.concatenate(tot_pred)
    tot_labels = np.concatenate(tot_labels)
    fscore = f1_score(tot_pred, tot_labels, average="weighted")
    fscore_a = np.round(fscore,3)
    print(f'f_score {fscore_a}')
    print(tot_pred[:32])
      
model.eval()       
tot_pred = []
tot_labels = []
k=0
s=0
for xm_batch, y_batch in test_dataloader:
    x_batch,mask_batch = xm_batch[:,:,:2],xm_batch[:,:,2]
    if k< 3:
        k+=1
        print(x_batch.shape)
        print(x_batch[:1][0][:5])
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
    mask_batch = mask_batch.to(device)
    pred_lab, _, _, _ = model(x_batch, mask_batch)
    if s <3 :
        print(pred_lab[:1])
        s+=1
    pred_npy = np.argmax(pred_lab.cpu().detach().numpy(), axis=1)
    
    
    tot_pred.append( pred_npy )
    tot_labels.append( y_batch.cpu().detach().numpy())
tot_pred = np.concatenate(tot_pred)
tot_labels = np.concatenate(tot_labels)
print(tot_pred[:32])
print(tot_labels[:32])
fscore= f1_score(tot_pred, tot_labels, average="weighted")
print(fscore)
