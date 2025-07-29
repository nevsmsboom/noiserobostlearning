import os
import argparse, sys
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms

from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from O2Udata.mask_data import Mask_Select  

from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score, ConfusionMatrixDisplay

from medmnistutils.evaluationmetrics import accuracy, roc, presenf1cfsmtx
from medmnistutils.dataloaderwithnoise_O2U import PathMNIST, OrganMNIST3D, PneumoniaMNIST, VesselMNIST3D, OCTMNIST, BloodMNIST, OrganAMNIST


from medmnistutils.twodresnet import ResNet18 as twodresnet18
from medmnistutils.threedresnet import resnet18 as threedresnet18

from linearregression.lrpredict import lrpredict


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='PneumoniaMNIST', help='PathMNIST, OCTMNIST, PneumoniaMNIST, OrganMNIST3D, VesselMNIST3D')
parser.add_argument('--noise_rate', type = float, default = 0.4, help = 'corruption rate, should be less than 1')
parser.add_argument('--batchsize', type=int, default=128, help='batchsize')

parser.add_argument('--phase1_epoch', type=int, default=90)
parser.add_argument('--phase2_epoch', type=int, default=20)
parser.add_argument('--phase3_epoch', type=int, default=90)
#parser.add_argument('--seed', type=int, default=23333)

#args = parser.parse_args(args=[])
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#torch.manual_seed(args.seed)
#torch.cuda.manual_seed(args.seed)


if args.dataset =='PathMNIST':
    newtransform = transforms.Compose([transforms.ToTensor()])
    train_dataset = PathMNIST(split = 'train', root = '../../medmnistdata',  transform=newtransform, noise_rate=args.noise_rate)
    val_dataset = PathMNIST(split = 'val', root = '../../medmnistdata',  transform=newtransform)
    test_dataset = PathMNIST(split = 'test', root = '../../medmnistdata',  transform=newtransform)
    model = twodresnet18(input_channel=train_dataset.in_channels, n_outputs=train_dataset.num_classes)

if args.dataset =='OCTMNIST':
    newtransform = transforms.Compose([transforms.ToTensor()])
    train_dataset = OCTMNIST(split = 'train', root = '../../medmnistdata',  transform=newtransform, noise_rate=args.noise_rate)
    val_dataset = OCTMNIST(split = 'val', root = '../../medmnistdata',  transform=newtransform)
    test_dataset = OCTMNIST(split = 'test', root = '../../medmnistdata',  transform=newtransform)
    model = twodresnet18(input_channel=train_dataset.in_channels, n_outputs=train_dataset.num_classes)

elif args.dataset =='PneumoniaMNIST':
    newtransform = transforms.Compose([transforms.ToTensor()])
    train_dataset = PneumoniaMNIST(split = 'train', root = '../../medmnistdata',  transform=newtransform, noise_rate=args.noise_rate)
    val_dataset = PneumoniaMNIST(split = 'val', root = '../../medmnistdata',  transform=newtransform)
    test_dataset = PneumoniaMNIST(split = 'test', root = '../../medmnistdata',  transform=newtransform)
    model = twodresnet18(input_channel=train_dataset.in_channels, n_outputs=train_dataset.num_classes)
    
elif args.dataset =='OrganMNIST3D':
    train_dataset = OrganMNIST3D(split = 'train', root = '../../medmnistdata',  transform=None, noise_rate=args.noise_rate)
    val_dataset = OrganMNIST3D(split = 'val', root = '../../medmnistdata',  transform=None)
    test_dataset = OrganMNIST3D(split = 'test', root = '../../medmnistdata',  transform=None)
    model = threedresnet18(num_classes = train_dataset.num_classes)

elif args.dataset =='VesselMNIST3D':
    train_dataset = VesselMNIST3D(split = 'train', root = '../../medmnistdata',  transform=None, noise_rate=args.noise_rate)
    val_dataset = VesselMNIST3D(split = 'val', root = '../../medmnistdata',  transform=None)
    test_dataset = VesselMNIST3D(split = 'test', root = '../../medmnistdata',  transform=None)
    model = threedresnet18(num_classes = train_dataset.num_classes)



model.to(device)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,batch_size=args.batchsize,num_workers=0,shuffle=True, pin_memory=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=args.batchsize,num_workers=0,shuffle=True, pin_memory=True)

noise_or_not = train_dataset.noise_or_not


###############################################################################lnlsr的

tau, p, lamb, rho, freq = 0.5, 0.1, 1.2, 1.03, 1

weight_decay = 1e-4
lr = 0.01 

def calculate_loss(criterion, out, y, norm=None, lamb=None, tau=None, p=None):
    out = F.normalize(out, dim=1)
    if train_dataset.num_classes ==2:
        loss = criterion(out, y)
    else:
        loss = criterion(out / tau, y) + lamb * norm(out / tau, p)
    return loss

eps = 1e-7
class GCELoss(nn.Module):
    def __init__(self, num_classes, q=0.7):
        super(GCELoss, self).__init__()
        self.q = q
        self.num_classes = num_classes
    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=eps, max=1.0)
        labels = labels.to(torch.int64)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return loss.mean()

class pNorm(nn.Module):
    def __init__(self, p=0.5):
        super(pNorm, self).__init__()
        self.p = p
    def forward(self, pred, p=None):
        if p:
            self.p = p
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1)
        norm = torch.sum(pred ** self.p, dim=1)
        return norm.mean()

gcecriterion = GCELoss(num_classes=train_dataset.num_classes)
norm = pNorm(p)

gceoptimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
gcescheduler = StepLR(gceoptimizer, gamma=0.1, step_size=25)




###############################################################################
#first state
print('first stage')

#model = model
filter_mask=None
train_loader_init = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=128,num_workers=0,shuffle=True, pin_memory=True)



criterion1 = GCELoss(num_classes=train_dataset.num_classes)
optimizer1 = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=weight_decay)
scheduler1 = StepLR(optimizer1, gamma=0.1, step_size=25)

for epoch in range(args.phase1_epoch):
    model.train()
    for i, (images, labels, indexes) in enumerate(train_loader_init):
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)
        logits = model(images)
        loss_1 = calculate_loss(criterion1, logits, labels.squeeze(), norm, lamb, tau, p)
        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()
    scheduler1.step()
    
    if (epoch + 1) % freq == 0:
        lamb = lamb * rho
        
    #with torch.no_grad():
    valaccuracy = accuracy(model, val_loader) 
    testaccuracy = accuracy(model, test_loader)

    print ("epoch", epoch+1,"val_accuarcy", valaccuracy,"test_accuarcy", testaccuracy)

save_checkpoint ='middle.pt'
torch.save(model.state_dict(), save_checkpoint)



###############################################################################
#second stage
print('second stage')

#model=model
#test_loader=test_loader

train_loader_detection = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=16,num_workers=0,shuffle=True)
optimizer1 = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
criterion=torch.nn.CrossEntropyLoss(reduce=False, ignore_index=-1).to(device)
moving_loss_dic=np.zeros_like(noise_or_not)
ndata = train_dataset.__len__()
        
for epoch in range(args.phase2_epoch):
    # train models
    globals_loss=0
    
    model.train()
    example_loss= np.zeros_like(noise_or_not,dtype=float)
    
    t = (epoch % 10 + 1) / float(10)
    lr = (1 - t) * 0.01 + t * 0.001
    
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr
        
    for i, (images, labels, indexes) in enumerate(train_loader_detection):
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)
        logits = model(images)
        loss_1 =criterion(logits.float(),labels.squeeze().long()) 
        for pi, cl in zip(indexes, loss_1):
            example_loss[pi] = cl.cpu().data.item()
        globals_loss += loss_1.sum().cpu().data.item()
        loss_1 = loss_1.mean()
        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()

    #with torch.no_grad():
    valaccuracy = accuracy(model, val_loader)
    testaccuracy = accuracy(model, test_loader)        

    example_loss=example_loss - example_loss.mean()
    moving_loss_dic=moving_loss_dic+example_loss

    #predict noiserate and forget rate
    predictednoiserate = lrpredict(linearregressionmodel=joblib.load('linearregression/' + args.dataset), 
                                    nparray=moving_loss_dic, 
                                    size_dataset=len(train_dataset), 
                                    num_classes=train_dataset.num_classes)[0] 
    print('predicted noiserate', predictednoiserate, 'actual noiserate', args.noise_rate)
    predictednoiserate = max(0, predictednoiserate)
    predictednoiserate = min(0.5, predictednoiserate)
    forget_rate = 0.8*predictednoiserate
      
    ind_1_sorted = np.argsort(moving_loss_dic)
    loss_1_sorted = moving_loss_dic[ind_1_sorted]
    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))
    noise_accuracy=1-np.sum(np.array(noise_or_not)[ind_1_sorted[num_remember:]]) / float(len(loss_1_sorted)-num_remember)
    print("epoch", epoch+1,"val_accuarcy", valaccuracy,"test_accuarcy", testaccuracy, "noise_accuracy", noise_accuracy)
mask = np.ones_like(noise_or_not,dtype=np.float32)
mask[ind_1_sorted[num_remember:]]=0

filter_mask = mask

lossfilename = 'lossvalues_20231013_' + args.dataset + '_' + str(args.noise_rate)
np.save(lossfilename, moving_loss_dic)


###############################################################################
#third state
print('third stage')

tau, p, lamb, rho, freq = 0.5, 0.1, 1.2, 1.03, 1 

cleaned_train_dataset = Mask_Select(train_dataset,filter_mask)
train_loader_init = torch.utils.data.DataLoader(cleaned_train_dataset,batch_size=128,num_workers=0,shuffle=True,pin_memory=True)
#train_loader_init = torch.utils.data.DataLoader(dataset=Mask_Select(train_dataset,filter_mask),batch_size=128,num_workers=0,shuffle=True,pin_memory=True)

save_checkpoint= 'middle.pt'
model.load_state_dict(torch.load(save_checkpoint))

ndata=train_dataset.__len__()

criterion3 = GCELoss(num_classes=train_dataset.num_classes)
optimizer3 = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=weight_decay)
scheduler3 = StepLR(optimizer3, gamma=0.1, step_size=25)

for epoch in range(args.phase3_epoch):
    # train models
    globals_loss = 0
    model.train()
    example_loss = np.zeros_like(noise_or_not, dtype=float)
    for i, (images, labels, indexes) in enumerate(train_loader_init):
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)
        logits = model(images)
        loss_3 = calculate_loss(criterion3, logits, labels.squeeze(), norm, lamb, tau, p)
        optimizer3.zero_grad()
        loss_3.backward()
        optimizer3.step()
    scheduler3.step()
    
    if (epoch + 1) % freq == 0:
        lamb = lamb * rho
        
    #with torch.no_grad():
    valaccuracy = accuracy(model, val_loader,)
    testaccuracy = accuracy(model, test_loader)

    
    print ("epoch", epoch+1,"val_accuarcy", valaccuracy,"test_accuarcy", testaccuracy)

