import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


from scipy import interp
from itertools import cycle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def accuracy(model, loader):
    
    model.eval()
    correct = 0.
    total = 0.
    for x, y, _ in loader:
        x, y = x.to(device), y.squeeze().to(device)
        z = model(x)
        probs = F.softmax(z, dim=1) 
        pred = torch.argmax(probs, 1)
        total += y.size(0)
        correct += (pred==y).sum().item()

    acc = float(correct) / float(total)
    return acc




def roc(model, loader):
    
    model.eval()
    
    allscorelist = []
    alllabellist = []


    for images, labels, _ in loader:
        
        images, labels = images.to(device), labels.to(device)
        labels = labels.squeeze().long()
        label_one_hot = F.one_hot(labels, loader.dataset.num_classes).float().to(device)
        
        outputs = model(images)
        #predictions = torch.max(outputs, 1)[1].to(device)
        batchscorelist = outputs.tolist()
        batchlabellist = label_one_hot.tolist()
        allscorelist.extend(batchscorelist)
        alllabellist.extend(batchlabellist)
        #correct += (predictions == labels).sum()
        #total += len(labels)
    
    allscorearray = np.array(allscorelist)#.float()
    alllabelarray = np.array(alllabellist)#.float()
        
    
    fpr_dict = dict()
    tpr_dict = dict()
    roc_auc_dict = dict()
    for i in range(loader.dataset.num_classes):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(alllabelarray[:, i], allscorearray[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
    
    
    # micro AUC
    fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(alllabelarray.ravel(), allscorearray.ravel())
    roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])
    

    # macro AUR
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(loader.dataset.num_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(loader.dataset.num_classes):
        mean_tpr += interp(all_fpr, fpr_dict[i], tpr_dict[i])
    # Finally average it and compute AUC
    mean_tpr /= loader.dataset.num_classes
    fpr_dict["macro"] = all_fpr
    tpr_dict["macro"] = mean_tpr
    roc_auc_dict["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])
    
    
    return fpr_dict, tpr_dict, roc_auc_dict  
    

def presenf1cfsmtx(model, loader):
    model.eval()
    y_true_list = []
    y_pred_list = []
    for x, y, _ in loader:
        x, y = x.to(device), y.squeeze().to(device)
        z = model(x)
        probs = F.softmax(z, dim=1) 
        pred = torch.argmax(probs, 1)
        
        y_true_list.extend(y.tolist())
        y_pred_list.extend(pred.tolist())
        
    precision = precision_score(y_true_list, y_pred_list, average='weighted')
    recall = recall_score(y_true_list, y_pred_list, average='weighted')
    f1 = f1_score(y_true_list, y_pred_list, average='weighted')
    cfsmtx = confusion_matrix(y_true_list, y_pred_list)

    return precision, recall, f1, y_true_list, y_pred_list, cfsmtx

