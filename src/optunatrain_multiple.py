import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torchvision
import numpy as np
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from earlystopping import EarlyStopping
import joblib
import optuna
import matplotlib.pyplot as plt

#definiere Pfade und Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_PATH = os.path.abspath('./')
data_path = os.path.join(ROOT_PATH, 'data')
train_data_path = os.path.join(data_path, '224', 'train')
val_data_path = os.path.join(data_path, '224', 'val')
test_data_path = os.path.join(data_path, '224', 'test')
path_AP = os.path.join(data_path, 'evaluate.txt')
#definiere Transformationen
transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0, 0, 0],
                              std=[1, 1, 1]),
         ])
#definiere Datasets
train_data = torchvision.datasets.ImageFolder(train_data_path, transform=transform)
val_data = torchvision.datasets.ImageFolder(val_data_path, transform=transform)
test_data = torchvision.datasets.ImageFolder(test_data_path, transform=transform)
#definiere Augmentationen
transformdata = transforms.Compose(
            [transforms.ToTensor(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomInvert(0.5),
            transforms.GaussianBlur(kernel_size=(5, 5)),
            transforms.ColorJitter(brightness=np.random.random(), contrast=np.random.random(), saturation=np.random.random(), hue=np.random.uniform(0., 0.5)),
            transforms.Normalize(mean=[0, 0, 0],
                                std=[1, 1, 1]),
            ])
transformed_train_data = torchvision.datasets.ImageFolder(train_data_path, transform=transformdata)
#definiere globale Variable f1
f1_max = 0

#Gibt die Parameter zurück, für die Optuna optimieren soll
def get_suggested_params(trial):
    return {
        'opt': trial.suggest_categorical("opt", ['adam', 'sgd', 'rmsprop']),
        'lr': trial.suggest_loguniform('lr', 1e-5, 1e-1),
        'batch_size': trial.suggest_categorical('batch_size', [8, 16, 24, 32]),
        'decay': trial.suggest_loguniform('decay', 1e-8, 1e-4),
        'momentum': trial.suggest_float('momentum', 0.1, 0.9, step=0.1),
        'nEpochs': trial.suggest_int('nEpochs', 50, 300, step=50),
        'dropout': trial.suggest_float('dropout', 0.2, 0.8, step=0.1),
        'hidden_layers': trial.suggest_int('hidden_layers', 128, 1024, step = 128),
        'net': trial.suggest_categorical('net', ['VGG', 'Densenet', 'AlexNet'])
    }

#Gibt die Metriken in der Ausgabe aus und speichert diese in einem File namens evaluate.txt
def ausgabeMetriken(tp, tn, fp, fn, p, n, running_correct, len, loss, mode):
    list = []
    acc = (running_correct / len)
    list.append('\n' + mode + '\n')
    ploss = loss*100.
    pacc = acc*100.
    list.append("Loss: " + str(ploss) + '\n')
    list.append('Accuracy: ' + str(pacc) + '\n')
    if (tp+fp) != 0:
        epoch_precision = tp / (tp+fp)
        list.append('Precision: ' + str(epoch_precision*100.) + '\n')
        if (tp+fn) != 0:
            epoch_recall = tp / (tp+fn)
            list.append('Recall: ' + str(epoch_recall*100.) + '\n')
            f1 = (2*epoch_recall*epoch_precision)/(epoch_precision+epoch_recall)*100.
            f2 = (5*epoch_recall*epoch_precision)/(4*epoch_precision+epoch_recall)*100.
            f3 = (10*epoch_recall*epoch_precision)/(9*epoch_precision+epoch_recall)*100.
            list.append("F3-score: " + str(f3) + "  F2-score: " + str(f2) + "  F1-score: " + str(f1) + '\n')
            fnr = fn / (fn+tp) *100.
            list.append("False negative rate: " + str(fnr) + '\n')
        else:
            f1 = 0
        fdr = fp / (fp+tp) *100.
        list.append("False discovery rate: " + str(fdr) + '\n')
    if (tn+fp != 0):
        tnr = tn /(tn+fp) *100.
        list.append("Specificity: " + str(tnr) + '\n')
        fpr = fp / (fp+tn) *100.
        list.append("Fall-out: " + str(fpr) + '\n')
    if (tn+fn != 0):
        npv = tn / (tn+fn) *100.
        list.append("Negative predictive value: " + str(npv) + '\n')
        falseor = fn / (fn+tn) *100.
        list.append("False omission rate: " + str(falseor) + '\n')
    
    list.append("P: " + str(p) + '\n')
    list.append("N: " + str(n) + '\n')
    list.append("TP: " + str(tp) + '\n')
    list.append("TN: " + str(tn) + '\n')
    list.append("FP: " + str(fp) + '\n')
    list.append("FN: " + str(fn) + '\n')
    ausgabe = "".join(list)
    print(ausgabe)
    
    with open(path_AP, "a+") as myfile:
        myfile.write(ausgabe)
    
    return f1


#Klasse für das Ausführen der Optuna-Optimierung
class MyObjective(object):
    def __init__(self):
        pass

    def __call__(self, trial):
        params = get_suggested_params(trial)
        params = ("Params: Netz: " + str(trial.params['net']) + " Optimizer: " + str(trial.params['opt']) + " Lr: " + str(trial.params['lr']) + " Batchsize: " + str(trial.params['batch_size']) + " Decay: " + str(trial.params['decay']) + 
                    " Momentum: " + str(trial.params['momentum']) + " nEpochs: " + str(trial.params['nEpochs']) + ' Dropout: ' + str(trial.params['dropout']) + " Hidden_layers: " + str(trial.params['hidden_layers']))
        num_workers = 0
        batch_size = trial.params['batch_size']
        Hs = trial.params['hidden_layers']
        dropout = trial.params['dropout']
        n_epochs = trial.params['nEpochs']
        criterion = nn.BCELoss()
        train_on_gpu = torch.cuda.is_available()
        early_stopping = EarlyStopping(patience=17, verbose=True)
        print(params)
        with open(path_AP, "a+") as myfile:
            myfile.write(params)
        
        # prepare data loaders (combine dataset and sampler)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
            num_workers=num_workers, shuffle=True)
        transformed_train_loader = torch.utils.data.DataLoader(transformed_train_data, batch_size=batch_size,
            num_workers=num_workers, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, 
            num_workers=num_workers, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
            num_workers=num_workers, shuffle=True)
    
        if(trial.params['net'] == 'Densenet'):
            model = torchvision.models.densenet121(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False
            model.classifier = nn.Sequential(nn.Linear(1024, Hs),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(Hs, 1),
                                 nn.Sigmoid())
        elif(trial.params['net'] == 'VGG'):
            model = torchvision.models.vgg16(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False
            model.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, Hs),
                nn.ReLU(True),
                nn.Dropout(p=dropout),
                nn.Linear(Hs, Hs),
                nn.ReLU(True),
                nn.Dropout(p=dropout),
                nn.Linear(Hs, 1),
                nn.Sigmoid())
        elif(trial.params['net'] == 'AlexNet'):
            model = torchvision.models.alexnet(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False
            model.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(256 * 6 * 6, Hs),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(Hs, Hs),
                nn.ReLU(inplace=True),
                nn.Linear(Hs, 1),
                nn.Sigmoid())
        
        model.to(device)
                                    
        if trial.params['opt'] == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=trial.params['lr'],
                                momentum=trial.params['momentum'], weight_decay=trial.params['decay'])
        elif trial.params['opt']:
            optimizer = optim.Adam(model.parameters(), lr=trial.params['lr'], weight_decay=trial.params['decay'])
        elif trial.params['opt'] == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=trial.params['lr'],
                                momentum=trial.params['momentum'], weight_decay=trial.params['decay'])

        #Trainiere und Validiere für jede Epoche, Earlystopping ist für f1 eingebaut
        for epoch in range(1, n_epochs+1):
            print("EPOCH: " + str(epoch))
            with open(path_AP, "a+") as myfile:
                myfile.write("Epoch : " + str(epoch) + '\n')
            train(model, train_loader, transformed_train_loader, train_on_gpu, optimizer, criterion)
            f1 = val(model, valid_loader, train_on_gpu, criterion)
            early_stopping(f1, model)
        
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        return f1



    
#Training
def train(model, train_loader, transformed_train_loader, train_on_gpu, optimizer, criterion):
       # Verfolge die Parameter 
        train_loss = 0.0
        train_running_correct = 0
        fp = 0
        fn = 0
        tp = 0
        tn = 0
        p = 0
        n = 0
        model.train()
        #Training für die augmentierten Trainingsdaten
        for data, labels in transformed_train_loader:
            # Tensoren auf Cuda laden, falls sie verfügbar ist
            if train_on_gpu:
                data, labels = data.cuda(), labels.cuda()
            # Die Gradienten des Optimierers auf null setzen
            optimizer.zero_grad()
            # forward pass: Berechnung der Vorhersagen
            outputs = model(data)
            target = outputs.squeeze()
            labels = labels.to(torch.float32)
            t = torch.Tensor([0.5])  # Schwellwert
            target = target.to(device)
            t = t.to(device)
            preds = (target > t).float() * 1
            train_running_correct += (preds == labels).sum().item()
            for i in range(0, len(preds)):
                if (preds[i] == 1.) & (labels[i] == 0.):
                    fp += 1
                elif (preds[i] == 0.) & (labels[i] == 1.):
                    fn += 1
                elif preds[i] == labels[i] == 0.:
                    tn += 1
                elif preds[i] == labels[i] == 1.:
                    tp += 1
                if labels[i] == 1.:
                    p +=1
                else:
                    n +=1
            # Batchloss berechnen
            loss = criterion(target, labels)
            # backward pass: berechne den Gradienten des Loss mit Bezug auf die Modellparameter
            loss.backward()
            # Optimierer-schritt für das Update der Parameter
            optimizer.step()
            # Trainingloss updaten
            train_loss += loss.item()*data.size(0)
    
        #Training für die nicht augmentierten Trainingsdaten
        for data, labels in train_loader:
            if train_on_gpu:
                data, labels = data.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(data)
            target = outputs.squeeze()
            labels = labels.to(torch.float32)
            t = torch.Tensor([0.5])  
            target = target.to(device)
            t = t.to(device)
            preds = (target > t).float() * 1
            train_running_correct += (preds == labels).sum().item()
            for i in range(0, len(preds)):
                if (preds[i] == 1.) & (labels[i] == 0.):
                    fp += 1
                elif (preds[i] == 0.) & (labels[i] == 1.):
                    fn += 1
                elif preds[i] == labels[i] == 0.:
                    tn += 1
                elif preds[i] == labels[i] == 1.:
                    tp += 1
                if labels[i] == 1.:
                    p +=1
                else:
                    n +=1
            loss = criterion(target, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data.size(0)
        
        #Trainingloss insgesamt und Ausgabemetriken berechnen
        train_loss = train_loss/len(train_loader.sampler)
        ausgabeMetriken(tp, tn, fp, fn, p, n, train_running_correct, len(train_loader.sampler), train_loss, "TRAINING")
        
#Validierung
def val(model, valid_loader, train_on_gpu, criterion):
    # Verfolge die Parameter
    valid_loss = 0.0
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    p = 0
    n = 0
    val_running_correct = 0
    model.eval()
    with torch.no_grad():
        for data, labels in valid_loader:
            if train_on_gpu:
                data, labels = data.cuda(), labels.cuda()
            # forward pass: Berechnung der Vorhersagen
            outputs = model(data)
            target = outputs.squeeze()
            labels = labels.to(torch.float32)
            t = torch.Tensor([0.5])  # Schwellwert
            target = target.to(device)
            t = t.to(device)
            preds = (target > t).float() * 1
            val_running_correct += (preds == labels).sum().item()
            for i in range(0, len(preds)):
                if (preds[i] == 1.) & (labels[i] == 0.):
                    fp += 1
                elif (preds[i] == 0.) & (labels[i] == 1.):
                    fn += 1
                elif preds[i] == labels[i] == 0.:
                    tn += 1
                elif preds[i] == labels[i] == 1.:
                    tp += 1
                if labels[i] == 1.:
                    p +=1
                else:
                    n +=1
            # Batchloss berechnen
            loss = criterion(target, labels)
            # Validierungsloss updaten
            valid_loss += loss.item()*data.size(0)
    
    #Validierungsloss insgesamt und Ausgabemetriken berechnen
    valid_loss = valid_loss/len(valid_loader.sampler)
    f1 = ausgabeMetriken(tp, tn, fp, fn, p, n, val_running_correct, len(valid_loader.sampler), valid_loss, "VALIDATION")

    # Speichere das Modell, falls sich der F1-Score verbessert hat (abhängig von globaler Variable f1)
    global f1_max 
    if f1 >= f1_max:
        print('F1-Score increased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        f1_max,
        f1))
        try:
            torch.save(model.state_dict(), 'model_waste.pt')
            print(f'Model saved successfully to model_waste.pt')
        except:
            print(f'ERROR saving model!')
        f1_max = f1
    
    return f1
    
    
    

def __main__():
    #Erstelle die Optuna-Studie und führe sie aus
    sampler = optuna.samplers.TPESampler(n_startup_trials=25)
    pruner = optuna.pruners.HyperbandPruner()

    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(MyObjective(), n_trials = 30, show_progress_bar=True)

    best_trial = study.best_trial
    for key, value in best_trial.params.items():
        print("{}: {}".format(key, value))
        return
    
    joblib.dump(study, "study.pkl")
    return

if __name__=='__main__':
    __main__()
