import torch
import os
from torchvision import transforms
import torchvision
import torch.nn as nn
from optunatrain_multiple import val
import numpy as np
from PIL import Image

#Pfade angeben
ROOT_PATH = os.path.abspath('./')
data_path = os.path.join(ROOT_PATH, 'data')
test_data_path = os.path.join(data_path, '224', 'test')
#Transformationen und Variablen angeben
transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0, 0, 0],
                              std=[1, 1, 1]),
         ])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dropout=0.7
Hs=128*3

#Testloader erstellen
test_data = torchvision.datasets.ImageFolder(test_data_path, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, 
            num_workers=0, shuffle=False)

#Modell laden
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
model.load_state_dict(torch.load(os.path.join(ROOT_PATH, 'data', 'model_waste.pt'), map_location=device))
model.eval()
model.to(device)

#Hiermit kann man eine Vorhersage für ein spezielles Bild treffen
# image = Image.open(os.path.join(test_data_path, 'rubbish', 'batch_01_frame_4_120682.jpg'))
# image = transform(image)
# image = image[None,:,:,:]
# print(image.shape)
# output = model(image)
# t = torch.Tensor([0.5])
# pred = (output > t).float() * 1
# print(pred)

#finales Evaluieren auf dem Test-Datensatz, Ausgabe der Metriken
criterion = nn.BCELoss()
val(model, test_loader, False,  criterion)
