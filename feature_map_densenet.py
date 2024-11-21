import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
import time
import os
import sys
from customdataset_3 import CustomImageDataset
from confusion_matrix_210825 import confusion_matrix
from efficientnet_pytorch import EfficientNet
from pytorch_pretrained_vit import ViT
import torch, torchvision
from torchvision import transforms
from PIL import Image

transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

device = torch.device('cpu')
model = models.densenet161()
num_classes = 24
actual_class = [[0 for j in range(num_classes)] for k in range(num_classes)]
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, num_classes)

model.load_state_dict(torch.load("best_checkpoint_densenet.pth", map_location='cpu'))
model.eval()

model_weights = []
conv_layers = []
model_children = list(model.children())
counter = 0
for i in range(len(model_children)):
    if type(model_children[i]) == nn.Conv2d:
        counter += 1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            if type(model_children[i][j]) == nn.Conv2d:
                counter += 1
                model_weights.append(model_children[i][j].weight)
                conv_layers.append(model_children[i][j])
            elif str(type(model_children[i][j])) == "<class 'torchvision.models.densenet._DenseBlock'>":
                conv_layers.append(model_children[i][j])
            else:
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter += 1
                        model_weights.append(child.weight)
                        conv_layers.append(child)

image = Image.open(str('test.JPG')).convert('RGB')
image = transforms_test(image).unsqueeze(0).to(device)

outputs = []
names = []
for layer in conv_layers[0:]:
    image = layer(image)
    outputs.append(image)
    names.append(str(layer))

processed = []
for feature_map in outputs:
    feature_map = feature_map.squeeze(0)
    gray_scale = torch.sum(feature_map, 0)
    gray_scale = gray_scale / feature_map.shape[0]
    processed.append(gray_scale.data.cpu().numpy())

fig = plt.figure(figsize=(30, 50))
for i in range(len(processed)):
    a = fig.add_subplot(5, 4, i + 1)
    imgplot = plt.imshow(processed[i])
    a.axis("off")
    a.set_title(names[i].split('(')[0], fontsize=30)
plt.savefig(str('feature_maps_densenet.jpg'), bbox_inches='tight')