import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import time
import os
import sys
from customdataset_kfold import CustomImageDataset
from confusion_matrix_kfold import confusion_matrix
from efficientnet_pytorch import EfficientNet
from pytorch_pretrained_vit import ViT
import seaborn as sns
import pandas as pd
from PIL import Image
import torch
import timm
import requests
import torchvision.transforms as transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def run(train_loader, val_loader, device, optimizer, net, criterion, scheduler, train_set, val_set, num_classes,
        model_name, actual_class, ex_path, model):
    setSysoutpath(ex_path)
    now = time.localtime()

    acc_max = 0
    f1_max = 0
    epoch = 100

    epochs = []
    train_acc = []
    validation_acc = []
    f1_list = []

    strat_time = time.time()
    for epoch in range(epoch):
        epochs.append(epoch)
        epoch_start = time.localtime()

        print("Start train.py")
        print("[%04d/%02d/%02d %02d:%02d:%02d] Plant_dataset %s Train_start" % (
            now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec, model_name))

        class_names = train_set.class_names
        print(f"class names: {class_names}")

        print("[%04d/%02d/%02d %02d:%02d:%02d] %dth epoch_started" % (
            epoch_start.tm_year, epoch_start.tm_mon, epoch_start.tm_mday, epoch_start.tm_hour, epoch_start.tm_min,
            epoch_start.tm_sec, epoch + 1))

        e_loss, e_acc = doTrain(train_loader, net, device, optimizer, criterion, scheduler, epoch, train_set)
        print('train loss ', e_loss, ' epoch: ', epoch + 1)
        print('train acc ', e_acc, ' epoch: ', epoch + 1)

        train_acc.append(e_acc)

        end = time.localtime()
        print("[%04d/%02d/%02d %02d:%02d:%02d] Training_finished" % (
            end.tm_year, end.tm_mon, end.tm_mday, end.tm_hour, end.tm_min, end.tm_sec))
        print("")

        print("Start validation.py")
        print("[%04d/%02d/%02d %02d:%02d:%02d] Plant_dataset %s Validation_start" % (
            now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec, model_name))

        class_names = val_set.class_names
        print(f"class names: {class_names}")

        print("[%04d/%02d/%02d %02d:%02d:%02d] %dth epoch_validation" % (
            epoch_start.tm_year, epoch_start.tm_mon, epoch_start.tm_mday, epoch_start.tm_hour, epoch_start.tm_min,
            epoch_start.tm_sec, epoch + 1))

        e_acc, f1_score = doValidation(val_loader, device, net, actual_class, model_name, val_set, num_classes, model,
                                       f1_max, ex_path)

        validation_acc.append(e_acc)

        if f1_max < f1_score:
            f1_max = max(f1_max, f1_score)
            save_checkpoint(net, f'./{ex_path}/best_checkpoint_{model_name}.pth')

        acc_max = max(acc_max, e_acc)

        f1_list.append(f1_score)

    total_time = time.time() - strat_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print(f'Best {model_name} Acc :', acc_max)
    print(f'Best F1 score :', f1_max)

    figure(epochs, train_acc, f'./{ex_path}/result_train_{ex_path}.jpg')
    figure(epochs, validation_acc, f'./{ex_path}/result_valid_{ex_path}.jpg')
    figure(epochs, f1_list, f'./{ex_path}/result_f1_{ex_path}.jpg')

    sys.stdout.flush()


def figure(x_list, y_list, save_path):
    plt.clf()
    plt.plot(x_list, y_list)
    plt.savefig(save_path, bbox_inches='tight')


def doTrain(train_loader, net, device, optimizer, criterion, scheduler, epoch, train_set):
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    total_loss = 0
    total_correct = 0
    net.train()

    for i, data in loop:
        inputs = data['image'].to(device)
        labels = data['label'].to(device)

        optimizer.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs, 1)

        train_loss = loss.item()
        train_correct = torch.sum(predicted == labels.data).item()

        total_loss += train_loss
        total_correct += train_correct

    now = time.localtime()
    current_time = "%04d/%02d/%02d %02d:%02d:%02d" % (
        now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    loop.set_description(f"[{current_time}] Epoch [{epoch + 1}/{epoch}]")

    loop.set_postfix(loss=train_loss, acc=train_correct / len(labels))

    scheduler.step()

    e_loss = total_loss / len(train_set)

    e_acc = total_correct / len(train_set)

    return e_loss, e_acc


def doValidation(val_loader, device, net, actual_class, model_name, val_set, num_classes, model, current_f1_score,
                 ex_path):
    net.eval()  # Convert the model to validation mode

    class_names = val_set.class_names

    total_correct, actual_data, predicted, image_path_list = process(val_loader,
                                                                     device,
                                                                     net,
                                                                     actual_class)

    e_acc = total_correct / len(val_set)

    f1_score = getF1Score(predicted, actual_data, model_name, actual_class, num_classes, class_names, model,
                          current_f1_score, ex_path)

    end = time.localtime()
    print("[%04d/%02d/%02d %02d:%02d:%02d] Validation_finished" % (
        end.tm_year, end.tm_mon, end.tm_mday, end.tm_hour, end.tm_min, end.tm_sec))
    print("-----------------------------------------------------------------------------------------------------")
    print(" ")

    return e_acc, f1_score


def process(val_loader, device, net, actual_class):
    with torch.no_grad():
        total_correct = 0
        loop = tqdm(enumerate(val_loader), total=len(val_loader))
        image_path_list = []
        actual_data = []
        predict_data = []

        for i, data in loop:
            inputs = data['image'].to(device)
            labels = data['label'].to(device)
            image_path = data['image_path']

            outputs = net(inputs)

            _, predicted = torch.max(outputs, 1)

            val_correct = torch.sum(predicted == labels.data).item()

            total_correct += val_correct

            labels_list = labels.tolist()
            predicted_label_list = predicted.tolist()

            actual_class[labels_list[0]][predicted_label_list[0]] += 1

            actual_data.extend(labels.tolist())
            predict_data.extend(predicted.tolist())

            image_path_list.extend(image_path)

    return total_correct, actual_data, predict_data, image_path_list


def setSysoutpath(ex_path):
    if not os.path.isdir(f'./{ex_path}'):
        os.makedirs(f'./{ex_path}')
    sys.stdout = open(f'./{ex_path}/output_{ex_path}.txt', 'w', encoding='utf8')


def save_checkpoint(model, filename):
    torch.save(model.state_dict(), filename)


def sklearnF1Score(y_pred_list, my_data, model_name, class_names, model, current_f1_score, now_f1_score, ex_path):
    y_pred_list = [a for a in y_pred_list]
    my_data = [a for a in my_data]

    my_data = torch.tensor(my_data)
    y_pred_list = torch.tensor(y_pred_list)

    my_data = torch.flatten(my_data)
    y_pred_list = torch.flatten(y_pred_list)
    f1_score = classification_report(my_data, y_pred_list)
    print(f"***************{model_name} F-1 Score*******************")
    print("")
    print(f1_score)


def getF1Score(predict_list, actual_data, model_name, actual_class, num_classes, class_names, model, current_f1_score,
               ex_path):
    Average_precision, Average_recall, Accuracy, F1_Score = confusion_matrix(actual_class, num_classes,
                                                                             class_names)
    sklearnF1Score(predict_list, actual_data, model_name, class_names, model, current_f1_score, F1_Score, ex_path)

    return F1_Score


# VGG16 Algorithm
def runVgg16(train_loader, validation_loader, train_set, validation_set, kfold):
    model = models.vgg16(pretrained=True)

    num_features = model.classifier[6].in_features

    actual_class = [[0 for j in range(24)] for k in range(24)]
    features = list(model.classifier.children())[:-1]
    features.extend([nn.Linear(num_features, 24)])
    model.classifier = nn.Sequential(*features)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)

    run(train_loader, validation_loader, device, optimizer, net, criterion, scheduler, train_set, validation_set, 24,
        "vgg16", actual_class, f"{kfold}Plantdataset vgg16", model)


def runResnet(train_loader, validation_loader, train_set, validation_set, kfold):
    model = models.resnet50(pretrained=True)

    num_classes = 24
    actual_class = [[0 for j in range(num_classes)] for k in range(num_classes)]
    num_classes = train_set.num_classes
    class_names = train_set.class_names
    print(class_names)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
    run(train_loader, validation_loader, device, optimizer, net, criterion, scheduler, train_set, validation_set,
        num_classes, "resnet50", actual_class, f"{kfold}Plantdataset resnet50", model)


# Densenet161 Algorithm
def runDensenet(train_loader, validation_loader, train_set, validation_set, kfold):
    model = models.densenet161(pretrained=True)

    num_classes = 24
    actual_class = [[0 for j in range(num_classes)] for k in range(num_classes)]
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)

    run(train_loader, validation_loader, device, optimizer, net, criterion, scheduler, train_set, validation_set,
        num_classes, "densenet", actual_class, f"{kfold}Plantdataset densenet161", model)


# Efficientnet Algorithm
def runEfficientNet(train_loader, validation_loader, train_set, validation_set, kfold):
    num_classes = 24
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)

    actual_class = [[0 for j in range(num_classes)] for k in range(num_classes)]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)

    run(train_loader, validation_loader, device, optimizer, net, criterion, scheduler, train_set, validation_set,
        num_classes, "efficientnet", actual_class, f"{kfold}Plantdataset efficientnet-b0", model)


# VIT Algorithm
def runVIT(train_loader, validation_loader, train_set, validation_set, kfold):
    num_classes = 24
    model = ViT('B_16_imagenet1k', pretrained=True, num_classes=num_classes, image_size=224)

    actual_class = [[0 for j in range(num_classes)] for k in range(num_classes)]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)

    run(train_loader, validation_loader, device, optimizer, net, criterion, scheduler, train_set, validation_set,
        num_classes, "vit", actual_class, f"{kfold}Plantdataset VIT_B_16_imagenet1k", model)


# DeiT Algorithm
def runDeiT(train_loader, validation_loader, train_set, validation_set, kfold):
    num_classes = 24
    model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)

    actual_class = [[0 for j in range(num_classes)] for k in range(num_classes)]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)

    run(train_loader, validation_loader, device, optimizer, net, criterion, scheduler, train_set, validation_set,
        num_classes, "deit", actual_class, f"{kfold}Plantdataset Deit", model)


def main():
    trans_train = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.GaussianBlur(kernel_size=(7, 13), sigma=(0.5, 0.5)),
                                      transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.ColorJitter(brightness=(0.35, 1), contrast=(0.35, 2),
                                                             saturation=(0.35, 2), hue=(-0.01, 0.1)),
                                      transforms.RandomResizedCrop((224, 224), scale=(0.1, 1), ratio=(0.5, 2)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trans_validation = transforms.Compose([transforms.Resize((224, 224)),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    kfold = 5
    for fold in range(kfold):
        train_data_set = CustomImageDataset(
            data_set_path="../train",
            transforms=trans_train,
            kfold_size=kfold,
            is_kfold=True,
            kfold_index=fold,
            is_train=True)
        val_data_set = CustomImageDataset(
            data_set_path="../val",
            transforms=trans_validation,
            kfold_size=kfold,
            is_kfold=True,
            kfold_index=fold,
            is_train=False)
        train_loader = DataLoader(train_data_set, num_workers=4, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_data_set, num_workers=4, batch_size=64, shuffle=False)

        runVgg16(train_loader, val_loader, train_data_set, val_data_set, fold)
        runResnet(train_loader, val_loader, train_data_set, val_data_set, fold)
        runDensenet(train_loader, val_loader, train_data_set, val_data_set, fold)
        runEfficientNet(train_loader, val_loader, train_data_set, val_data_set, fold)
        runVIT(train_loader, val_loader, train_data_set, val_data_set, fold)
        runDeiT(train_loader, val_loader, train_data_set, val_data_set, fold)


if __name__ == '__main__':
    main()
