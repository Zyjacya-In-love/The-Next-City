# -*- coding: utf-8 -*-
"""Originated in Google Colaboratory.
# use Places365 to get place pictures' label and score
"""
import codecs
from os import path
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
import torch
from torch.autograd import Variable as V
from torch.nn import functional as F
import torchvision.models as models
import torchvision.datasets as dset
from torchvision import transforms as trn
from PIL import Image
import os
import zipfile
import shutil

""" # check if model file exist """
def check_model_file(model_file):
    if not os.access(model_file, os.W_OK):
        print("error : not exist " + model_file + "\nyou can download it from \n" + '      http://places2.csail.mit.edu/models_places365/' + model_file)
        exit()

""" # pre initialize  load the image transformer 、 load the class label """
def init(folder_name):
    # load the image transformer
    centre_crop = trn.Compose([
        trn.Resize((256, 256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # load the class label
    file_name = 'categories_places365.txt'
    if not os.access(file_name, os.W_OK):
        print("error : not exist " + file_name + "\nyou can download it from \n" + '      https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt')
        exit()
    classes = list()
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)
    # load picture
    folder_dataset = dset.ImageFolder(folder_name)  # only need images
    # idx_label = {y: os.path.basename(os.path.dirname(x)) for (x, y) in folder_dataset.imgs}
    # print(idx_label)
    # tmp = {y: os.path.dirname(x) for (x, y) in folder_dataset.imgs}
    # print(folder_dataset.imgs)
    # print(tmp)
    return centre_crop, classes, folder_dataset


""" # save the result csv """
# load class label and image transformer && get picture in data
folder_name = "data"
centre_crop, classes, folder_dataset = init(folder_name)
# PyTorch Places365 models: AlexNet, ResNet18, ResNet50
# architecture = ['resnet18', 'resnet50', 'alexnet']
architecture = ['resnet50']
for arch in architecture:
    # load the pre-trained weights
    model_file = '%s_places365.pth.tar' % arch
    check_model_file(model_file)
    model = models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    # load every image
    for img_name, _ in folder_dataset.imgs:
        img = Image.open(img_name)
        input_img = V(centre_crop(img).unsqueeze(0))
        logit = model.forward(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        path = arch + "_result_csv\\" + img_name
        print(path)
        path = os.path.dirname(path)
        print(path)
        # path.replace('', '/');
        # print(path)

        if not os.path.exists(path):  # check if exist path
            os.makedirs(path)  # if not create the folder
        csv_file = arch + "_result_csv\\" + img_name[0:-3] + "csv"
        print(img_name[0:-3])
        print(csv_file);
        with codecs.open(csv_file, 'w', 'utf-8') as f:
            for i in range(0, 10):
                f.write('{:.3f}:{}'.format(probs[i], classes[idx[i]]) + ',')
            f.write('\r\n')
        print(csv_file + "  has Done")