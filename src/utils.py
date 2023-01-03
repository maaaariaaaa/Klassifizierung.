import json
import os
import torch
import random
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as FT
from xml.dom import minidom
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from numpy import asarray
from torchvision.io import read_image
import cv2
import xml.etree.ElementTree as ET

#Quelle: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py
def save_checkpoint(epoch, model, optimizer):
    """
    Save model checkpoint.

    :param epoch: epoch number
    :param model: model
    :param optimizer: optimizer
    """
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer}
    filename = 'checkpoint_ssd300.pth.tar'
    torch.save(state, filename)

#Quelle: https://stackoverflow.com/questions/27152904/calculate-overlapped-area-between-two-rectangles
def area(a, b):  
    """
    returns None, if rectangles dont intersect, otherwise returns the area of intersection

    :param a: first rectangle
    :param b: second rectangle
    """
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx>=0) and (dy>=0):
        return dx*dy
    

def create_dirs(dirs):
    if isinstance(dirs, list):
        for cur_dir in dirs:
            if not os.path.exists(cur_dir):
                os.makedirs(cur_dir)
                print(f'Directory {cur_dir} created')
    else:
        if not os.path.exists(dirs):
            os.makedirs(dirs)
            print(f'Directory {dirs} created')

#Visualisiere  Bild ggf. mit bbox
def visualize(path_jpg, path_xml):
    tree = ET.parse(path_xml)
    root = tree.getroot()
    image_bbox = read_image(path_jpg)
    imgbb = Image.open(path_jpg)
    numpydata = asarray(imgbb)
    hasbox = False
    for object in root.findall('object'):
        box = object.find('bndbox')
        print("a")
        x1 = int(box.find('xmin').text)
        x2 = int(box.find('xmax').text)
        y1 = int(box.find('ymin').text)
        y2 = int(box.find('ymax').text)
        print(x1, y2, x2, y2)
        image_bbox = cv2.rectangle(numpydata,(x1,y1),(x2,y2),(255,0,0),6)
        hasbox = True
    if hasbox:
        plt.imshow(image_bbox)
        plt.show()

    else:
        im = Image.open(path_jpg)
        im.show()
