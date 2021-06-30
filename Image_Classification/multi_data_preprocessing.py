import shutil
import numpy as np
from random import seed
from random import random
import os


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def search(dirname):
    img_list = []
    for (path, dir, files) in os.walk(dirname):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.jpg':
                img_list.append(path+'/'+filename)
    return img_list


label = np.array(['dog', 'cat', 'car', 'airplane', 'person', 'flower', 'fruit', 'motorbike'])


img_list = []
for i in range(8):
    img_list += search("8-multi-class_data/natural_images/"+str(label[i]))
val_ratio = 0.2
test_ratio = 0.1


data_set = './8_multi_data/'
subdirs = ['train/', 'val/', 'test/']
labeldirs = ['dogs/', 'cats/', 'cars/', 'airplanes/', 'people/', 'flowers/', 'fruits/', 'motorbikes/']

for sub in subdirs:
    for labelsub in labeldirs:
        dir_name = data_set + sub + labelsub
        # print(dir_name)
        createFolder(dir_name)

for i, img in enumerate(img_list):
    dst_dir = 'train/'
    temp = random()
    if temp < test_ratio:
        dst_dir = 'test/'
    elif temp < val_ratio:
        dst_dir = 'val/'
    if img_list[i].split('/')[-1].find('cat') >= 0:
        dst = data_set + dst_dir + 'cats/' + img_list[i].split('/')[-1]
        shutil.copy(img, dst)
    elif img_list[i].split('/')[-1].find('dog') >= 0:
        dst = data_set + dst_dir + 'dogs/' + img_list[i].split('/')[-1]
        shutil.copy(img, dst)
    elif img_list[i].split('/')[-1].find('car') >= 0:
        dst = data_set + dst_dir + 'cars/' + img_list[i].split('/')[-1]
        shutil.copy(img, dst)
    elif img_list[i].split('/')[-1].find('airplane') >= 0:
        dst = data_set + dst_dir + 'airplanes/' + img_list[i].split('/')[-1]
        shutil.copy(img, dst)
    elif img_list[i].split('/')[-1].find('person') >= 0:
        dst = data_set + dst_dir + 'people/' + img_list[i].split('/')[-1]
        shutil.copy(img, dst)
    elif img_list[i].split('/')[-1].find('flower') >= 0:
        dst = data_set + dst_dir + 'flowers/' + img_list[i].split('/')[-1]
        shutil.copy(img, dst)
    elif img_list[i].split('/')[-1].find('fruit') >= 0:
        dst = data_set + dst_dir + 'fruits/' + img_list[i].split('/')[-1]
        shutil.copy(img, dst)
    elif img_list[i].split('/')[-1].find('motorbike') >= 0:
        dst = data_set + dst_dir + 'motorbikes/' + img_list[i].split('/')[-1]
        shutil.copy(img, dst)