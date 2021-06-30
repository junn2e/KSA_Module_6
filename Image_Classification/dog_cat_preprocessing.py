import shutil
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

img_list = search("dog_cat_test_data/train")
val_ratio = 0.25

data_set = './data_dog_cat/'
subdirs = ['train/', 'val/']
labeldirs = ['dogs/', 'cats/']
for sub in subdirs:
    for labelsub in labeldirs:
        dir_name = data_set + sub + labelsub
        # print(dir_name)
        createFolder(dir_name)

for i, img in enumerate(img_list):
    dst_dir = 'train/'
    if random() < val_ratio:
        dst_dir = 'val/'
    if img_list[i].split('/')[-1].find('cat') >= 0:
        dst = data_set + dst_dir + 'cats/' + img_list[i].split('/')[-1]
        shutil.copy(img, dst)
    elif img_list[i].split('/')[-1].find('dog') >= 0:
        dst = data_set + dst_dir + 'dogs/' + img_list[i].split('/')[-1]
        shutil.copy(img, dst)