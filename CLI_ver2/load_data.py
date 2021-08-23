
import os
import numpy as np
from PIL import Image
import argparse

#Re-used from the implementation of paper "Overlearning Reveals Sensitive Attributes"

BASE_DIR = './data/UTKFace/'

if not os.path.exists(BASE_DIR):
    try:        
        os.makedirs(BASE_DIR)
    except OSError:
        pass


def resize_utk(dataFolder):

    UTK_DIR = dataFolder
    if not os.path.exists(UTK_DIR):
        print('UTKFace data directory does not exist. Kindly provide the correct path')
        exit()

    ages = []
    genders = []
    races = []
    imgs = []
    for filename in os.listdir(UTK_DIR):
        img_path = os.path.join(UTK_DIR, filename)
        attrs = filename.split('_')[:3]
        if attrs[2] not in {'0', '1', '2', '3', '4'}:
            print(filename)
            continue

        ages.append(int(attrs[0]))
        genders.append(int(attrs[1]))
        races.append(int(attrs[2]))

        image = Image.open(img_path)
        img_resized = image.resize(size=(50, 50))
        imgs.append(np.asarray(img_resized))

    imgs = np.asarray(imgs)
    ages = np.asarray(ages)
    genders = np.asarray(genders)
    races = np.asarray(races)
    np.savez(BASE_DIR + 'utk_resize.npz', imgs=imgs, ages=ages, genders=genders, races=races)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataPath', default='./UTKFace', type=str, help='Data Path')
    args = parser.parse_args()
    resize_utk(args.dataPath)