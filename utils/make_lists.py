# Author: Jingxiao Gu
# Baidu Account: Seigato
# Description: Make Data Lists for Lane Segmentation Competition

import os
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np

def encode_labels(color_mask):
    encode_mask = np.zeros((color_mask.shape[0], color_mask.shape[1]))
    encode_dict = {0:[0,249,255,213,206,207,211,208,216,215,218,219,232,202,231,230,228,229,233,212,223,249,255],1:[200,204,209],2:[201,203],
                   3:[217],4:[210],5:[214],6:[220,221,222,224,225,226],
                   7:[205,227,250]}
    for key in encode_dict.keys():
        for color in encode_dict[key]:
            encode_mask[color_mask == color] = key
    return encode_mask


def decode_labels(labels):
    deocde_mask = np.zeros((labels.shape[0], labels.shape[1]), dtype='uint8')
    encode_dict = {0: 0, 1: 200, 2: 201,3: 217,
                   4: 210, 5: 214,6: 220,7: 205}
    for key in encode_dict.keys():
        deocde_mask[labels==key] = encode_dict[key]
    return deocde_mask

def decode_color_labels(labels):
    decode_mask = np.zeros((3, labels.shape[0], labels.shape[1]), dtype='uint8')
    encode_dict = {1: [0, 0, 0], 1: [220, 20, 60], 2: [119, 11, 32],
                   3: [220, 220, 0], 4: [128,64, 128],
                   5: [190,153,153],6: [128, 128, 0],7: [178, 132, 190]}
    for key in encode_dict.keys():
        for i in range(3):
            decode_mask[i][labels==key] = encode_dict[key][i]
    return decode_mask

def verify_labels(labels):
    pixels = [0]
    for x in range(labels.shape[0]):
        for y in range(labels.shape[1]):
            pixel = labels[x, y]
            if pixel not in pixels:
                pixels.append(pixel)
    print('The Labels Has Value:', pixels)

#================================================
# make train & validation lists
#================================================
path = "/root/private/yanzx/data/"
train_data_path = path + "/Image_Data"
test_data_path = path + "/ColorImage"
train_label_path = path + "/Image_Label"
save_path = "/root/private/yanzx/Lane-Segmentation-Solution-For-BaiduAI-Autonomous-Driving-Competition-master/data_list"
train_csv = save_path + "/train.csv"
val_csv = save_path + "/val.csv"
test_csv = save_path + "/test.csv"

def generate_csv(data_path,label_path,csv_path,val_csv_path):
    list_data = []
    list_label = []
    for road in os.listdir(data_path):
        road_data_path = os.path.join(data_path, road,'ColorImage')
        road_label_path = os.path.join(label_path,road.replace('ColorImage','Labels'),'Label')
        for record in os.listdir(road_data_path):
            record_data_path = os.path.join(road_data_path, record)
            record_label_path = os.path.join(road_label_path, record)
            for camera in os.listdir(record_data_path):
                camera_data_path = os.path.join(record_data_path, camera)
                camera_label_path = os.path.join(record_label_path, camera)
                for image in os.listdir(camera_data_path):
                    image_data_path = os.path.join(camera_data_path, image)
                    image_label_path = os.path.join(camera_label_path, image.replace('.jpg', '_bin.png'))
                    list_data.append(image_data_path)
                    list_label.append(image_label_path)
    print(len(list_data), len(list_label))
    image_train,image_val, label_train, label_val = train_test_split(list_data,list_label,test_size=0.4, random_state=0)
    frame = pd.DataFrame({'image': image_train, 'label': label_train})
    frame_shuffle = shuffle(frame)
    frame_shuffle.to_csv(csv_path, index=False)

    frame = pd.DataFrame({'image': image_val, 'label': label_val})
    frame_shuffle = shuffle(frame)
    frame_shuffle.to_csv(val_csv_path, index=False)
generate_csv(train_data_path,train_label_path,train_csv,val_csv)

def generate_test_csv(test_path,csv_path):
    list_data = []
    for image in os.listdir(test_path):
        image_data_path = os.path.join(test_path, image)
        list_data.append(image_data_path)
    print(len(list_data))
    frame = pd.DataFrame({'image': list_data})
    frame_shuffle = shuffle(frame)
    frame_shuffle.to_csv(csv_path, index=False)

generate_test_csv(test_data_path,test_csv)
