import os
from unicodedata import name
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import random
from matplotlib import pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import  train_test_split
from sklearn.utils.class_weight import compute_class_weight

from patchify import patchify

import segmentation_models as sm

from tensorflow.keras.utils import to_categorical

from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda

from unet_model import unet_model

data_dir = 'Semantic_segmentation_dataset/'
scaler = MinMaxScaler()
patch_size = 256

image_dataset = []  
mask_dataset = []  

weights = [0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666]

def rgb_to_label(label, class_dict):

    Building = np.array(tuple(int(class_dict['color'][0].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)))
    Land = np.array(tuple(int(class_dict['color'][1].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)))
    Road = np.array(tuple(int(class_dict['color'][2].lstrip('#') [i:i+2], 16) for i in (0, 2, 4)))
    Vegetation = np.array(tuple(int(class_dict['color'][3].lstrip('#') [i:i+2], 16) for i in (0, 2, 4)))
    Water = np.array(tuple(int(class_dict['color'][4].lstrip('#') [i:i+2], 16) for i in (0, 2, 4)))
    Unlabeled = np.array(tuple(int(class_dict['color'][5].lstrip('#') [i:i+2], 16) for i in (0, 2, 4)))

    label_seg = np.zeros(label.shape,dtype=np.uint8)

    label_seg [np.all(label == Building,axis=-1)] = 0
    label_seg [np.all(label==Land,axis=-1)] = 1
    label_seg [np.all(label==Road,axis=-1)] = 2
    label_seg [np.all(label==Vegetation,axis=-1)] = 3
    label_seg [np.all(label==Water,axis=-1)] = 4
    label_seg [np.all(label==Unlabeled,axis=-1)] = 5

    label_seg = label_seg[:,:,0]

    return label_seg


def main():
    
    for path, subdirs, files in os.walk(data_dir):
        dirname = path.split(os.path.sep)[-1]
        if dirname == 'images':
            images = os.listdir(path)
            for i, image_name in enumerate(images):  
                if image_name.endswith(".jpg"):
                    
                    image = cv2.imread(path+"/"+image_name, 1) 
                    
                    SIZE_X = (image.shape[1]//patch_size)*patch_size 
                    SIZE_Y = (image.shape[0]//patch_size)*patch_size 
                    
                    image = Image.fromarray(image)
                    image = image.crop((0 ,0, SIZE_X, SIZE_Y))
                    image = np.array(image)

                    patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)
                    for i in range(patches_img.shape[0]):
                        for j in range(patches_img.shape[1]):
                            
                            single_patch_img = patches_img[i,j,:,:]
                            single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
                            single_patch_img = single_patch_img[0]
                            
                            image_dataset.append(single_patch_img)


    for path, subdirs, files in os.walk(data_dir):
    
        dirname = path.split(os.path.sep)[-1]
        if dirname == 'masks':  
            masks = os.listdir(path) 
            for i, mask_name in enumerate(masks):  
                if mask_name.endswith(".png"):
                
                    mask = cv2.imread(path+"/"+mask_name, 1)
                    mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
                    
                    SIZE_X = (mask.shape[1]//patch_size)*patch_size 
                    SIZE_Y = (mask.shape[0]//patch_size)*patch_size
                    
                    mask = Image.fromarray(mask)
                    mask = mask.crop((0 ,0, SIZE_X, SIZE_Y))  
                    mask = np.array(mask)             
        
                    patches_mask = patchify(mask, (patch_size, patch_size, 3), step=patch_size)
            
                    for i in range(patches_mask.shape[0]):
                        for j in range(patches_mask.shape[1]):
                            
                            single_patch_mask = patches_mask[i,j,:,:]
                            single_patch_mask = single_patch_mask[0]                            
                            mask_dataset.append(single_patch_mask) 
    

    image_dataset = np.array(image_dataset)
    mask_dataset =  np.array(mask_dataset)

    class_dict = pd.read_csv("class.csv", index_col=False, skipinitialspace=True)

    label = single_patch_mask

    rgb_to_label(label ,class_dict)

    labels = []
    for i in range(mask_dataset.shape[0]):
        label = rgb_to_label(mask_dataset[i])
        labels.append(label)

    labels = np.array(labels)   
    labels = np.expand_dims(labels, axis=3)

    n_classes = len(np.unique(labels))

    categorical_labels = to_categorical(labels, num_classes=n_classes)

    X_train, X_test, y_train, y_test = train_test_split(image_dataset, categorical_labels, test_size = 0.10, random_state = 42)

    IMG_HEIGHT = X_train.shape[1]
    IMG_WIDTH  = X_train.shape[2]
    IMG_CHANNELS = X_train.shape[3]

    dice_loss = sm.losses.DiceLoss(class_weights=weights)
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)
    
    """
    model = unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)

    model = unet_model(n_classes, X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model.compile(optimizer='adam', loss=total_loss, metrics='accuracy')
    model.summary()
    """

    model = load_model("model_100Epochs.h5", custom_objects={'dice_loss_plus_1focal_loss' : total_loss})

    y_pred=model.predict(X_test)
    y_pred_argmax=np.argmax(y_pred, axis=3)
    y_test_argmax=np.argmax(y_test, axis=3)


    test_img_number = random.randint(0, len(X_test))
    test_img = X_test[test_img_number]
    ground_truth=y_test_argmax[test_img_number]
    test_img_input=np.expand_dims(test_img, 0)
    prediction = (model.predict(test_img_input))
    predicted_img=np.argmax(prediction, axis=3)[0,:,:]

    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.title('Test Image')
    plt.imshow(test_img)
    plt.subplot(232)
    plt.title('Test Label')
    plt.imshow(ground_truth)
    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(predicted_img)
    plt.show()


if __name__ == "__main__":
    main()

