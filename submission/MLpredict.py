from keras.models import load_model
import cv2
import numpy as np
import os

img_width, img_height = 150, 100

def preprocessing(img_name=None, i=None):
    if i == 0:
        img_dir = paperDir
    elif i == 1:
        img_dir = rockDir
    else:
        img_dir = scissorDir

    img = cv2.imread(img_dir+img_name) # Open image

    min_HSV = np.array([0, 60, 40], dtype = "uint8")
    max_HSV = np.array([33, 255, 255], dtype = "uint8")
    hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    binaryImg = cv2.inRange(hsvImg, min_HSV, max_HSV)
    masked = cv2.bitwise_and(img, img, mask=binaryImg)

    result = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    result = cv2.resize(result, (img_width, img_height))
    result = np.expand_dims(result, axis=2)
    # print(binaryImg)
    # cv2.imshow('See This', result)
    # cv2.waitKey(0)

    return result

paperDir = 'rockpaperscissors/paper/'
rockDir = 'rockpaperscissors/rock/'
scissorDir = 'rockpaperscissors/scissors/'

paperList = os.listdir(paperDir)
rockList = os.listdir(rockDir)
scissorList = os.listdir(scissorDir)

dirTest = [paperList[650:], rockList[650:], scissorList[650:]]

model = load_model('model-dicoding-razif.h5')
model.compile(loss='binary_crossentropy',
            optimizer='SGD',
            # optimizer='adam',
            metrics=['accuracy'])

print(model.summary())

iTrain = 0
trueThings = 0
total = 0
for dir in dirTest:
    for file in dir:
        newImage = preprocessing(file, iTrain)
        newImage = [newImage]
        newImage = np.array(newImage)
        predict = model.predict_classes(newImage)
        print(file, ' is ', iTrain, ' predicted as ', predict)
        if iTrain == predict:
            trueThings+=1
            total+=1
        else:
            total+=1

    iTrain+=1

print('persentase benar = ', (trueThings/total)*100)