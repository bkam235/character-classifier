import os
from shutil import copyfile
from sklearn.model_selection import train_test_split
import shutil
from subprocess import call
import numpy as np
import cv2

# remove all data directories if they exist
shutil.rmtree('train_data', ignore_errors=True)
shutil.rmtree('train_data_sample', ignore_errors=True)
shutil.rmtree('validation_data', ignore_errors=True)
shutil.rmtree('test_data', ignore_errors=True)

# copy out to train_data
data_dir = "container/app/out"
call(["cp", "-r", data_dir, "train_data/"])
os.makedirs("validation_data")

# create train_data sample for inspection
os.makedirs("train_data_sample")
files = os.listdir("train_data")
idx = np.random.randint(0, len(files), 100)
for id in idx:
    file = files[id]
    shutil.copy(os.path.join("train_data", file), os.path.join("train_data_sample", file))

# move all files
img_files = os.listdir("train_data")
train, validation = train_test_split(img_files,test_size=0.2, random_state=42)

for f in validation:
    copyfile(os.path.join("train_data", f), os.path.join("validation_data", f))
    os.remove(os.path.join("train_data", f))

# move into subdirectories by char
chars = [f.split("_")[0] for f in img_files]
chars = list(set(chars))

for char in chars:
    os.makedirs(os.path.join("train_data", char), exist_ok=True)
    os.makedirs(os.path.join("validation_data", char), exist_ok=True)

for char in chars:
    print(char)
    train_files = os.listdir("train_data")
    files = [f for f in train_files if f.startswith(str(char)+"_")]
    for f in files:
        shutil.move(os.path.join("train_data", f), os.path.join("train_data", str(char), f))
        
    val_files = os.listdir("validation_data")
    files = [f for f in val_files if f.startswith(str(char)+"_")]
    for f in files:
        shutil.move(os.path.join("validation_data", f), os.path.join("validation_data", str(char), f))

# create test data
os.makedirs("test_data")
test_file = "IBAN.jpg"

img = cv2.imread(test_file)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 

kernel_blackhat = np.ones((20,20),np.uint8) 
blackhat = cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,kernel_blackhat) 

#ret,thresh = cv2.threshold(blackhat,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret,thresh = cv2.threshold(blackhat,64,255,cv2.THRESH_BINARY)

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] 

def predict_cnt(c, i):
    x, y, w, h = cv2.boundingRect(c)
    roi = gray[y:y + h, x:x + w]
    roi = cv2.threshold(roi,192,255,cv2.THRESH_BINARY)[1]
    roi = cv2.resize(roi,(50,50))
    cv2.imwrite(os.path.join("test_data", str(i)+".jpg"), roi)
    
i=0
for c in cnts:
    pred = predict_cnt(c,i)
    i= i+1
