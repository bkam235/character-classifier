from wand.color import Color
from wand.image import Image
from wand.drawing import Drawing
import os
import numpy as np
import cv2 
from joblib import parallel_backend, Parallel, delayed
import shutil

np.random.seed(42)

out_dir = "out"
font_dir = "fonts"
chars = list("0123456789")
img_xy = 20
n_augmentations = 5

os.makedirs(out_dir, exist_ok=True)
shutil.rmtree(out_dir, ignore_errors=True)
fonts = os.listdir(font_dir)
     
def extract_contour(filename):
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    kernel = np.ones((20,20),np.uint8) 
    blackhat = cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,kernel)  
    ret,thresh = cv2.threshold(blackhat,64,255,cv2.THRESH_BINARY)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] 

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        roi = gray[y:y + h, x:x + w]
        ret, thresh = cv2.threshold(roi,np.random.randint(127,200),255,cv2.THRESH_BINARY)
        try:
            roi = cv2.resize(thresh,(50,50))
            cv2.imwrite(filename, roi)
        except:
            next


def draw_image(char, font, i):
    with Drawing() as draw:
        with Image(width=img_xy, 
                    height=img_xy, 
                    background=Color('white'),
                    resolution=100) as img:
            draw.font = os.path.join(font_dir, font)
            draw.font_size = img_xy
            draw.text_alignment = "center"
            text_height = draw.get_font_metrics(img, "0123456789")[5]
            draw.push()
            draw.fill_color = Color('BLACK')
            draw.text(int(img.width / 2), int(img.width /2 + text_height / 3), char)
            
            # draw_left = np.random.uniform(0,1)
            # draw_right = np.random.uniform(0,1)
            
            # if draw_left > 0.8:
            #     draw.text(0, int(img.width /2 + text_height / 3), chars[np.random.randint(0,len(chars))])
            
            # if draw_right > 0.8:
            #     draw.text(int(img.width), int(img.width /2 + text_height / 3), chars[np.random.randint(0,len(chars))])
            
            draw.pop()
            draw(img)
            
            #Data Augmentation
            # img.resize(np.random.randint(int(img_xy/1.2),int(img_xy*1.2)), np.random.randint(int(img_xy/1.2),int(img_xy*1.2)))
            # img.rotate(np.random.uniform(-10,10), background=Color('white'))
            
            # add_choice = np.random.choice([4])
            # if add_choice == 0:
            #     img.implode(amount=np.random.uniform(0,0.5))
            # elif add_choice == 1:
            #     img.statistic('minimum', width=1, height=1)
            # elif add_choice == 2:
            #     img.blur(radius=0, sigma=np.random.uniform(0,0.5))
            # elif add_choice == 3:
            #     noise_types = ['gaussian','uniform']
            #     img.noise(noise_types[np.random.randint(0, len(noise_types))], attenuate=np.random.uniform(0,1)) #
            
            img.transform_colorspace('gray')
            
            img_filename = os.path.join(out_dir, char+"_"+font+"_aug"+str(i)+".jpg")
            img.save(filename=img_filename)

def generate_data(char, font):
    i = 0
    while i < n_augmentations:
        draw_image(char, font, i)
        i = i+1

def generate_data_char(char):
    for font in fonts:
        generate_data(char, font)

with parallel_backend('loky', n_jobs=12):
    Parallel()(delayed(generate_data_char)(char) for char in chars)

files = os.listdir(out_dir)
for file in files:
    extract_contour(os.path.join(out_dir, file))
