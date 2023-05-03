import numpy as np 
import cv2 
from keras.models import load_model
#import os
import pandas as pd

file = "IBAN.jpg"
numeric_char_model = load_model("numeric_char_classifier.h5")
chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def classify_chars(file, chars, model):
    img = cv2.imread(file)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    
    kernel_blackhat = np.ones((20,20),np.uint8) 
    blackhat = cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,kernel_blackhat) 
    ret,thresh = cv2.threshold(blackhat,64,255,cv2.THRESH_BINARY)
    
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] 
    
    df = pd.DataFrame([])
    
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        
        roi = gray[y:y + h, x:x + w]
        roi = cv2.resize(roi,(50,50))
        roi = cv2.threshold(roi,197,255,cv2.THRESH_BINARY)[1]
        
        roi = np.array(roi)
        roi = roi / 255
        roi = roi.reshape(1, 50, 50, 1) 
        roi = roi.astype('float32')
        
        prediction = model.predict(roi)
        pred_char = chars[prediction.argmax()]
        prob = prediction.max()
        # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1) 
        # cv2.putText(img,str(pred_char),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,0),1)
    
        df_cnt = pd.DataFrame(
            {"x": x,
            "y": y,
            "w": w,
            "h": h,
            "prediction": pred_char,
            "probability": prob},
            index=[0])
        
        df = df.append(df_cnt, ignore_index=True)
    
    return(df)

res = classify_chars(file = file, 
                     chars = chars,
                     model = numeric_char_model)

# For debugging, draw predictions on image
def draw_predictions(file, df_predictions):
    img = cv2.imread(file)
    for index, row in df_predictions.iterrows():
        x = row["x"]
        y = row["y"]
        w = row["w"]
        h = row["h"]
        pred_char = row["prediction"]
        
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
        cv2.putText(img,str(pred_char),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,0),1)
    
    cv2.imwrite("pred_"+file, img)

draw_predictions(file, res)
