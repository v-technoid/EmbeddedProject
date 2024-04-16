#Do the Model Here...
from ultralytics import YOLO
import os
import easyocr

def initreader():
    reader = easyocr.Reader(['en'])
    return reader

def extract_plate(img,bbox):
    return img[bbox[1]:bbox[3],bbox[0]:bbox[2]]

model=YOLO(r"best.pt")
source=[r"image source"]
results=model.predict(source=source)
for i,r in enumerate(results):
    plate=extract_plate(r.orig_img,r.boxes.xyxy.numpy().astype(int)[0])

reader=initreader()
os.system('cls')
print(reader.readtext(plate)[0][1])
