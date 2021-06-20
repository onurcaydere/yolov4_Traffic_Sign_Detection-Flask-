import cv2
import time
import requests
import numpy as np
import json
url2="http://192.168.10.4:8080/shot.jpg"
url="http://localhost:8080/testGo"

CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

class_names = []
with open("obj.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

net = cv2.dnn.readNet("detect/final_traffic_light.weights", "detect/traffic_light.cfg")

while(True):
   img_resp=requests.get(url2)
   img_arr=np.array(bytearray(img_resp.content),dtype=np.uint8)
   img=cv2.imdecode(img_arr,cv2.IMREAD_COLOR)
   frame=cv2.resize(img,(208,208))
   lists = frame.tolist()
   json_str = json.dumps(lists)
   data = {'img_pixels': lists}
   response = requests.post(url,json=data)

       