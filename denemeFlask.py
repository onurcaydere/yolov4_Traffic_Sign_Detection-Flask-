from flask import Flask, request, Response
import cv2
import numpy as np
import json as js

import time
app = Flask(__name__)
# YOLONUN DAHİL EDİLMESİ;
CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
url="http://192.168.10.4:8080/shot.jpg"
class_names = []
with open("obj.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
net = cv2.dnn.readNet("detect/final_traffic_light.weights", "detect/traffic_light.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
def yolo_tahmin(img):
    classes, scores, boxes = model.detect(img, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    start_drawing = time.time()
    print(boxes)
    print(classes)
    return boxes,classes




@app.route("/testGo",methods=["GET","POST"])
def test2():   
    data = request.get_json()
    array = np.array(data['img_pixels'],np.uint8)
    box_predict,class_predict=yolo_tahmin(array)
    
    return "box_predict,class_predict"
@app.route('/one', methods=['GET', 'POST'])
def handle_request():
    return "Get Fonksiyonu Çalışmakta"
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True)
