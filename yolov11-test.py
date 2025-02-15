import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from ultralytics import YOLO

model = YOLO('best.pt')  


test_folder = "D:\\batin_hastalik\\tummodel_original\\test\\images"
labels_folder = "D:\\batin_hastalik\\tummodel_original\\test\\labels"
detected_folder = "detectedimages"
os.makedirs(detected_folder, exist_ok=True)


image_files = [f for f in os.listdir(test_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]


def draw_boxes(image, boxes, color, label):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

for image_file in image_files:
    image_path = os.path.join(test_folder, image_file)
    label_path = os.path.join(labels_folder, image_file.rsplit('.', 1)[0] + '.txt')
    
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    
  
    results = model(image)
    pred_boxes = results[0].boxes.xyxy.cpu().numpy()  
     
    image = draw_boxes(image, pred_boxes, (0, 255, 0), "Prediction")
    
    
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            labels = f.readlines()
        gt_boxes = []
        for line in labels:
            data = line.strip().split()
            class_id = int(data[0])
            x_center, y_center, width, height = map(float, data[1:])
            
            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)
            gt_boxes.append([x1, y1, x2, y2])
        
        image = draw_boxes(image, gt_boxes, (255, 0, 0), "Ground Truth")
    
    output_path = os.path.join(detected_folder, image_file)
    cv2.imwrite(output_path, image)
    
    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()