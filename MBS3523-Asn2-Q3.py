# YOLO v3

import cv2
import numpy as np

confThreshold = 0.4

cam = cv2.VideoCapture('images.jpg')

# Create an empty list - classes[] and point the classesFile to 'coco80.names'
classesFile = 'coco80.names'
classes = []
# Load all classes in coco80.names into classes[]
with open(classesFile, 'r') as f:
    classes = f.read().splitlines()
    print(classes)
    print(len(classes))

# Load the configuration and weights file
# You need to download the weights and cfg files from https://pjreddie.com/darknet/yolo/
net = cv2.dnn.readNetFromDarknet('yolov3-608.cfg','yolov3-608.weights')
# Use OpenCV as backend and use CPU
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


while True:
    success , img = cam.read()
    if not success:
        cam = cv2.VideoCapture('images.jpg')
        success , img = cam.read()

    height, width, ch = img.shape

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    print(layerNames)

    output_layers_names = net.getUnconnectedOutLayersNames()
    print(output_layers_names)

    LayerOutputs = net.forward(output_layers_names)
    print(len(LayerOutputs))
    # print(LayerOutputs[0].shape)
    # print(LayerOutputs[1].shape)
    # print(LayerOutputs[2].shape)
    # print(LayerOutputs[0][0])


    bboxes = [] # array for all bounding boxes of detected classes
    confidences = [] # array for all confidence values of matching detected classes
    class_ids = [] # array for all class IDs of matching detected classes

    for output in LayerOutputs:
        for detection in output:
            scores = detection[5:] # omit the first 5 values
            class_id = np.argmax(scores) # find the highest score ID out of 80 values which has the highest confidence value
            confidence = scores[class_id]
            if confidence > confThreshold:
                center_x = int(detection[0]*width) #YOLO predicts centers of image
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)

                bboxes.append([x,y,w,h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)
                # cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 2)

    # print(len(bboxes))
    indexes = cv2.dnn.NMSBoxes(bboxes, confidences, confThreshold, 0.4) #Non-maximum suppresion
    # print(indexes)
    # print(indexes.flatten())

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0,255,size=(len(bboxes),3))
    ap = 0
    ba = 0
    og = 0
    if len(indexes) > 0:
        for i in indexes.flatten():
            x,y,w,h = bboxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = colors[i]
            if label == 'apple':
                ap = ap+1
                cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
                cv2.putText(img,label +" "+ "["+str(ap)+"]  " + str(round(float(confidence)*100,2)) + '%', (x,y+20),font,1,(255,255,255),2)
            if label == 'banana':
                ba = ba+1
                cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
                cv2.putText(img,label +" "+ "["+str(ba)+"]  " + str(round(float(confidence)*100,2))+ '%', (x,y+20),font,1,(255,255,255),2)
            if label == 'orange':
                og = og+1
                cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
                cv2.putText(img,label +" "+ "["+str(og)+"]  " + str(round(float(confidence)*100,2)) + '%', (x,y+20),font,1,(255,255,255),2)

        cv2.putText(img, "There are: " + str(ap) + " " + 'apple', (800, 20), font, 1.5, (0, 0, 255), 2)
        cv2.putText(img, "There are: " + str(ba) + " " + 'banana', (800, 40), font, 1.5, (0, 255, 255), 2)
        cv2.putText(img, "There are: " + str(og) + " " + 'orange', (800, 60), font, 1.5, (9, 9, 70), 2)
        apt = ""
        bat = ""
        ogt = ""
        Ft = ""
        if ap>0:
            Ft+= str(ap)+'x1'
        if ba>0:
            if ap > 0:
                Ft += "+"
            Ft+= str(ba)+'x2'
        if og>0:
            if ap > 0 or ba > 0:
                Ft+="+"
            Ft+= str(og)+'x3'
        cv2.putText(img, "total $: " + Ft + '='+ str(round(ap*1+ba*2+og*3)), (700, 80), font, 1.5, (0, 0, 0), 2)


    cv2.imshow('Image', img)

    if cv2.waitKey(1) & 0xff == 27:
        break

cam.release()
cv2.destroyAllWindows()