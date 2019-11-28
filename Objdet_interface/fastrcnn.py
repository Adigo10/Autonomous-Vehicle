from flask import Flask, render_template, request, redirect, url_for

# imports 
import cgitb cgitb.enable()
import torchvision
import torch
import cv2
import imutils

# load pretrained model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
model = model.to(torch.device('cuda'))



COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
# Video Stream


"""

Replace the Video file location with 0 ,1 2 for camera stream

"""
cap = cv2.VideoCapture(0)


# draws bounding Box on frame
# needs model output and image to draw on

def draw_on_frame(outs, image):

    dicti = outs[0]

    # tensors need to be on cpu to perform operations other than NN
    boxes = dicti['boxes'].cpu()
    labels = dicti['labels'].cpu()
    scores = dicti['scores'].cpu()

    for number, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        if score.item() > 0.8 and label > 0:
            label_string = COCO_INSTANCE_CATEGORY_NAMES[label.item()]
            row1,col1,row2,col2 = [int(box[i].item()) for i in range(4)]
            p1 = (row1,col1)
            p2 = (row2,col2)
            image = cv2.rectangle(image,p1,p2,color=(0,255,0),thickness=2)
            image = cv2.putText(image,label_string,(row1,col1),cv2.FONT_HERSHEY_COMPLEX,0.5,color=(0,255,0),thickness=1)
            
            # print('drew')
    return image    





    return image

# only detects objects after 3 frames

AFTER_FRAME = 1

counter = 0



# Video Loop
while True:

    _, frame=cap.read()
    #frame = imutils.rotate(frame,180)
    frame = cv2.resize(frame,(300,300),interpolation=cv2.INTER_CUBIC)
    frame2=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor=torch.from_numpy(frame2).permute(2, 0, 1).unsqueeze(0).float()
    # print(frame_tensor.shape)
   
    frame_tensor=frame_tensor.to(torch.device('cuda'))
    # print(frame_tensor.device)
    counter += 1
    if counter % AFTER_FRAME == 0:
        with torch.no_grad():
            outs=model(frame_tensor/255)
            # print(outs)
            frame = draw_on_frame(outs,frame)

    frame_display = cv2.resize(frame ,(640,480),interpolation=cv2.INTER_CUBIC)
    cv2.imshow('frame', frame_display)
    if cv2.waitKey(24) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
