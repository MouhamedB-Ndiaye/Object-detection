import cv2

cap = cv2.VideoCapture(0)
cap.set(3,648) # to define how big our image isq
cap.set(4,488)

classNames= []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)
                                                                                                       
weightsPath = "frozen_inference_graph.pb"
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"

# now we can simply create our model
# the good thing about opencv is that , it already provides us with the function that actually does all the processing for us
# and all we have to do is to input our configurationPath and weightsPath

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# now we need to send our image to our model then it will give us their predictions
# bbox: bounding box - with this we are going to create the rectangle of our objects
# we can also write the name based on our classIds
while True:
    success,img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=0.5) #confThreshold : at what point do we detect it as an actual object
    print(classIds, bbox)

    if len(classIds)  != 0:
        for classId, confidence,box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img,box, color=(0,255,0),thickness=2) # allow us to make the rectangle(box) on our image
            cv2.putText(img, classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2) # --> to write the class inside the box
            cv2.putText(img, str(round(confidence*100, 2)), (box[0] + 200, box[1] + 30),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            #cv2.putText(img, str(confidence*100), (box[0] + 200, box[1] + 30),
                #cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2) # doing that we don't have the accuracy (confidence) in percentage

    cv2.imshow("output", img)
    cv2.waitKey(1)

