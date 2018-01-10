#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 01:46:56 2018

@author: snigdharao
"""

 # Object Detection

# Importing the libraries

import torch    #builds dynamic graphs which allows fast and efficient computations
from torch.autograd import Variable   # helps in converting torch variables into gradients
import cv2    #imported for the graphs drawn around the objects
from data import BaseTransform, VOC_CLASSES as labelmap   # BaseTransform is a class used to transform images into a compatible form which fits the neural networks
#VOC_CLASSES is a dictionary which encodes and maps text fields and some integers
from ssd  import build_ssd   #constructor which builds the architecture of the ssd model
import imageio    #used to process the images of the video and apply the detect function on the images

# Defining a function for detections

def detect(frame, net, transform):
    height, width = frame.shape[:2]
    #FOUR transformations to go from original frame to a torch variable that will be accepted in the ssd nn
    #1. apply 'transform' transformation to make sure the image has the right dimensions and right color values
    frame_t = transform(frame)[0]
    
    #2. convert transform frame from numpy array to a torch tensor
    x = torch.from_numpy(frame_t).permute(2, 0, 1)
    
    #3. add a fake dimension to the torch tensor, which corresponds to the batch
    #4. convert it into a torch variable which contains both the tensor and the gradient
    #3rd and 4th transformation done in one line of code
    x = Variable(x.unsqueeze(0))
    y = net(x)   # feeding x to the neural network
    detections = y.data   # gets the torch tensor
    scale = torch.Tensor([width, height, width, height])        # creating a tensor object of dimensions specified
    
    # detections contains four elements
    # detections = [batch, number of classes, i.e, the number of objects that are detected, number of occurences of the class, (score, x0, y0, x1, y1)]
    # if score < 0.6, occurence is not found & if score > 0.6, occurence is found
    for i in range(detections.size(1)):   # i = class
        j = 0 # j = occurence
        while detections[0, i, j, 0] > 0.6:         # taking into account all the occurrences j of the class i that have a matching score larger than 0.6.
            pt = (detections[0, i, j, 1:] * scale).numpy()      # getting the coordinates of the points at the upper left and the lower right of the detector rectangle.
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2)        # drawing a rectangle around the detected object.
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)       # putting the label of the class right above the rectangle.
            j+=1
    return frame    # returning the original frame with the detector rectangle and the label around the detected object.


# creating the SSD neural networks
net = build_ssd('test')
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location= lambda storage, loc: storage))   # We get the weights of the neural network from another one that is pretrained (ssd300_mAP_77.43_v2.pth).

# creating the transformations

transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0)) # We create an object of the BaseTransform class, a class that will do the required transformations so that the image can be the input of the neural network.


# doing some object detection on a video

reader = imageio.get_reader('funny_dog.mp4')   # opening the video
fps = reader.get_meta_data()['fps']     # getting the 'frames per second' frequence
writer = imageio.get_writer('output.mp4', fps = fps)        # creating an output video of the same fps
for i, frame in enumerate(reader):          # iterating on the frames of the output video
    frame = detect(frame, net.eval(), transform)          #  calling our detect function (defined above) to detect the object on the frame
    writer.append_data(frame)           # adding the next frame in the output video
    print(i)            # printing the number of the processed frame
writer.close()      # closing the process which handles the creation of the output video