
import streamlit as st
import torch
from PIL import Image
import cv2
import numpy as np

#load model YOLOV5 
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

st.title('Traffic Light Detection Application')

uploaded_image = st.file_uploader('Choose an Image...', type = ['jpg', 'jpeg', 'png', 'webp'])

#define function to detect colors
def Color_detection(image, box):
    #crop image
    x1, y1, x2, y2 = map(int, box)
    cropped_image = image[y1:y2, x1:x2] #height , width

    #convert image to HSV Color Space
    hsv_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2HSV)

    #define color ranges for red, yellow, green
    red_lower1 = np.array([0, 70, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 70, 50])
    red_upper2 = np.array([180, 255, 255])

    green_lower = np.array([40,40,40])
    green_upper = np.array([80, 255, 255])

    yellow_lower = np.array([20,100,100])
    yellow_upper = np.array([30, 255, 255])

    #check for red mask 
    red_mask1 = cv2.inRange(hsv_image, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv_image, red_lower2, red_upper2)
    red_mask = red_mask1 | red_mask2
    red_pixels = cv2.countNonZero(red_mask)

    #check for green mask
    green_mask = cv2.inRange(hsv_image, green_lower, green_upper)
    green_pixels = cv2.countNonZero(green_mask)

    #check for yellow mask
    yellow_mask = cv2.inRange(hsv_image, yellow_lower, yellow_upper)
    yellow_pixels = cv2.countNonZero(yellow_mask)

    #determine color of traffic light
    if max(red_pixels, green_pixels, yellow_pixels) == red_pixels:
        return 'Red'
    elif max(red_pixels, green_pixels, yellow_pixels) == green_pixels:
        return 'Green'
    elif max(red_pixels, green_pixels, yellow_pixels) == yellow_pixels:
        return 'Yellow'
    else:
        return 'Unknown Color'

if uploaded_image is not None:
    img = Image.open(uploaded_image)
    image_array = np.array(img)
    st.image(image_array, caption = 'Uploaded Image', use_container_width = True)

    #perform detection
    detection_result = model(image_array)

    #get bounding result
    boxes = detection_result.xyxy[0].numpy()

    #display results 
    for box in boxes:
        label = detection_result.names[int(box[5])]
        if label == 'traffic light':
            color = Color_detection(image_array, box[:4])
            st.write(f'Detected Traffic Light Color: {color}')
        else:
            print(label)
    #show bounding box
    st.image(detection_result.render()[0], caption = 'Detected Traffic Light', use_container_width = True)
