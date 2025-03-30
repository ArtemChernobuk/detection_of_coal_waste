import cv2
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(vid_path)
backSub = cv2.createBackgroundSubtractorMOG2()
if not cap.isOpened():
    print("Error opening video file")
    while cap.isOpened():
        # Захват кадр за кадром
          ret, frame = cap.read()
          if ret:
            # Вычитание фона
            fg_mask = backSub.apply(frame)