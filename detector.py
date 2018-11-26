#!/usr/bin/env python3

import cv2
import numpy as np
import time
import argparse

args = argparse.ArgumentParser()
args.add_argument("-t", "--threshold", help="motion level to trigger an action", type=int, default=10)
args.add_argument("--debug", help="show the video input and motion diff", action="store_true")
options = args.parse_args()

print(f"Using OpenCV {cv2.__version__}")

# This code is basically an improved version of https://software.intel.com/en-us/node/754940

if options.debug:
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.namedWindow('frame')
    cv2.namedWindow('dist')

def distMap(frame1, frame2):
    """outputs pythagorean distance between two frames"""
    frame1_32 = np.float32(frame1)
    frame2_32 = np.float32(frame2)
    diff32 = frame1_32 - frame2_32
    norm32 = np.sqrt(diff32[:,:,0]**2 + diff32[:,:,1]**2 + diff32[:,:,2]**2)/np.sqrt(255**2 + 255**2 + 255**2)
    dist = np.uint8(norm32*255)
    return dist

# capture video stream from camera source. 0 refers to first camera, 1 refers to 2nd and so on.
cap = cv2.VideoCapture(0)

_, frame1 = cap.read()
_, frame2 = cap.read()
print("Motion detection is started.")

while(True):
    _, frame3 = cap.read()
    rows, cols, _ = np.shape(frame3)
    if options.debug:
        cv2.imshow('dist', frame3)
    dist = distMap(frame1, frame3)

    frame1 = frame2
    frame2 = frame3

    # apply Gaussian smoothing
    mod = cv2.GaussianBlur(dist, (9,9), 0)

    # apply thresholding
    _, thresh = cv2.threshold(mod, 100, 255, 0)

    # calculate st dev test
    _, st_dev = cv2.meanStdDev(mod)
    st_dev = st_dev[0][0]

    if options.debug:
        cv2.imshow('dist', mod)
        cv2.putText(frame2, "Standard Deviation = {}".format(round(st_dev, 0)), (70, 70), font, 1, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow('frame', frame2)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    if st_dev > options.threshold:
        print(f"Motion detected! Level: {round(st_dev, 1)}")

cap.release()
cv2.destroyAllWindows()
