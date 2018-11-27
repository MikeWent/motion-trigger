#!/usr/bin/env python3

import argparse
import time
from os.path import exists as path_exists
import configparser

import numpy as np

import cv2

args = argparse.ArgumentParser()
args.add_argument("-t", "--threshold", help="motion level to trigger an action (overrides a value in the config file)", type=int)
args.add_argument("-c", "--cooldown", help="cooldown in seconds to wait before switching the pin back to the default state (overrides a value in the config file)", type=int)
args.add_argument("--debug", help="show the video input and motion diff", action="store_true")
options = args.parse_args()

config = configparser.ConfigParser()
config.read("config.ini")

THRESHOLD = options.threshold if options.threshold else int(config["defaults"].get("threshold", 10))
COOLDOWN = options.cooldown if options.cooldown else int(config["defaults"].get("cooldown", 30))
HEIGHT = int(config["webcam"]["height"])
WIDTH = int(config["webcam"]["width"])


def timestamp():
    """Get current UNIX time"""
    return int(time.time())


def gpio_set(pin, value):
    """Set GPIO pin to value
    
    https://github.com/MikeWent/gpio-remote-control/blob/master/server.py
    """
    if not path_exists("/sys/class/gpio/gpio{}".format(pin)):
        # Export pin
        with open("/sys/class/gpio/export", "w") as f:
            f.write(str(pin))
        time.sleep(0.05)
        # Set pin direction to "out"
        with open("/sys/class/gpio/gpio{}/direction".format(pin), "w") as f:
            f.write("out")
        time.sleep(0.05)
    # Set value
    with open("/sys/class/gpio/gpio{}/value".format(pin), "w") as f:
        f.write(str(value))

    if options.debug:
        print(f"Pin {pin} is switched to {value}")


print(f"# OpenCV {cv2.__version__}")
print(f"# Threshold  = {THRESHOLD}")
print(f"# Cooldown   = {COOLDOWN} sec")
print(f"# Resolution = {HEIGHT}x{WIDTH} px\n")

# This code is basically an improved version of https://software.intel.com/en-us/node/754940

if options.debug:
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.namedWindow('frame')
    cv2.namedWindow('dist')

def distMap(frame1, frame2):
    """Pythagorean distance between two frames"""
    frame1_32 = np.float32(frame1)
    frame2_32 = np.float32(frame2)
    diff32 = frame1_32 - frame2_32
    norm32 = np.sqrt(diff32[:,:,0]**2 + diff32[:,:,1]**2 + diff32[:,:,2]**2)/np.sqrt(255**2 + 255**2 + 255**2)
    dist = np.uint8(norm32*255)
    return dist

# capture video stream from camera source. 0 refers to first camera, 1 refers to 2nd and so on.
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)

_, frame1 = cap.read()
_, frame2 = cap.read()
print("Motion detection is started.")
last_time_motion_detected = 0
pin_state = "default"
frames_to_skip = 0

while(True):
    _, frame3 = cap.read()
    rows, cols, _ = np.shape(frame3)
    if options.debug:
        cv2.imshow('dist', frame3)
    dist = distMap(frame1, frame3)

    frame1 = frame2
    frame2 = frame3

    if frames_to_skip > 0:
        frames_to_skip -= 1
        continue

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

    if st_dev > THRESHOLD:
        print(f"Motion detected! Level: {round(st_dev, 1)}")
        last_time_motion_detected = timestamp()
        gpio_set(config["action"]["pin"], config["action"]["on_motion"])
        pin_state = "non-default"

    if last_time_motion_detected + COOLDOWN < timestamp() and pin_state == "non-default":
        print("Cooldown reached.")
        default_value = 1 if config["action"]["on_motion"] == "0" else 0
        gpio_set(config["action"]["pin"], default_value)
        pin_state = "default"
        # skip next 10 frames to prevent loop because light switch triggers motion detector
        frames_to_skip = 10


cap.release()
cv2.destroyAllWindows()
