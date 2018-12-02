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
BRIGHTNESS = int(config["webcam"]["brightness"])


def timestamp():
    """Current UNIX time"""
    return int(time.time())


def gpio_set(pin, value):
    """Set GPIO pin to value
    
    https://github.com/MikeWent/gpio-remote-control/blob/master/server.py
    """
    if options.debug:
        print(f"Pin {pin} is switched to {value}")
        return
    
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


def diff_img(t0, t1, t2):
    """Difference between three images"""
    d1 = cv2.absdiff(t2, t1)
    d2 = cv2.absdiff(t1, t0)
    return cv2.bitwise_and(d1, d2)


def get_frame():
    """Retrive a graysacle frame from capturing device (webcam)"""
    global cap
    _, frame = cap.read()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


print(f"# OpenCV {cv2.__version__}")
print(f"# Threshold  = {THRESHOLD}")
print(f"# Cooldown   = {COOLDOWN} sec")
print(f"# Resolution = {WIDTH}x{HEIGHT} px\n")

if options.debug:
    cv2.namedWindow("motion", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("raw", cv2.WINDOW_AUTOSIZE)

# capture video stream from camera source. 0 refers to first camera, 1 refers to 2nd and so on.
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_BRIGHTNESS, BRIGHTNESS)

frame1 = get_frame()
frame2 = get_frame()
print("Motion detection is started.")
last_time_motion_detected = 0
pin_state = "default"
frames_to_skip = 0

while(True):
    frame3 = get_frame()
    if options.debug:
        cv2.imshow("raw", frame3)

    # Skip frames if needed
    if frames_to_skip > 0:
        frames_to_skip -= 1
        continue

    # Calculate diff between frames
    diff = diff_img(frame1, frame2, frame3)

    # Shift frames
    frame1 = frame2
    frame2 = frame3

    # Apply Gaussian smoothing
    blurred = cv2.GaussianBlur(diff, (13,13), 0)

    # Apply thresholding
    _, thresh = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)

    # Calculate st dev
    _, st_dev = cv2.meanStdDev(thresh)
    st_dev = st_dev[0][0]

    if options.debug:
        cv2.imshow("motion", thresh)
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
        # Skip next 10 frames to prevent loop because light switch triggers motion detector
        frames_to_skip = 10

cap.release()
cv2.destroyAllWindows()
