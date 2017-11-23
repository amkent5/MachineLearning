# Lot's of sources say OpenCV is a bitch to install
# Second comment on this:
# https://stackoverflow.com/questions/34853220/cannot-import-cv2-in-python-in-osx
# provides an easy way

# sudo pip install opencv-python

import cv2
import sys

print ('cv2' in sys.modules)
