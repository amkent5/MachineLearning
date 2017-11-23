# Simple template matching example using the OpenCV library.
'''
The cv2.matchTemplate function takes a sliding window of our wally query image and slides it across our puzzle
image from left to right and top to bottom, one pixel at a time. Then, for each of these locations, we compute
the correlation coefficient to determine how 'good' or 'bad' the match is. Regions with sufficiently high correlation 
can be considered 'matches' for our wally template.
'''

import numpy as np
import cv2

# import utility functions
import image_utils

# load images
puzzle_path = '/Users/admin/Documents/Projects/ComputerVision/my_code/wheres_wally/puzzle.png'
wally_path = '/Users/admin/Documents/Projects/ComputerVision/my_code/wheres_wally/wally.png'

# store images as multi-dimensional numpy arrays
puzzle = cv2.imread(puzzle_path)
wally = cv2.imread(wally_path)

# get wally's dimensions
print wally.shape
wally_height = wally.shape[0]
wally_width = wally.shape[1]

# perform template matching
result = cv2.matchTemplate(puzzle, wally, cv2.TM_CCOEFF)
print result

# then find where the 'good' matches are (or in this simple case, our 'match')
min_location = cv2.minMaxLoc(result)[2]
max_location = cv2.minMaxLoc(result)[3]

print min_location
print max_location

# Grab the bounding box and extract Wally from the puzzle
top_left = max_location
bottom_right = (top_left[0] + wally_width, top_left[1] + wally_height)
region_of_interest = puzzle[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

# show original puzzle
# (press ESC to move onto next image)
cv2.imshow("Puzzle", image_utils.resize(puzzle, height=650))
cv2.waitKey(0)

# darken everything out but Wally
mask = np.zeros(puzzle.shape, dtype = "uint8")
puzzle = cv2.addWeighted(puzzle, 0.25, mask, 0.75, 0)

# put Wally back!
puzzle[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = region_of_interest
cv2.imshow("Puzzle", image_utils.resize(puzzle, height=650))
cv2.waitKey(0)
