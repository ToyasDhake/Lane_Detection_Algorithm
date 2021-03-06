from time import time, sleep

import cv2
import numpy as np

# Set VideoCapture
c = cv2.VideoCapture('data_2/challenge_video.mp4')

# HSV values for white and yellow
lower_white = np.array([0, 0, 177], dtype=np.uint8)
upper_white = np.array([179, 65, 255], dtype=np.uint8)
lower_yellow = np.array([82, 40, 0], dtype=np.uint8)
upper_yellow = np.array([115, 255, 255], dtype=np.uint8)

# Calculate time per frame
fps = c.get(cv2.CAP_PROP_FPS)
time_per_frame = 1 / fps
previous_time = time()
old_value = 0


# Transforms image to world frame
def inverse(turn, image, contour, m_inverse, left_line, right_line):
    x = np.linspace(0, contour.shape[0] - 1, contour.shape[0])
    left_curve = left_line[0] * x ** 2 + left_line[1] * x + left_line[2]
    right_curve = right_line[0] * x ** 2 + right_line[1] * x + right_line[2] + 155
    color = np.zeros_like(contour).astype(np.uint8)
    # Get points on the curve
    points_left = np.array([np.transpose(np.vstack([left_curve, x]))])
    points_right = np.array([np.flipud(np.transpose(np.vstack([right_curve, x])))])
    points = np.hstack((points_left, points_right))
    cv2.fillPoly(color, np.int_([points]), (0, 255, 0))
    # Inverse transform the frame to world frame
    new_warp = cv2.warpPerspective(color, m_inverse, (image.shape[1], image.shape[0]))
    # Put text for turn instructions
    cv2.putText(new_warp, str(turn), (630, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
    result = cv2.addWeighted(image, 1, new_warp, 0.3, 0)
    return result


# Iterate through video
while (True):
    # Read frame
    ret, image = c.read()
    # Check if frame exists
    if ret == True:
        # Points on original image
        points_1 = np.float32([[570, 475], [740, 475], [95, 710], [1260, 710]])
        points_2 = np.float32([[0, 0], [256, 0], [0, 256], [256, 256]])
        # Transform the ROI from world frame to camera frame
        m = cv2.getPerspectiveTransform(points_1, points_2)
        m_inverse = np.linalg.inv(m)
        warp = cv2.warpPerspective(image, m, (256, 256))
        # Convert to HSV
        hsv = cv2.cvtColor(warp, cv2.COLOR_RGB2HSV)
        # Create mask for white and yellow color
        mask_1 = cv2.inRange(hsv, lower_white, upper_white)
        mask_2 = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask = cv2.add(mask_1, mask_2)
        # Split mask in left and right halves
        left_half = mask[:, :128]
        left_half = np.array(left_half)
        left_half_points = left_half.nonzero()
        # Get polynomial equation for all points on the left half
        left_poly_fit = np.polyfit(left_half_points[0], left_half_points[1], 2)
        right_half = mask[:, 155:]
        right_half = np.array(right_half)
        right_half_points = right_half.nonzero()
        # Get polynomial equation for all points on the right half
        right_poly_fit = np.polyfit(right_half_points[0], right_half_points[1], 2)
        val = (right_poly_fit[0] + left_poly_fit[0] * 3) / 4
        val = (val + old_value * 6) / 7
        old_value = val
        # Based on the curve of curve determine the turn to be taken
        if abs(val) < 0.00065:
            original = inverse("Straight", image, warp, m_inverse, left_poly_fit, right_poly_fit)
        else:
            if val > 0:
                original = inverse("Turn Right", image, warp, m_inverse, left_poly_fit, right_poly_fit)
            else:
                original = inverse("Turn Left", image, warp, m_inverse, left_poly_fit, right_poly_fit)

        # Display output frames
        cv2.imshow("Original", original)
        cv2.imshow("x", mask)
        # Quit if user press escape key
        k = cv2.waitKey(10)
        if k == 27:
            break
        # Limit the video playback to the original video frame rate
        new_time = time()
        if new_time - previous_time < time_per_frame:
            sleep(time_per_frame - (new_time - previous_time))
        previous_time = new_time
    # Break if frame is not available
    else:
        break
# Release VideoCapture and destroy open windows
c.release
cv2.destroyAllWindows()
