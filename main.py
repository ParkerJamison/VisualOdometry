"""
    Monocular Odometry ----- Parker Jamison 2024

    Calculates the 'Trajectory' of a camera using
    FAST feature detector, and the Lucas-Kanade
    feature tracker.

    Using IPhone 14 Pro main camera for camera
    specifications.
"""
import cv2
import numpy as np
from odometry import MonoCam


def main():

    # Insert a video here
    cap = cv2.VideoCapture('IMG_2653.MOV')

    # Initialize the trajectory plot
    traj = np.zeros(shape=(600, 600, 3))
    cv2.imshow("trajectory", traj)

    ret, prev = cap.read()
    image = None

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            input("press enter to exit: ")
            break
        # on the first iteration, initialize the monocam object
        if image is None:
            image = MonoCam(frame, prev)
            continue

        # update the frames
        image.prev = image.frame
        image.frame = frame

        image.odometry()

        mono = image.multDiagonol()
        draw_x, draw_y, draw_z = [int(round(x)) for x in mono]
        # multiply the coordinates by two in smaller scale situations to see the trajectory better
        draw_z <<= 2
        draw_x <<= 2

        traj = cv2.circle(traj, (-draw_x+300, draw_z+300), 1, list((0, 255, 0)), 1)

        cv2.imshow('webcam', image.frame)
        cv2.imshow('trajectory', traj)

        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()


if __name__ == '__main__':
    main()
