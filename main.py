import cv2
import numpy as np
from odometry import MonoCam

lk_params = dict(winSize=(21, 21),
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def main():

    cap = cv2.VideoCapture('IMG_2644.MOV')
    #cap = cv2.VideoCapture('IMG_2645.MOV')
    #cap = cv2.VideoCapture(0)
    traj = np.zeros(shape=(600, 600, 3))

    ret, prev = cap.read()
    image = None

    while True:
        ret, frame = cap.read()

        # on the first iteration, initialize the monocam object
        if image is None:
            image = MonoCam(frame, prev, lk_params)
            continue

        # update the frames
        image.prev = image.frame
        image.frame = frame

        image.odometry()


        mono = image.monocoordinate()
        #print(mono)
        draw_x, draw_y, draw_z = [int(round(x)) for x in mono]
        traj = cv2.circle(traj, (draw_x+300, draw_z+300), 1, list((0, 255, 0)), 1)

        cv2.imshow('webcam', image.frame)
        cv2.imshow('trajectory', traj)

        if cv2.waitKey(1) == ord('q'):
            break

if __name__ == '__main__':
    main()