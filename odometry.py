import cv2
import numpy as np


class MonoCam:
    def __init__(self, frame, prev,):
        self.frame = frame
        self.prev = prev

        self.lk_params = dict(winSize=(21, 21), criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.detector = cv2.FastFeatureDetector.create(threshold=10, nonmaxSuppression=True)

        # Camera specifications
        self.focal_length = 400
        self.principal_point = (500, 1000)

        self.R = np.zeros(shape=(3, 3))  # Rotational Matrix
        self.t = np.zeros(shape=(3, 3))  # Translational Vector

        self.prevPoints = None
        self.firstRun = True
        self.n_features = 0

    def odometry(self):
        """
        Main method for odometry,
        detects keypoints with the FAST feature detector,
        tracks the keypoints with Lucas-Kanade feature tracker
        updates the rotational matrix and translational vector
        """
        # get the current and previous frame ready for detecting
        curr_gray = self.setFrame(self.frame)
        prev_gray = self.setFrame(self.prev)

        # if the number of features gets below 500, re-call detect
        if self.n_features < 500:
            self.prevPoints = self.getDetector(prev_gray)

        # Lucas Kanade optical flow
        points, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, self.prevPoints, None, **self.lk_params)

        # filter out the best points in Lucas Kanade
        good_old = self.prevPoints[st == 1]
        good_new = points[st == 1]

        # We need to initialize the rotation and translation vectors the first time through,
        if self.firstRun:
            E, _ = cv2.findEssentialMat(good_new, good_old, focal=self.focal_length, pp=self.principal_point,
                                        method=cv2.RANSAC, prob=0.999, threshold=1.0)

            _, self.R, self.t, _ = cv2.recoverPose(E, good_old, good_new, focal=self.focal_length,
                                                   pp=self.principal_point)
            self.firstRun = False

        else:
            E, _ = cv2.findEssentialMat(good_new, good_old, focal=self.focal_length, pp=self.principal_point,
                                        method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, R, t, _ = cv2.recoverPose(E, good_old, good_new, focal=self.focal_length,
                                         pp=self.principal_point)

            # get the absolute scale of the motion, use it to update the Rotational matrix and translation vector
            absolute_scale = self.getAbsoluteScale(good_old, good_new)

            if (absolute_scale > 0.1 and (t[2][0]) > (t[0][0]) and (t[2][0]) > (t[1][0])):
                self.R = R.dot(self.R)
                self.t = self.t + absolute_scale * self.R.dot(t)

        # update the number of features
        self.n_features = good_new.shape[0]

    def getAbsoluteScale(self, good_old, good_new):
        """
        Calculates the 'scale' of the motion.
        inputs:
            good_old ------>  the points calculated through calcopticalflowpyrlk from the prev frame
            good_new ------>  the points found by calcopticalflowpyrlk
        output:
            the absolute scale between 0 and 1
        """
        distance = np.linalg.norm(good_new - good_old, axis=1)
        median = np.median(distance)

        return np.divide((median % 8), 8)

    def getDetector(self, frame):
        """
        Detects points in the frame using the FAST detector.
        input:
            frame ------> the frame to be detected
        output:
            an array of keypoints detected by FAST
        """
        self.prevPoints = self.detector.detect(frame, None)
        return np.array([x.pt for x in self.prevPoints], dtype=np.float32).reshape((-1, 1, 2))

    def setFrame(self, frame):
        """
        Modifies the frame to aid in detecting keypoints.
        input:
            frame ------> the frame to be modified
        output:
            the color shifted frame
        """
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def multDiagonol(self):
        """
        Multiply the translation vector by the diagonol matrix to get the correct orientation

        output:
            the one dimensional product of the diagonol matrix and the translation vector
        """
        diagonol = np.array([[-1, 0, 0],
                           [0, -1, 0],
                           [0, 0, -1]])
        coord = np.matmul(diagonol, self.t)
        return coord.flatten()
