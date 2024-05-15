import cv2
import numpy as np


class MonoCam:
    def __init__(self,
                 frame,
                 prev,
                 lk_params=None,
                 focal_length=900,
                 principal_point=(400, 600),
                 detector=cv2.FastFeatureDetector.create(nonmaxSuppression=True)
                 ):

        self.prevPoints = None
        if lk_params is None:
            lk_params=dict(winSize=(21,21), criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.frame = frame
        self.prev = prev
        self.focal_length = focal_length
        self.principal_point = principal_point
        self.coordinate = np.array([[0], [0], [0]])
        self.lk_params = lk_params
        self.detector = detector
        self.R = np.zeros(shape=(3, 3))
        self.t = np.zeros(shape=(3, 3))
        self.id = 0
        self.n_features = 0

    def odometry(self):
        # get the current and previous frame ready for detecting
        curr_gray = self.setFrame(self.frame)
        prev_gray = self.setFrame(self.prev)

        # if the number of features gets below 500, recall detect
        if self.n_features < 2000:
            self.prevPoints = self.getDetector(prev_gray)

        # Lukas Kanade optical flow
        points, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, self.prevPoints, None, **self.lk_params)


        # filter out the best points in Lukas Kanade
        good_old = self.prevPoints[st == 1]
        good_new = points[st == 1]

        # cv2.undistortPoints(good_new, self.cameraMatrix, dst=good_new)

        if self.id < 2:
            E, _ = cv2.findEssentialMat(good_new, good_old, focal=self.focal_length, pp=self.principal_point,
                                        method=cv2.RANSAC, prob=0.999, threshold=1.0)

            _, self.R, self.t, _ = cv2.recoverPose(E, good_old, good_new, focal=self.focal_length,
                                                   pp=self.principal_point)
            self.id =2
        else:
            E, _ = cv2.findEssentialMat(good_new, good_old, focal=self.focal_length, pp=self.principal_point,
                                        method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, R, t, _ = cv2.recoverPose(E, good_old, good_new, self.R.copy(), self.t.copy(), focal=self.focal_length,
                                         pp=self.principal_point)

            absolute_scale = self.getAbsoluteScale(good_old, good_new)

            if (absolute_scale > 0.1 and abs(t[2][0]) > abs(t[0][0]) and abs(t[2][0]) > abs(t[1][0])):
                self.R = R.dot(self.R)
                self.t = self.t + absolute_scale * self.R.dot(t)


        '''
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            a, b= int(a), int(b)

            self.frame = cv2.circle(self.frame, (a, b), 5, (0, 0, 220), -1)
        '''

        self.n_features = good_new.shape[0]

    def getAbsoluteScale(self, good_old, good_new):

        distance = np.linalg.norm(good_new - good_old, axis = 1)
        return np.log(np.median(distance))


    def getDetector(self, frame):
        self.prevPoints = self.detector.detect(frame, None)
        return np.array([x.pt for x in self.prevPoints], dtype=np.float32).reshape(-1, 1, 2)

    def setFrame(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def monocoordinate(self):
        diagnol = np.array([[-1, 0, 0],
                           [0, -1, 0],
                           [0, 0, -1]])
        coord = np.matmul(diagnol, self.t)
        return coord.flatten()

    #def triangulate(self):
        #todo

