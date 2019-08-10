# -*- coding: utf-8 -*-

import sys
from hw2_ui import Ui_Dialog
import cv2
import os
import sys
import numpy as np
import glob
from PyQt5.QtWidgets import QMainWindow, QApplication
from matplotlib import pyplot as plt

# Back up the reference to the exceptionhook
sys._excepthook = sys.excepthook

def my_exception_hook(exctype, value, traceback):
    # Print the error and traceback
    print(exctype, value, traceback)
    # Call the normal Exception hook after
    sys._excepthook(exctype, value, traceback)
    sys.exit(1)

# Set the exception hook to our wrapping function
sys.excepthook = my_exception_hook

class MainWindow(QMainWindow, Ui_Dialog):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.onBindingUI()

    # Write your code below
    # UI components are defined in hw1_ui.py, please take a look.
    # You can also open hw1.ui by qt-designer to check ui components.

    def onBindingUI(self):
        self.pushButton.clicked.connect(self.on_btn1_1_click)
        self.pushButton_2.clicked.connect(self.on_btn1_2_click)
        self.pushButton_21.clicked.connect(self.on_btn2_1_click)
        self.pushButton_22.clicked.connect(self.on_btn2_2_click)
        self.pushButton_5.clicked.connect(self.on_btn2_3_click)
        self.pushButton_6.clicked.connect(self.on_btn3_1_click)
        self.pushButton_7.clicked.connect(self.on_btn3_2_click)
        self.pushButton_8.clicked.connect(self.on_btn3_3_click)
        self.pushButton_9.clicked.connect(self.on_btn3_4_click)
        self.pushButton_10.clicked.connect(self.on_btn4_1_click)

    # button for problem 1.1
    def on_btn1_1_click(self):
        img = cv2.imread("./images/plant.jpg", cv2.COLOR_BGR2GRAY)
        cv2.imshow("Original Image", img)
        # hist = cv2.calcHist([img], [0], None, [256], [0, 256])

        plt.hist(img.ravel(), 256, [0, 256])
        plt.title('Histogram for gray scale picture')
        plt.xlabel("gray value")
        plt.ylabel("Pixel numbers")
        plt.xlim([0, 256])

        plt.show()

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def on_btn1_2_click(self):
        img = cv2.imread("./images/plant.jpg")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        equImg = cv2.equalizeHist(gray)
        cv2.imshow("After Equlization", equImg)

        # hist = cv2.calcHist([equImg], [0], None, [256], [0, 256])

        plt.hist(equImg.ravel(), 256, [0, 256])
        plt.title('Histogram afrer Equalization')
        plt.xlabel("gray value")
        plt.ylabel("Pixel numbers")
        plt.xlim([0, 256])

        plt.show()

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def on_btn2_1_click(self):
        img = cv2.imread("./images/q2_train.jpg")
        img = cv2.medianBlur(img, 5)
        cImg = img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1 = 100, param2 = 22, minRadius = 14, maxRadius = 30)
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(cImg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(cImg, (i[0], i[1]), 2, (0, 0, 255), 3)

        cv2.imshow('detected circles', cImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def on_btn2_2_click(self):
        img = cv2.imread("./images/q2_train.jpg")
        mask = np.zeros(img.shape, np.uint8)
        cImg = img

        img = cv2.medianBlur(img, 5)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1 = 100, param2 = 22, minRadius = 14, maxRadius = 30)
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(mask, (i[0], i[1]), i[2], (255, 255, 255), cv2.FILLED)

        #certain_area is image with only area in the circle we drew
        certain_area = cv2.bitwise_and(cImg, mask)

        certain_area = cv2.cvtColor(certain_area, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([certain_area], [0], None, [180], [1, 180])
        hist = hist / hist.max()
        x = np.arange(len(hist))


        plt.bar(x, hist.ravel())

        plt.title('Normalized Hue Histogram')
        plt.xlabel("Angle")
        plt.ylabel("Probabilitis")
        plt.xlim([0, 180])
        plt.show()

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def on_btn2_3_click(self):
        img = cv2.imread("./images/q2_train.jpg")
        targetImg = cv2.imread("./images/q2_test.jpg")
        cv2.pyrMeanShiftFiltering(img, 2, 10, img, 4)

        ##certain area!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        mask = np.zeros(img.shape, np.uint8)
        cImg = img
        grayImg = img

        grayImg = cv2.medianBlur(img, 5)
        grayImg = cv2.cvtColor(grayImg, cv2.COLOR_BGR2GRAY)

        circles = cv2.HoughCircles(grayImg, cv2.HOUGH_GRADIENT, 1, 20, param1=100, param2=22, minRadius=14, maxRadius=30)
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(mask, (i[0], i[1]), i[2], (255, 255, 255), cv2.FILLED)

        # certain_area is image with only area in the circle we drew
        certain_area = cv2.bitwise_and(cImg, mask)
        ##certain area!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        certain_area = cv2.cvtColor(certain_area, cv2.COLOR_BGR2HSV)
        targetImg = cv2.cvtColor(targetImg, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([certain_area], [0, 1], None, [2, 2], [1, 180, 0, 256])

        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

        backproj = cv2.calcBackProject([targetImg], [0, 1], hist, [0, 180, 0, 256], 1)
        (T, saliency) = cv2.threshold(backproj, 200, 255, cv2.THRESH_BINARY)
        cv2.imshow('backprojection result', saliency)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def on_btn3_1_click(self):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((11 * 8, 3), np.float32)
        objp[:, :2] = np.mgrid[0:8, 0:11].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        img = cv2.imread("./images/CameraCalibration/" + self.comboBox.currentText() + ".bmp")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (8, 11), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (8, 11), corners2, ret)
            cv2.imshow('img', img)

            cv2.waitKey(0)
            cv2.destroyAllWindows()


    def on_btn3_2_click(self):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((11 * 8, 3), np.float32)
        objp[:, :2] = np.mgrid[0:8, 0:11].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        images = glob.glob('./images/CameraCalibration/*.bmp')

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (8, 11), None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

        ret, cameraMtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], cameraMatrix= None, distCoeffs=None)
        print(cameraMtx)


    def on_btn3_3_click(self):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((11 * 8, 3), np.float32)
        objp[:, :2] = np.mgrid[0:8, 0:11].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        
        for order in range(1, 16):
            img = cv2.imread('./images/CameraCalibration/' + str(order) + '.bmp')
            #print('./images/CameraCalibration/' + str(order) + '.bmp')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (8, 11), None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], cameraMatrix= None, distCoeffs=None)
        #select which img
        i = int(self.comboBox.currentText()) - 1

        retval, rvec, tvec = cv2.solvePnP(objpoints[i], imgpoints[i], mtx, dist)

        rotateMat, _jacobian= cv2.Rodrigues(rvec)
        #print(rotateMat)
        print(np.hstack((rotateMat, tvec)))

        ####I thought rvec will be a 3X1 matrix of each axis
        ####so this is old way
        # c, s = np.cos(rvec[0]), np.sin(rvec[0])
        # rx = np.matrix([[1, 0, 0], [0, c, -s], [0, s, c]])
        # # print(rx)
        #
        # cy, sy = np.cos(rvec[1]), np.sin(rvec[1])
        # ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        # # print(ry)
        #
        # c, s = np.cos(rvec[2]), np.sin(rvec[2])
        # rz = np.matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        # # print(rz)
        #
        # rot = rx * ry * rz

        # trick method to dimish array
        # extrinsic = np.zeros((3, 3))
        # extrinsic[:, :] = rot
        # extrinsic = np.concatenate((extrinsic, tvec), axis=1)
        # print(extrinsic)


    def on_btn3_4_click(self):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((11 * 8, 3), np.float32)
        objp[:, :2] = np.mgrid[0:8, 0:11].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        images = glob.glob('./images/CameraCalibration/*.bmp')

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (8, 11), None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], cameraMatrix= None, distCoeffs=None)
        print(dist)

    def on_btn4_1_click(self):
        def draw(img, corners, imgpts):
            corner = tuple(corners[0].ravel())
            #lower horizontal line
            img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (0, 0, 255), 5)
            img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 0, 255), 5)
            img = cv2.line(img, tuple(imgpts[3].ravel()), tuple(imgpts[0].ravel()), (0, 0, 255), 5)
            img = cv2.line(img, tuple(imgpts[3].ravel()), tuple(imgpts[1].ravel()), (0, 0, 255), 5)
            #vertical line
            img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
            img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[4].ravel()), (0, 0, 255), 5)
            img = cv2.line(img, tuple(imgpts[1].ravel()), tuple(imgpts[5].ravel()), (0, 0, 255), 5)
            img = cv2.line(img, tuple(imgpts[3].ravel()), tuple(imgpts[6].ravel()), (0, 0, 255), 5)
            #higher horizontal line
            img = cv2.line(img, tuple(imgpts[2].ravel()), tuple(imgpts[4].ravel()), (0, 0, 255), 5)
            img = cv2.line(img, tuple(imgpts[2].ravel()), tuple(imgpts[5].ravel()), (0, 0, 255), 5)
            img = cv2.line(img, tuple(imgpts[6].ravel()), tuple(imgpts[4].ravel()), (0, 0, 255), 5)
            img = cv2.line(img, tuple(imgpts[6].ravel()), tuple(imgpts[5].ravel()), (0, 0, 255), 5)
            return img

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((11 * 8, 3), np.float32)
        objp[:, :2] = np.mgrid[0:8, 0:11].T.reshape(-1, 2)

        drawingDot = np.float32([[2, 0, 0], [0, 2, 0], [0, 0, -2], [2, 2, 0], [2, 0, -2], [0, 2, -2], [2, 2, -2]]).reshape(-1, 3)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        images = glob.glob('./images/CameraCalibration/*.bmp')


        ##get rotational matrix and distorsion matrix
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (8, 11), None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

        ret, cameraMtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], cameraMatrix= None, distCoeffs=None)

        #AR!
        for order in range(1,6):
            img = cv2.imread('./images/CameraCalibration/' + str(order) + '.bmp')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (8, 11), None)
            if ret == True:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                # Find the rotation and translation vectors.
                ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, cameraMtx, dist)
                # project 3D points to image plane
                imgpts, jac = cv2.projectPoints(drawingDot, rvecs, tvecs, cameraMtx, dist)
                img = draw(img, corners2, imgpts)
                img = cv2.flip(img, 1)
                cv2.imshow('img', img)
                cv2.waitKey(500)
        cv2.waitKey()
        cv2.destroyAllWindows()
    ### ### ###


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
