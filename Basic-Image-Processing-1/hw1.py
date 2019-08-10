# -*- coding: utf-8 -*-

import sys
from hw1_ui import Ui_MainWindow
import cv2
import os
import sys
import numpy as np
import tkinter as tk
import PIL.Image, PIL.ImageTk
import math
from PyQt5.QtWidgets import QMainWindow, QApplication

def nothing(x):
    pass


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

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.onBindingUI()

    # Write your code below
    # UI components are defined in hw1_ui.py, please take a look.
    # You can also open hw1.ui by qt-designer to check ui components.

    def onBindingUI(self):
        self.btn1_1.clicked.connect(self.on_btn1_1_click)
        self.btn1_2.clicked.connect(self.on_btn1_2_click)
        self.btn1_3.clicked.connect(self.on_btn1_3_click)
        self.btn1_4.clicked.connect(self.on_btn1_4_click)
        self.btn2_1.clicked.connect(self.on_btn2_1_click)
        self.btn3_1.clicked.connect(self.on_btn3_1_click)
        self.btn4_1.clicked.connect(self.on_btn4_1_click)
        self.btn4_2.clicked.connect(self.on_btn4_2_click)
        self.btn5_1.clicked.connect(self.on_btn5_1_click)
        self.btn5_2.clicked.connect(self.on_btn5_2_click)

    # button for problem 1.1
    def on_btn1_1_click(self):
        img = cv2.imread('dog.bmp')
        cv2.imshow("Doggy", img)
        print("Height = " + str(img.shape[0]))
        print("Width = " + str(img.shape[1]))
        print(img.shape)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

    def on_btn1_2_click(self):
        img = cv2.imread('color.png')
        cv2.imshow("Color", img)
        newImg = np.zeros(img.shape, np.uint8)
        mat = np.matrix('0 0 1; 1 0 0; 0 1 0')   #pixel color order [B, G, R]
        h, w, no_channels = img.shape

        for py in range(0,h):
            for px in range(0,w):
                    newImg[py][px] = img[py][px]  * mat
        cv2.imshow("Color2", newImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        
    def on_btn1_3_click(self):
        img = cv2.imread('dog.bmp')
        cv2.imshow("Doggy", img)
        newImg = np.zeros(img.shape, np.uint8)
        h, w, no_channels = img.shape
         
        for py in range(0,h):
            for px in range(0,w):
                    newImg[py][h - px] = img[py][px]
        cv2.imshow("Doggy2", newImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def on_btn1_4_click(self):
        img = cv2.imread('dog.bmp')
        flipImg = np.zeros(img.shape, np.uint8)
        h, w, no_channels = img.shape
        
        #flip original img
        for py in range(0,h):
            for px in range(0,w):
                    flipImg[py][h - px] = img[py][px]
        
        cv2.namedWindow('Blending(Press Esc to exit)')
        cv2.createTrackbar('Blend','Blending(Press Esc to exit)',0,100,nothing)
                
        #create a newImg which is a blend of img & flipImg
        newImg = np.zeros(img.shape, np.uint8)
        while(1):
            percent = cv2.getTrackbarPos('Blend', 'Blending(Press Esc to exit)') / 100
            reversePercent = 1 - percent
            cv2.addWeighted(img, percent, flipImg, reversePercent, 0, newImg)
            cv2.imshow('Blending(Press Esc to exit)', newImg)
            k = cv2.waitKey(1) & 0xFF #i dont get it
            if k == 27: #esc
                break;
                
        cv2.destroyAllWindows()
        
    def on_btn2_1_click(self):
        img = cv2.imread('m8.jpg')
        h, w, no_channels = img.shape

        # broaden the img for gaussion filter
        gauImg = np.zeros([h + 2, w + 2, 1], np.uint8)#two for extra
        print("shape of gauImg" )
        print(gauImg.shape)
        print()
        RGBtoGray = np.matrix('0.114;0.587;0.299')
        for py in range(0, h):
            for px in range(0, w):
                gauImg[py + 1][px + 1] = img[py][px] * RGBtoGray
        # cv2.imshow('broaden_1_pixel', gauImg)
        
        # smoothImg = cv2.GaussianBlur(gauImg, (3, 3), 0)
        #smooooooooooooth
        gauMatrix = np.matrix('1 2 1; 2 4 2; 1 2 1')
        gauMatrix = gauMatrix / 16
        smoothImg = np.zeros([h, w, 1], np.uint8)
        for py in range(1, h + 1):
           for px in range(1, w + 1):
                #gauImg & gauMatrix multiply element-wisely
                smoothImg[py - 1][px - 1] = np.sum(np.multiply(np.squeeze(gauImg[py - 1 : py + 2, px - 1 : px + 2], axis = 2), gauMatrix))
        # cv2.imshow('smoothed', smoothImg)

        threshold = 40
        #vertical edges
        Ix = np.matrix('-1 0 1; -2 0 2; -1 0 1')
        verImg = np.zeros([h - 2, w - 2, 1], np.uint8)
        verImg2 = np.zeros([h - 2, w - 2, 1], np.int32)
        for py in range(0, h - 2):   #smoothImg is original size(h, w)
            for px in range(0, w - 2):
                verImg2[py][px] = np.sum(np.multiply(np.squeeze(smoothImg[py : py + 3, px : px + 3], axis = 2), Ix))
        cv2.convertScaleAbs(verImg2,verImg)
        verImg = (verImg > threshold) * verImg
        cv2.imshow('verImg', verImg)

        #horizontal edges
        Ix = np.matrix('-1 -2 -1; 0 0 0; 1 2 1')
        horImg = np.zeros([h - 2, w - 2, 1], np.uint8)
        horImg2 = np.zeros([h - 2, w - 2, 1], np.int32)
        for py in range(0, h - 2):   #smoothImg is original size(h, w)
            for px in range(0, w - 2):
                horImg2[py][px] = np.sum(np.multiply(np.squeeze(smoothImg[py : py + 3, px : px + 3], axis = 2), Ix))
        cv2.convertScaleAbs(horImg2, horImg)
        horImg = (horImg > threshold) * horImg
        cv2.imshow('horImg', horImg)

        #magnitude
        magnitude = np.zeros([h - 2, w - 2, 1], np.uint8)
        magnitude2 = np.zeros([h - 2, w - 2, 1], np.int32)
        for py in range(0, h - 2):
            for px in range(0, w - 2):
                magnitude2[py][px] = np.sqrt(np.square(verImg2[py][px]) + np.square(horImg2[py][px]))
        cv2.convertScaleAbs(magnitude2, magnitude)
        cv2.imshow('Magnitude', magnitude)

        # trackbar for magnitude
        def showMagnitude(x):
            magnitude2 = np.zeros([h - 2, w - 2, 1], np.uint8)
            magThreshold = cv2.getTrackbarPos('magnitude', 'Magnitude')
            magnitude2 = (magnitude > magThreshold) * magnitude
            cv2.imshow('Magnitude', magnitude2)
        cv2.namedWindow('Magnitude', cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar('magnitude', 'Magnitude', 40, 255, showMagnitude)    #display wrong value beside it

        #angle image
        angle = np.zeros([h - 2, w - 2, 1], np.uint8)
        for py in range(0, h - 2):
            for px in range(0, w - 2):
                if(verImg2[py][px] == 0):
                    angle[py][px] = 0
                else:
                    angle[py][px] = int(math.degrees(np.arctan(horImg2[py][px] / verImg2[py][px]) + np.finfo(float).eps))

        #deal with tkinter which is to show angle
        angleWindow = tk.Tk()
        angleWindow.title('Angle')
        e = tk.Entry(angleWindow, show = None)
        e.pack()
        height, width, no_channels = angle.shape
        canvas = tk.Canvas(angleWindow, width=width, height=height)
        canvas.pack()
        angle_Mag = np.zeros([h - 2, w - 2, 1], np.uint8)
        angle_Mag = (angle <= 20) * magnitude - (angle < 10) * magnitude
        cv_img = cv2.cvtColor(angle_Mag, cv2.COLOR_GRAY2RGB)
        photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv_img))
        canvas.create_image(0, 0, image=photo, anchor=tk.NW)

        def func(event):
            canvas.delete()
            angleThreshold = int(e.get())
            if(angleThreshold > 180):
                angleThreshold = angleThreshold - 180
            e.delete(0,10)
            angle_Mag = (angle <= angleThreshold + 10) * magnitude - (angle < angleThreshold - 10) * magnitude
            cv_img = cv2.cvtColor(angle_Mag, cv2.COLOR_GRAY2RGB)
            photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv_img))
            canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            angleWindow.mainloop()
        angleWindow.bind('<Return>', func)
        angleWindow.mainloop()

        cv2.destroyAllWindows()

    def on_btn3_1_click(self):

        level0Origin = cv2.imread('pyramids_Gray.jpg')
        level1Origin = cv2.pyrDown(level0Origin)
        level2Origin = cv2.pyrDown(level1Origin)

        h, w, no_channels = level0Origin.shape

        level2to1 = cv2.pyrUp(level2Origin)
        level1to0 = cv2.pyrUp(level1Origin)


        level1Edge = np.subtract(level1Origin, level2to1)
        level0Edge = np.subtract(level0Origin, level1to0)

        level1UP = level2to1 + level1Edge
        level0UP = level1to0 + level0Edge

        cv2.imshow('Gaussian(3x3) pyramid level 1', level1Origin)
        cv2.imshow('Laplacian pyramid level 0', level0Edge)
        cv2.imshow('Inverse pyramid level 1', level1UP)
        cv2.imshow('Inverse pyramid level 0', level0UP)

        cv2.waitKey()
        cv2.destroyAllWindows()

    def on_btn4_1_click(self):

        img = cv2.imread('QR.png', 0)
        gloThreshImg = np.zeros(img.shape, np.uint8)

        _null, gloThreshImg = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)

        cv2.imshow('Original Image', img)
        cv2.imshow('Threshold Image', gloThreshImg)

        cv2.waitKey()
        cv2.destroyAllWindows()

    def on_btn4_2_click(self):

        img = cv2.imread('QR.png', 0)
        adaThreshImg = np.zeros(img.shape, np.uint8)

        adaThreshImg = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 19, -1)

        cv2.imshow('Original Image', img)
        cv2.imshow('Adaptive threshold Image', adaThreshImg)

        cv2.waitKey()
        cv2.destroyAllWindows()

    def on_btn5_1_click(self):
        # edtAngle, edtScale. edtTx, edtTy to access to the ui object

        print(self.edtAngle.text())
        print()
        oriImg = cv2.imread('OriginalTransform.png')

        #get contours
        grayImg = cv2.cvtColor(oriImg, cv2.COLOR_BGR2GRAY)
        _null, threshImg = cv2.threshold(grayImg, 100, 255, cv2.THRESH_BINARY)
        im2, contours, hierarchy = cv2.findContours(threshImg, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

        #extract square, get central point
        centralx, centraly = (contours[0][0][0][1] + contours[0][2][0][1]) / 2, (contours[0][0][0][0] + contours[0][2][0][0]) / 2

        M = cv2.getRotationMatrix2D((centralx, centraly), int(self.edtAngle.text()), float(self.edtScale.text()))#degree, scale
        M2 = np.float32([[1, 0, int(self.edtTx.text())], [0, 1, int(self.edtTy.text())]])#translation

        h, w = oriImg.shape[:2]
        after_rotate_Img = cv2.warpAffine(oriImg, M, (w, h))
        after_moving_Img = cv2.warpAffine(after_rotate_Img, M2, (w, h))


        cv2.imshow('Original Image', oriImg)
        cv2.imshow('Rotation + Scale + Translation Image', after_moving_Img)

        cv2.waitKey()
        cv2.destroyAllWindows()

    def on_btn5_2_click(self):

        oriImg = cv2.imread('OriginalPerspective.png')

        cv2.namedWindow('Original Perspective')
        cv2.imshow('Original Perspective', oriImg)
        point1 = []
        point2 = np.float32([[20, 20], [450, 20], [450, 450], [20, 450]])

        def myOnMouse(event, x, y, flags, param):
            if(event == cv2.EVENT_LBUTTONDOWN):
                point1.append([x, y])
                print(point1[:])
                if(len(point1) == 4):
                    draw()
        cv2.setMouseCallback('Original Perspective', myOnMouse)


        def draw():
            point3 = np.float32(point1)
            M = cv2.getPerspectiveTransform(point3, point2)
            newImg = cv2.warpPerspective(oriImg, M, (450, 450))
            cv2.imshow('Perspective Image', newImg)
            cv2.waitKey()
            cv2.destroyAllWindows()

    ### ### ###


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
