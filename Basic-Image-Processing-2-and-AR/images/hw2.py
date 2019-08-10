# -*- coding: utf-8 -*-

import sys
from hw2_ui import Ui_Dialog
import cv2
import os
import sys
import numpy as np
import math
from PyQt5.QtWidgets import QMainWindow, QApplication


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
        img = cv2.imread("plant.jpg")
        cv2.imshow("Histogram", img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def on_btn1_2_click(self):
        print('btn1_2');
        
    def on_btn2_1_click(self):
        print('btn2_1');
        
    def on_btn2_2_click(self):
        print('btn2_1');
        
    def on_btn2_3_click(self):
        print('btn2_1');

    def on_btn3_1_click(self):
        print('btn2_1');

    def on_btn3_2_click(self):
        print('btn2_1');

    def on_btn3_3_click(self):
        print('btn2_1');


    def on_btn3_4_click(self):
        print('btn2_1');

    def on_btn4_1_click(self):
        print('btn2_1');


    ### ### ###


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
