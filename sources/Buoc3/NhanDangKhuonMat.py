import sys
import tkinter
from tkinter import Frame, Tk, BOTH, Text, Menu, END
from tkinter.filedialog import Open, SaveAs

import numpy as np
import os.path
import cv2
import joblib

from sklearn.svm import LinearSVC

detector = cv2.FaceDetectorYN.create(
    "face_detection_yunet_2022mar.onnx",
    "",
    (320, 320),
    0.9,
    0.3,
    5000
)

# detector.setInputSize((960, 1280))

recognizer = cv2.FaceRecognizerSF.create(
            "face_recognition_sface_2021dec.onnx","")

svc = joblib.load('svc.pkl')
mydict = ['BanAnh', 'BanBao','BanDat', 'BanDien', 'BanKy', 'BanNam', 'BanNinh', 'BanSang', 'BanThanh', 'BanTuan', 'DucHoa', 'LeTai', 'BanSon', 'SongHuy', 'ThayDuc']

class Main(Frame):
    
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.initUI()
  
    def initUI(self):
        self.parent.title("Nhan Dang Khuon Mat")
        self.pack(fill=BOTH, expand=1)
  
        menubar = Menu(self.parent)
        self.parent.config(menu=menubar)
  
        fileMenu = Menu(menubar)
        fileMenu.add_command(label="Open", command=self.onOpen)
        fileMenu.add_command(label="Recognition", command=self.onRecognition)
        fileMenu.add_separator()
        fileMenu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=fileMenu)
        self.txt = Text(self)
        self.txt.pack(fill=BOTH, expand=1)
  
    def onOpen(self):
        global ftypes
        ftypes = [('Images', '*.jpg *.tif *.bmp *.gif *.png')]
        dlg = Open(self, filetypes = ftypes)
        fl = dlg.show()
  
        if fl != '':
            global imgin
            imgin = cv2.imread(fl,cv2.IMREAD_COLOR)
            imgin = cv2.resize(imgin,(500,500))
            x=imgin.shape[0]
            y=imgin.shape[1]
            detector.setInputSize((y, x))
            cv2.namedWindow("ImageIn", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("ImageIn", imgin)

    def onRecognition(self):
        faces = detector.detect(imgin)
        face_align = recognizer.alignCrop(imgin, faces[1][0])
        face_feature = recognizer.feature(face_align)
        test_prediction = svc.predict(face_feature)

        result = mydict[test_prediction[0]]
        cv2.putText(imgin,result,(5,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
        cv2.namedWindow("ImageIn", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("ImageIn", imgin)

root = Tk()
Main(root)
root.geometry("480x480+100+100")
root.mainloop()
