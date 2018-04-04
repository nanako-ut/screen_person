
# coding: utf-8

# ## openCV（HOG特徴量）写真を保存


import numpy as np
import cv2
import sys
import os

# 入出力フォルダ
input_dir = './image/'
output_dir = './image_out/'

# input_dir 下にある画像をすべて処理する
files = os.listdir(input_dir) 
for file in files:
    # ファイル読込
    image = cv2.imread(input_dir + file,cv2.IMREAD_COLOR)

    hog = cv2.HOGDescriptor()
    hog = cv2.HOGDescriptor((48,96), (16,16), (8,8), (8,8), 9)

    # SVMによる人検出
    hog.setSVMDetector(cv2.HOGDescriptor_getDaimlerPeopleDetector())

    # リサイズした方が精度が高まる
    finalHeight = 800.0
    scale = finalHeight / image.shape[0]
    image = cv2.resize(image, None, fx=scale, fy=scale)

    # 人を検出した座標
    human, r = hog.detectMultiScale(image, hitThreshold = 0.6, winStride = (8,8), padding = (32, 32), scale = 1.05, finalThreshold=2)

    # 全員のバウンディングボックスを作成
    for (x, y, w, h) in human:
        cv2.rectangle(image, (x, y),(x+w, y+h),(0,255,0), 2)
    
    # ファイルを保存
    cv2.imwrite(output_dir + file, image)
    
