
# coding: utf-8

## openCV（Haar-like特徴）写真を保存

import cv2
import numpy as np
import os

# 分類器(以下から取得)
# https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

# Haar-like特徴分類器へのパス
cascade_path = './haarcascade_fullbody.xml'

# 入出力フォルダ
input_dir = './image/'
output_dir = './image_out/'

# 分類器の特徴量を取得する
faceCascade = cv2.CascadeClassifier(cascade_path)

# input_dir 下にある画像をすべて処理する
files = os.listdir(input_dir) 
for file in files:
    # ファイル読み込み
    image = cv2.imread(input_dir + file,cv2.IMREAD_COLOR)

    finalHeight = 800.0
    scale = finalHeight / image.shape[0]
    image = cv2.resize(image, None, fx=scale, fy=scale)

    # 物体認識（人）の実行
    facerect = faceCascade.detectMultiScale(image, scaleFactor=1.01, minNeighbors=1, minSize=(30, 30))

    #検出した人を囲む矩形の作成
    for x, y, w, h in facerect:    
        cv2.rectangle(image, (x, y),(x+w, y+h),(0,255,0), 2)

    # ファイルを保存
    cv2.imwrite(output_dir + file, image)

