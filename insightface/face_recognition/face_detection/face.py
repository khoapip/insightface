# -*- coding: utf-8 -*-
# @Author: fyr91
# @Date:   2019-10-22 15:05:15
# @Last Modified by:   fyr91
# @Last Modified time: 2019-10-31 12:38:02
import os.path
from os import path
import cv2
import dlib
import numpy as np
from imutils import face_utils
from .box_utils import *
from PIL import Image


import onnx
import onnxruntime as ort
from onnx_tf.backend import prepare

def face_detect(path, name):

    video_capture = cv2.VideoCapture(0)

    onnx_path = 'face_detection/UltraLight/models/ultra_light_640.onnx'
    onnx_model = onnx.load(onnx_path)
    predictor = prepare(onnx_model)
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name

    shape_predictor = dlib.shape_predictor('face_detection/FacialLandmarks/shape_predictor_5_face_landmarks.dat')
    fa = face_utils.facealigner.FaceAligner(shape_predictor, desiredFaceWidth=112, desiredLeftEye=(0.3, 0.3))
    iteration = 1
    while iteration >= 1:
        ret, frame = video_capture.read()
        if frame is not None:
            h, w, _ = frame.shape

            # preprocess img acquired
            img_org = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert bgr to rgb
            img_org = cv2.resize(img_org, (640, 480)) # resize
            img_mean = np.array([127, 127, 127])
            img = (img_org - img_mean) / 128
            img = np.transpose(img, [2, 0, 1])
            img = np.expand_dims(img, axis=0)
            img = img.astype(np.float32)

            confidences, boxes = ort_session.run(None, {input_name: img})
            boxes, labels, probs = predict(w, h, confidences, boxes, 0.7)
            

            for i in range(boxes.shape[0]):
                
                box = boxes[i, :]
                x1, y1, x2, y2 = box
                if boxes.shape[0] <= 1:
                    print(im.shape)
                    im = cv2.resize(frame[y1:y2,x1:x2, :], (112,112))
                    im = cv2.imwrite(os.path.join(path, name +str(iteration)+
                        ".jpg"),im)

                else:
                    im = cv2.resize(frame[y1:y2,x1:x2, :], (112,112))
                    im = cv2.imwrite(os.path.join(path.rstrip(name), "other",
                        name +str(iteration) + "(" + str(i) + ")" ".jpg"),im)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                shape = shape_predictor(gray, dlib.rectangle(left = x1, top=y1, right=x2, bottom=y2))
                shape = face_utils.shape_to_np(shape)
                for (x, y) in shape:
                    cv2.circle(frame, (x, y), 2, (80,18,236), -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (80,18,236), 2)
                cv2.rectangle(frame, (x1, y2 - 20), (x2, y2), (80,18,236), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                text = f"Face: {round(probs[i], 2)}"
                cv2.putText(frame, text, (x1 + 6, y2 - 6), font, 0.3, (255, 255, 255), 1)
            cv2.imshow('Video', frame)
            iteration +=1

            # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

def createFolder(path, name):
    index = ''
    while True:
        try:
            file_path = os.path.join(path, name+index)
            os.makedirs(file_path)
            return file_path
        except:
            if index:
                index = '('+str(int(index[1:-1])+1)+')' # Append 1 to number in brackets
            else:
                index =  '(1)'
            pass # Go and try create file again


if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    os.makedirs(os.path.join('data','other'), exist_ok=True)



    name = input("Dien ten: ")
    name = name.lower()
    name = name.replace(" ", "")

    file_path = createFolder('data',name)
    face_detect(file_path, name)

    

    



