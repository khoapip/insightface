import face_model
import argparse
import cv2
import sys
import numpy as np
import time

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='../models/model-r100-ii/model,0', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
args = parser.parse_args()

vec = args.model.split(',')
model_prefix = vec[0]
model_epoch = int(vec[1])
model = face_model.FaceModel(args.gpu, model_prefix, model_epoch)
img = cv2.imread('faces/tmp/Chandler_0.jpg')
start = time.time()
img = model.get_input(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.array([img,img])
end = time.time()

start = time.time()
f1 = model.get_feature(img)
end = time.time()
print("Extracting feature takes: {}s".format(end - start))
f2 = model.get_feature(img)
sim = np.dot(f1, f2)
assert(sim>=0.99 and sim<1.01)

