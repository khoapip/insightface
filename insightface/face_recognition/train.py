import face_model
import numpy as np
import pickle
import os 
import cv2

directory = "faces/tmp"
path = '../models/model-y1-test2/model,0' 
model_name = path.split('/')[2]
vec = path.split(',')
model_prefix = vec[0]
model_epoch = int(vec[1])
model = face_model.FaceModel(-1, model_prefix, model_epoch)
embedding = np.zeros((0,128))
names = []

for filename in os.listdir(directory):
    for imagename in os.listdir(os.path.join(directory, filename)):
        image = cv2.imread(os.path.join(directory,filename, imagename))
        added = model.get_input(np.array(image))
        added = model.get_feature(added)
        print(added.shape)
        added = np.array([added])
        embedding = np.concatenate((embedding, added),0)
        names.append(filename)

with open("embeddings/embeddings_{}.pkl".format(model_name), "wb") as f:
    pickle.dump((embedding, names), f)
print("Done!")
