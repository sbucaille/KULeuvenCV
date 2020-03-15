"""
This program takes gray scale images (200x200) from the training subdirectories
and trains an EigenFaceRecognizer (or alternative). 

"""

import os
import cv2
import numpy

# base_path = "/content/sample_data/CV__Group_assignment"
base_path = '..//data//'

train_dir = base_path + "train//"
test_dir = base_path + "test//"

def read_images(path, image_size):
    names = []
    training_images, training_labels = [], []
    label = 0
    for dirname, subdirnames, filenames in os.walk(path):
        for subdirname in subdirnames:
            names.append(subdirname)
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                img = cv2.imread(os.path.join(subject_path, filename),
                                 cv2.IMREAD_GRAYSCALE)
                if img is None:
                    # The file cannot be loaded as an image.
                    # Skip it.
                    continue
                img = cv2.resize(img, image_size)
                training_images.append(img)
                training_labels.append(label)
            label += 1
    training_images = numpy.asarray(training_images, numpy.uint8)
    training_labels = numpy.asarray(training_labels, numpy.int32)
    return names, training_images, training_labels


training_image_size = (200, 200)
names, training_images, training_labels = read_images(
    train_dir, training_image_size)

print("Start Training.")
model = cv2.face.EigenFaceRecognizer_create()
# model = cv2.face.FisherFaceRecognizer_create()
# model = cv2.face.LBPHFaceRecognizer_create()

model.train(training_images, training_labels)
print("End Training")

face_cascade = cv2.CascadeClassifier(
    './cascades/haarcascade_frontalface_default.xml')


for filename in os.listdir(test_dir):
    img = cv2.imread(os.path.join(test_dir, filename),
                     cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    label, confidence = model.predict(img)
    print (filename, "\t",label , "\t", confidence)
    
