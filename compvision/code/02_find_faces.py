"""
This code reads the actor-pictures, looks for faces and outputs the 
face info to 200x200 grayscale images, stored in the data/faces file.

After this step, one can select test and training pictures 
"""

import os
import cv2

# base_path = "/content/sample_data/CV__Group_assignment"
base_path = '..//data//'

if not(os.path.exists(base_path + "//haarcascade_frontalface_default.xml")):
    print("00_download_files first: haarcascade_frontalface_default NOT FOUND")

face_cascade = cv2.CascadeClassifier(
    base_path + "//haarcascade_frontalface_default.xml")

aLL_actors_dir = base_path + "actors//"
aLL_faces_dir = base_path + "faces//"

all_actors = [subject for subject in sorted(os.listdir(aLL_actors_dir))]

for actor in all_actors:
    actor_dir = aLL_actors_dir + actor
    face_dir = aLL_faces_dir + actor
    if not os.path.isdir(face_dir):
      os.makedirs(face_dir)
    all_files = [subject for subject in sorted(os.listdir(os.path.join(actor_dir))) if subject.endswith(".jpg")]
    for file in all_files:
        fileName = os.path.join(actor_dir,file)
        face_filename = os.path.join(face_dir,file)
        image = cv2.imread(fileName)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, 1.3, 5, minSize=(120, 120))
        for (x, y, w, h) in faces:
            face_img = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            # face_filename = '%s/%d.pgm' % (output_folder, count)
            cv2.imwrite(face_filename, face_img)
        