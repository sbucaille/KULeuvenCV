"""
Downloads 2 files 
- vgg_face_dataset.tar.gz => expands to textfiles
- haarcascade_frontalface_default.xml => needed to detect faces
"""

import os
import tarfile
from urllib import request


# base_path = "/content/sample_data/CV__Group_assignment"
base_path = '..//data//'

if os.path.exists(base_path + "//vgg_face_dataset.tar.gz"):
    print("Image files already available")
else:
    print("downloading files to basepath")
    if not os.path.isdir(base_path):
      os.makedirs(base_path)
    vgg_face_dataset_url = "http://www.robots.ox.ac.uk/~vgg/data/vgg_face/vgg_face_dataset.tar.gz"
    with request.urlopen(vgg_face_dataset_url) as r, open(os.path.join(base_path, "vgg_face_dataset.tar.gz"), 'wb') as f:
      f.write(r.read())
    with tarfile.open(os.path.join(base_path, "vgg_face_dataset.tar.gz")) as f:
      f.extractall(os.path.join(base_path))

if os.path.exists(base_path + "//haarcascade_frontalface_default.xml"):
    print("haarcascade already available")
else:
    print("downloading haarcascade to basepath")
    trained_haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    with request.urlopen(trained_haarcascade_url) as r, open(os.path.join(base_path, "haarcascade_frontalface_default.xml"), 'wb') as f:
        f.write(r.read())

