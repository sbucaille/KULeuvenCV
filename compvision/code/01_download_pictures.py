"""
Downloads the jpg files of some selected actors to the directory //data//actors
the dictionary fileDict select the actor file (value) and the target directory name (key)

Dependent on the amount of requested pictures and actors, the execution can take a while...
if the target directory of an actor is present, no downloading is done and a WARN is given
delete the directory if you want more pictures for that actor...
"""
import os
import cv2
import numpy as np
from urllib import request

# base_path = "/content/sample_data/CV__Group_assignment"
base_path = '..//data//'
fileDict = {"ac": "Alice_Greczyn", "bf": "Ben_Feldman", "cl": "Caity_Lotz", "df":"David_Faustino"}
nb_images_per_actor = 20


fileNr = 0
def fileNameGen(prefixIn = "file", suffixIn = ".jpg"):
    global fileNr
    fileNr += 1
    return "{0}{1:05d}{2}".format(prefixIn, fileNr, suffixIn)


if os.path.exists(base_path + "//vgg_face_dataset.tar.gz"):
    print("Image files already available")
else:
    print("00_download_index first")


images = []
for key, value in fileDict.items():
    dirName = base_path + "actors//" + key + "//"
    if os.path.isdir(dirName):
        print("WARN: Actor file is present")
        continue
    else:
        os.makedirs(dirName)
        actorFile = base_path + "vgg_face_dataset//files//" + value + ".txt"
        if not(os.path.exists):
            print("ERROR: Actor file NOT present: ", actorFile)
            continue
    with open(actorFile, 'r') as f:
        lines = f.readlines()
    images_ = []
    for line in lines:
        url = line[line.find("http://"): line.find(".jpg") + 4]
        try:
            res = request.urlopen(url)
            img = np.asarray(bytearray(res.read()), dtype="uint8")
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            images_.append(img)
            # cv2_imshow(cv2.resize(img, (w // 5, h // 5)))
            cv2.imwrite(fileNameGen(prefixIn=dirName),img)
        except:
            pass
        if len(images_) == nb_images_per_actor:
            images.append(images_)
            break
        
