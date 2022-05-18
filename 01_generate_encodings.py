import face_recognition
import glob
import numpy as np
import pickle
from tqdm import tqdm

NUM_FACES = 52000
encodings = []
filenames = []

# Face recognition is done with dlibs face rec DNN
# http://dlib.net/

# Note this can go faster if you rebuilt it with USE_SSE2 enabled (if your CPU supports that)
# for more info you can read here: http://dlib.net/face_landmark_detection_ex.cpp.html


with tqdm(total=NUM_FACES) as pbar:

    # 
    # Dataset is here:
    # https://storage.googleapis.com/kaggle-data-sets/546691/997012/bundle/archive.zip
    # If you get an access denied error, signup at kaggle.com and find the dataset there
    # 

    for face_file in glob.glob('faces/*.png'):

        image = face_recognition.load_image_file(face_file)
        face_encoding = face_recognition.face_encodings(image)

        # Did we detect at least 1 face? 
        if len(face_encoding) > 0:
            encodings.append(face_encoding[0])
            filenames.append(face_file)

        pbar.update(1)

np_encodings = np.array(encodings)

np.savez(open('data/01_kaggle_encodings.npz', 'wb'), np_encodings)
pickle.dump(filenames, open('data/01_kaggle_encodings_filenames.pkl', 'wb'))
