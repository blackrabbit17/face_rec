from datetime import datetime
import face_recognition
import numpy as np
import math
import os
import pickle
from scipy.spatial import KDTree

def find_nearest_vector(array, value):
  idx = np.array([np.linalg.norm(x+y) for (x,y) in array-value]).argmin()
  return array[idx]

#
# Load the encodings array and the filenames from the previous step
#
started = datetime.now()
encodings_path = os.path.join('data', '01_encodings.npz')
filenames_path = os.path.join('data', '01_encodings_filenames.pkl')
encodings = np.load(open(encodings_path, 'rb'))['arr_0']
filenames = pickle.load(open(filenames_path, 'rb'))
print('Load encodings :', (datetime.now() - started).total_seconds(), 's')

#
# Load the encodings array and the filenames from the previous step
#
# For more about a KD-Tree look here: https://en.wikipedia.org/wiki/K-d_tree
#
started = datetime.now()
kdtree = KDTree(encodings)
print('Build KDTree   :', (datetime.now() - started).total_seconds(), 's')

#
# Prepare our query by loading the image and converting to encodings
#
started = datetime.now()
query_image = face_recognition.load_image_file('knuth_query.png')
query_encodings = np.array(face_recognition.face_encodings(query_image))
print('Prepare query  :', (datetime.now() - started).total_seconds(), 's')

#
# Now the actual query - Search for the image inside our large dataset
#
started = datetime.now()
distance, index = kdtree.query(query_encodings)
took = (datetime.now() - started).total_seconds()
query_rate = 1/took * 52000  # 52000 images in dataset
print('Query          :', '{:f}'.format(took), 's', 'Rate: {:f}'.format(query_rate), '/s\n')

#
# Print the closest match
#
print('Distance :', distance)
print('Index    :', index)
print('Image    :', filenames[index[0]])
