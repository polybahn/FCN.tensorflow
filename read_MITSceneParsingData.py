__author__ = 'charlie'
import numpy as np
import os
import random
from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile
import glob
import cv2

import TensorflowUtils as utils

# DATA_URL = 'http://sceneparsing.csail.mit.edu/data/ADEChallengeData2016.zip'
# DATA_URL = 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'
DATA_PREFIX = 'http://synthia-dataset.cvc.uab.cat/SYNTHIA_SEQS/SYNTHIA-SEQS-04-'
DATA_NAMES = ['DAWN','FALL','FOG','NIGHT','SUNSET','WINTER']
DATA_SUFFIX = '.rar'

IMG_ORIS = ['Stereo_Left', 'Stereo_Right']
CAMERA_ORIS = ['Omni_B', 'Omni_F', 'Omni_L', 'Omni_R']

def read_dataset(data_dir):
    pickle_filename = "SYNTHIA_SEQ4.pickle"
    pickle_filepath = os.path.join(data_dir, pickle_filename)
    if not os.path.exists(pickle_filepath):
        image_dirs = []
        for name in DATA_NAMES:
            # download data and extract
            data_url = DATA_PREFIX + name + DATA_SUFFIX
            utils.maybe_download_and_extract(data_dir, data_url, is_rarfile=True)
            # get the folder name
            synthia_folder = os.path.splitext(data_url.split("/")[-1])[0]
            image_dirs.append(os.path.join(data_dir, synthia_folder))

        result = create_image_lists(image_dirs)
        print ("Pickling ...")
        # dump pickle data for image names
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
    else:
        print ("Found pickle file!")

    with open(pickle_filepath, 'rb') as f:
        result = pickle.load(f)
        training_records = result['training']
        validation_records = result['validation']
        del result

    return training_records, validation_records


def create_image_lists(image_dirs):
    # type: list -> dict
    '''
    Create image name list dictionary
    :param image_dirs: the directories which contains all images
    :return:{training -> {image -> image path
                          annotation -> annotation path
                          filename -> file name
                          }
            validation -> {same here...}
            }
    '''
    obj_list = []

    # for each directory we do something ...
    for image_dir in image_dirs:
        if not gfile.Exists(image_dir):
            print("Image directory '" + image_dir + "' not found.")
            continue

        for img_ori in IMG_ORIS:
            for camera_ori in CAMERA_ORIS:
                rgb_list = []
                file_glob = os.path.join(image_dir, "RGB", img_ori, camera_ori, '*.' + 'png')
                rgb_list.extend(glob.glob(file_glob))

                if not rgb_list:
                    print('No files found')
                else:
                    for f in rgb_list:
                        filename = os.path.splitext(f.split("/")[-1])[0]
                        annotation_file = os.path.join(image_dir, 'GT', 'LABELS', img_ori, camera_ori, filename + '.png')
                        if os.path.exists(annotation_file):
                            record = {'image': f, 'annotation': annotation_file, 'filename': filename}
                            obj_list.append(record)
                        else:
                            print("Annotation file not found for %s - Skipping" % filename)

    # random shuffle
    random.shuffle(obj_list)
    # construct resulting images list
    image_list = {}
    split_pos = int(len(obj_list) * 0.9)
    image_list['training'] = obj_list[:200]
    image_list['validation'] = obj_list[200:230]
    no_of_images = len(obj_list)
    print('No. of all files: %d' % no_of_images)
    print('No. of training files: %d' % len(image_list['training']))
    print('No. of validation files: %d' % len(image_list['validation']))

    return image_list
