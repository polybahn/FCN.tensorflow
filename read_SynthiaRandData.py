__author__ = 'charlie'
import numpy as np
import os
import random
from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile
import glob

import TensorflowUtils as utils

# DATA_URL = 'http://sceneparsing.csail.mit.edu/data/ADEChallengeData2016.zip'
DATA_URL = 'http://synthia-dataset.cvc.uab.cat/SYNTHIA_RAND_CVPR16.zip'


def read_dataset(data_dir):
    pickle_filename = "SynthiaRand.pickle"
    pickle_filepath = os.path.join(data_dir, pickle_filename)
    if not os.path.exists(pickle_filepath):
        utils.maybe_download_and_extract(data_dir, DATA_URL, is_zipfile=True)
        SceneParsing_folder = os.path.splitext(DATA_URL.split("/")[-1])[0]
        result = create_image_lists(os.path.join(data_dir, SceneParsing_folder))
        print ("Pickling ...")
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


def create_image_lists(image_dir):
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None

    file_list = []
    file_glob = os.path.join(image_dir, "RGB", '*.' + 'png')
    file_list.extend(glob.glob(file_glob))

    obj_list = []
    if not file_list:
        print('No files found')
    else:
        for f in file_list:
            filename = os.path.splitext(f.split("/")[-1])[0]
            annotation_file = os.path.join(image_dir, "GTTXT", filename + '.txt')
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
    image_list['training'] = obj_list[:split_pos]
    image_list['validation'] = obj_list[split_pos:]
    no_of_images = len(obj_list)
    print('No. of all files: %d' % no_of_images)
    print('No. of training files: %d' % len(image_list['training']))
    print('No. of validation files: %d' % len(image_list['validation']))

    return image_list
