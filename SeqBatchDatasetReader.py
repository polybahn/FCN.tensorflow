from __future__ import print_function

"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""
import numpy as np
import scipy.misc as misc
import progressbar
import sys
from six.moves import xrange
import cv2

class BatchDatset:
    files = []
    images = []
    annotations = []
    image_options = {}
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, records_list, image_options={}):
        """
        Intialize a generic file reader with batching for list of files
        :param records_list: list of file records to read -
        sample record: {'image': f, 'annotation': annotation_file, 'filename': filename}
        :param image_options: A dictionary of options for modifying the output image
        Available options:
        resize = True/ False
            by_ratio = True/ False
                resize_ratio = Float
            else:
            resize_width = #width of output image - does bilinear resize
            resize_height = #height of output image - does bilinear resize
            color=True/False
        """
        print("Initializing Batch Dataset Reader...")
        print(image_options)
        self.files = records_list
        self.image_options = image_options
        self._read_images()

    def _read_images(self):
        self.__channels = True
        bar1 = progressbar.ProgressBar()
        print('resizing RGB images')
        self.images = np.array([self._transform(filename['image'], is_ann=False) for filename in bar1(self.files)])
        print('resizing annotations')
        self.__channels = False
        bar2 = progressbar.ProgressBar()
        self.annotations = np.array(
            [np.expand_dims(self._transform(filename['annotation'], is_ann=True), axis=3) for filename in bar2(self.files)])
        print (self.images.shape)
        print (self.annotations.shape)

        # Debugging print mat
        # for k in xrange(3):
        #     print(str(k) + '\n')
        #     print('\n'.join([' '.join([str(self.annotations[2][i][j][k]) for j in xrange(120)]) for i in xrange(180)]))


    def _transform(self, filename, is_ann):
        image = cv2.imread(filename, -1)
        if self.__channels and len(image.shape) < 3:  # make sure images are of shape(h,w,3)
            image = np.array([image for i in range(3)])

        if self.image_options.get("resize", False) and self.image_options["resize"]:
            if self.image_options.get("by_ratio", False) and self.image_options['by_ratio']:
                resize_ratio = float(self.image_options["resize_ratio"])
                resize_image = cv2.resize(image, (0, 0), fx=resize_ratio, fy=resize_ratio)
            else:
                resize_width = int(self.image_options["resize_size"])
                # resize_height = int(self.image_options["resize_height"])
                resize_image = misc.imresize(image,
                                             [resize_width, resize_width], interp='nearest')
        else:
            resize_image = image

        if is_ann:
            resize_image = resize_image[:,:,2]
            # for i in range(resize_image.shape[0]):
            #     for j in range(resize_image.shape[1]):
            #         if resize_image[i][j] > 15:
            #             print(resize_image[i][j])
            #             print(filename)

        return np.array(resize_image)

    def get_records(self):
        return self.images, self.annotations

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.images.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.annotations = self.annotations[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        return self.images[start:end], self.annotations[start:end]

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return self.images[indexes], self.annotations[indexes]
