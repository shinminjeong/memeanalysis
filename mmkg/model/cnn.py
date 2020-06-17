# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import tensorflow as tf
import imghdr

from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

from PIL import Image
from mmkg.model.inception import maybe_download, Inception

def run_model(model, image):
    predictions = np.zeros((1, 2048))
    labels = []

    # Runs inference on an image.
    if not tf.io.gfile.exists(image):
        tf.logging.fatal('File does not exist %s', image)

    if imghdr.what(image) == 'png':
        image_data = Image.open(image)
        image_array = np.array(image_data)

        # convert png/colormap to png/colorRGB
        if len(image_array.shape) == 2:
            image_data = image_data.convert('RGB')
            image_array = np.array(image_data)

        if len(image_array.shape) == 3:
            image_array = image_array[:, :, 0:3]
            predictions, softmax0 = model.transfer_softmax(image=image_array) # get pool_3 layer
            labels = model.get_labels(pred=softmax0, k=5, only_first_name=False)
        else:
            tf.logging.fatal("false PNG format")

    elif imghdr.what(image) == 'jpeg':
        predictions, softmax0 = model.transfer_softmax(image_path=image) # get pool_3 layer
        labels = model.get_labels(pred=softmax0, k=5, only_first_name=False)
    else:
        try: # at least try as a jpeg... without format info
            predictions, softmax0 = model.transfer_softmax(image_path=image) # get pool_3 layer
            labels = model.get_labels(pred=softmax0, k=5, only_first_name=False)
        except Exception as e:
            print(e)
            tf.logging.fatal("not supported image format " + str (imghdr.what(image)))

    return predictions, labels


def extract_features(imagefiles):
    maybe_download()
    model = Inception()

    n_cpu = cpu_count()*2
    p = ThreadPool(n_cpu)
    print("Extract image features: use %d threads" % n_cpu)

    res = []
    threads = [p.apply_async(run_model, args=(model, image)) for image in imagefiles]
    for thread in threads:
        res.append(thread.get())

    model.close()
    return res
