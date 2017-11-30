
import argparse
import os
import sys
import time

from scipy import misc
import tensorflow as tf
import numpy as np

import align.detect_face
import facenet

def main(args):
    labels_set = set()
    image_files = []
    for images_path in args.images_paths:
        images_path = os.path.expanduser(images_path)
        for file_name in os.listdir(images_path):
            image_files.append(os.path.join(images_path, file_name))
            label = file_name.split('_')[0]
            labels_set.add(label)

    start_time = time.time()
    # images are pre-processed and paths are correspond with images
    images, paths = load_and_align_data(image_files, args.image_size, args.margin)
    print("load and align data has spend %5.3f s" % (time.time() - start_time))
    assert images.shape[0] == len(paths)

    with tf.Graph().as_default():

        with tf.Session() as session:

            facenet.load_model("")

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # emb = session.run(embeddings, feed_dict={images_placeholder: images, phase_train_placeholder: False})

def load_and_align_data(image_paths, image_size, margin):
    # the mini size of face
    mini_size = 20
    # three steps's threshold
    threshold = [0.6, 0.7, 0.7]
    # scale factor
    factor = 0.709

    print("Creating networks and loading parameters")
    with tf.Graph().as_default():

        session = tf.Session()
        with session.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(session, None)

    nrof_samples = len(image_paths)

    img_list = []
    img_path = []
    for i in range(nrof_samples):
        img = misc.imread(os.path.expanduser(image_paths[i]), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        # each bounding_box consists of [x1, y1, x2, y2]
        # (x1, y1) is the first point of face, (x2, y2) is the third point of face
        bounding_boxes, _ = align.detect_face.detect_face(img, mini_size, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) == 0:
            continue
        # just chose the first box as the main face
        # TODO: It maybe can detect face and compute each face embedding, in order to choice the best embedding as main face
        det = np.squeeze(bounding_boxes[0, 0: 4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)  # x1
        bb[1] = np.maximum(det[1] - margin / 2, 0)  # y1
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])  # x2
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])  # y2
        cropped = img[bb[1]: bb[3], bb[0]: bb[2], :]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
        img_path.append(image_paths[i])
    images = np.stack(img_list)
    paths = np.asarray(img_path)
    return images, paths


def parse_argments(argv):
    parser = argparse.ArgumentParser()
    # parser.add_argument('model', type=str, help="Could be either a directory containing the meta_file and ckpt_file or "
    #                                             "model protobuf (.pb) file")
    parser.add_argument('images_paths', type=str, nargs='+', help="Images path to classify")
    parser.add_argument('--margin', type=int, help="Margin for the crop around the bounding box", default=40)
    parser.add_argument('--image_size', type=int, help="Image size (height, width) in pixels", default=160)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_argments(sys.argv[1:]))