r"""Convert raw Microsoft COCO dataset to TFRecord for object_detection.

HOW TO RUN:
			
1)To use this script, you need to already have pycocotools installed and available to your project by either having it locally or as part of your PYTHONPATH variable

Example usage for training (windows): 
python create_coco_tf_record_2.py ^
  --train_image_dir=D:\FLIR\FLIR_ADAS_1_3\train\thermal_8_bit ^
  --train_annotations_filepath=D:\FLIR\FLIR_ADAS_1_3\train\thermal_annotations_fixed.json ^
  --set=train ^
  --output_filepath=C:\Users\oh_bo\Documents\EastBay\Covid_App\Exercises\ObjectDetectionTF2Retraining\train\training.tfrecord ^
  --shuffle_imgs=True
  
Example usage for validation (windows):
python create_coco_tf_record_2.py ^
  --val_image_dir=D:\FLIR\FLIR_ADAS_1_3\val\thermal_8_bit ^
  --val_annotations_filepath=D:\FLIR\FLIR_ADAS_1_3\val\thermal_annotations_fixed.json ^
  --set=val ^
  --output_filepath=C:\Users\oh_bo\Documents\EastBay\Covid_App\Exercises\ObjectDetectionTF2Retraining\valid\validation.tfrecord ^
  --shuffle_imgs=True
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pycocotools.coco import COCO
from PIL import Image
from random import shuffle
import os, sys
import numpy as np
import tensorflow as tf
import logging

# import dataset_util
from object_detection.utils import dataset_util

flags = tf.compat.v1.app.flags
# flags.DEFINE_string('data_dir', '', 'Root directory to raw COCO dataset.')
flags.DEFINE_string('train_image_dir', '', 'Directory where training images live')
flags.DEFINE_string('val_image_dir', '', 'Directory where validation images live')
flags.DEFINE_string('train_annotations_filepath', '', 'Path to where the training annotations JSON file lives')
flags.DEFINE_string('val_annotations_filepath', '', 'Path to where the validation annotations JSON file lives')

flags.DEFINE_string('set', 'train', 'Convert training set or validation set')
flags.DEFINE_string('output_filepath', '', 'Path to output TFRecord')
flags.DEFINE_bool('shuffle_imgs',True,'whether to shuffle images of coco')
FLAGS = flags.FLAGS


def load_coco_detection_dataset(imgs_dir, annotations_filepath, shuffle_img = True ):
    """Load data from dataset by pycocotools. This tools can be download from "http://mscoco.org/dataset/#download"
    Args:
        imgs_dir: directories of coco images
        annotations_filepath: file path of coco annotations file
        shuffle_img: whether to shuffle images order
    Return:
        coco_data: list of dictionary format information of each image
    """
    print("imgs_dir: " + imgs_dir)
    print("annotations_filepath: " + annotations_filepath)
	
    coco = COCO(annotations_filepath)
    img_ids = coco.getImgIds() # totally 82783 images
    cat_ids = coco.getCatIds() # totally 90 catagories, however, the number of categories is not continuous, \
                               # [0,12,26,29,30,45,66,68,69,71,83] are missing, this is the problem of coco dataset.

    if shuffle_img:
        shuffle(img_ids)

    coco_data = []

    nb_imgs = len(img_ids)
    for index, img_id in enumerate(img_ids):
        if index % 100 == 0:
            print("Reading images: %d / %d "%(index, nb_imgs))
        img_info = {}
        bboxes = []
        labels = []

        img_detail = coco.loadImgs(img_id)[0]
        pic_height = img_detail['height']
        pic_width = img_detail['width']

        ann_ids = coco.getAnnIds(imgIds=img_id,catIds=cat_ids)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            bboxes_data = ann['bbox']
            bboxes_data = [bboxes_data[0]/float(pic_width), bboxes_data[1]/float(pic_height),\
                                  bboxes_data[2]/float(pic_width), bboxes_data[3]/float(pic_height)]
                         # the format of coco bounding boxs is [Xmin, Ymin, width, height]
            bboxes.append(bboxes_data)
            labels.append(ann['category_id'])
        
        # Emmanuel's comment below
        '''
        original line:
        img_path = os.path.join(imgs_dir, img_detail['file_name'])
        we changed it to:
        img_path = os.path.join(imgs_dir, os.path.basename(img_detail['file_name']))
        because in the ADAS data set, the images in the data set contained metadata where the
        image name was "thermal_8_bit/[file_name.jpg]". And what we were expecting was just
        "[file_name.jpg]". To fix this, we used os.path.basename to grab the file name w/o the path
        HOWEVER, the images in your data set likely won't have that added odd piece of the file path
        connected to them, so you should be able to run the code with the original line above

        if this line is failing, try the adding the following line of code before this one:
        print("image filename (should be just xxxxx.png or xxxxx.jpg)" + img_detail['file_name'])
        if you see anything other than just 'xxxxx.jpg/png' then you may need to do some manipulation
        to make sure youre only grabbing the file name without any path data
        '''
        img_path = os.path.join(imgs_dir, os.path.basename(img_detail['file_name']))
		
        img_bytes = tf.compat.v1.gfile.FastGFile(img_path,'rb').read()

        img_info['pixel_data'] = img_bytes
        img_info['height'] = pic_height
        img_info['width'] = pic_width
        img_info['bboxes'] = bboxes
        img_info['labels'] = labels

        coco_data.append(img_info)
    return coco_data


def dict_to_coco_example(img_data):
    """Convert python dictionary formath data of one image to tf.Example proto.
    Args:
        img_data: infomation of one image, inclue bounding box, labels of bounding box,\
            height, width, encoded pixel data.
    Returns:
        example: The converted tf.Example
    """
    bboxes = img_data['bboxes']
    xmin, xmax, ymin, ymax = [], [], [], []
    for bbox in bboxes:
        xmin.append(bbox[0])
        xmax.append(bbox[0] + bbox[2])
        ymin.append(bbox[1])
        ymax.append(bbox[1] + bbox[3])

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(img_data['height']),
        'image/width': dataset_util.int64_feature(img_data['width']),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/label': dataset_util.int64_list_feature(img_data['labels']),
        'image/encoded': dataset_util.bytes_feature(img_data['pixel_data']),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf-8')),
    }))
    return example

# BE SURE TO MODIFY THE PATHS BELOW
# You are changing imgs_dir and annotations_filepath for both training and validation data

def main(_):
    if FLAGS.set == "train":
        imgs_dir = os.path.join(FLAGS.train_image_dir)
        annotations_filepath = os.path.join(FLAGS.train_annotations_filepath)
        print("Convert coco train file to tf record")
    elif FLAGS.set == "val":
        imgs_dir = os.path.join(FLAGS.val_image_dir)
        annotations_filepath = os.path.join(FLAGS.val_annotations_filepath)
        print("Convert coco val file to tf record")
    else:
        raise ValueError("you must either convert train data or val data")	
    print("output file path: " + FLAGS.output_filepath)
    # load total coco data
    coco_data = load_coco_detection_dataset(imgs_dir,annotations_filepath,shuffle_img=FLAGS.shuffle_imgs)
    total_imgs = len(coco_data)
    # write coco data to tf record
    with tf.io.TFRecordWriter(FLAGS.output_filepath) as tfrecord_writer:
        for index, img_data in enumerate(coco_data):
            if index % 100 == 0:
                print("Converting images: %d / %d" % (index, total_imgs))
            example = dict_to_coco_example(img_data)
            tfrecord_writer.write(example.SerializeToString())


if __name__ == "__main__":
    tf.compat.v1.app.run()