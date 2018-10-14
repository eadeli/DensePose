# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import sys
import time
from multiprocessing import Pool

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils
import pickle
from detectron.utils.vis import kp_connections, keypoint_utils, mask_util

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def run_one_image(params):
        model = params["model"]
        im_name = params["im_name"]
        in_pos = params['in_pos']
        out_name = params["out_name"]
        if os.path.exists(out_name):
            print('Existed {}'.format(im_name))
            return
        #down = params['down']
        #if out_name in down:
        #    return
        print('Processing {}'.format(im_name))
        im = cv2.imread(im_name)
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps, cls_bodys = infer_engine.im_detect_all(
                model, im, None, boxes_in=in_pos
            )
        f = open(out_name, "wb")
        dataset_keypoints, _ = keypoint_utils.get_keypoints()
        kp_lines = kp_connections(dataset_keypoints)
        if cls_boxes is not None:
            boxes = cls_boxes[1]
            masks = mask_util.decode(cls_segms[1])
            keypoints = cls_keyps[1]
            bodys = cls_bodys[1]
            pickle.dump({'boxes':boxes, 'masks':masks, 'keyps':keypoints, 'bodys':bodys, 'kp_lines':kp_lines}, f)
        else:
            pickle.dump({'boxes':None, 'masks':None, 'keyps':None, 'bodys':None, 'kp_lines':kp_lines}, f)
        #down[out_name] = 1
        print('Processed {}'.format(im_name))

def main(args):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()
    
    params_list = []
    down_ = {}
    for folder in os.listdir(args.im_or_folder):
        os.system("mkdir -p "+"/workspace/caozhangjie/DensePose/JAAD_result/"+folder)
        im_list = glob.iglob(args.im_or_folder + '/'+folder+'/*.' + args.image_ext)
        for im_name in im_list:
            out_name = "/workspace/caozhangjie/DensePose/JAAD_result/"+folder+'/'+im_name.split("/")[-1].split(".")[0]+".pkl"
            img_name = '/data/JAAD_clip_images/'+folder+'.mp4/'+str(int(im_name.split("/")[-1].split(".")[0]))+'.jpg'
            params_list.append({"im_name":img_name, "model":model, "out_name":out_name, 'in_pos':im_name})
    #pickle.dump(down_, open('down_file.pkl', 'wb'))
    #pickle.dump(params_list, open("JAAD_param_list.pkl", "wb"))
    
    params_list = pickle.load(open("JAAD_param_list.pkl", "r"))
    down_ = pickle.load(open('down_file.pkl', 'rb'))
    #print(params_list[0]
    for params in params_list[0:20000]:
        params['down'] = down_
        run_one_image(params)
    pickle.dump(down_, open('down_file.pkl', 'wb'))

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
