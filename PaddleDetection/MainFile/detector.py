from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, sys
# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

# ignore warning log
import warnings
warnings.filterwarnings('ignore')
import glob

import paddle
from ppdet.core.workspace import load_config, merge_config
from ppdet.engine import Trainer
from ppdet.utils.check import check_gpu, check_version, check_config
from ppdet.utils.cli import ArgsParser
from ppdet.slim import build_slim_model

from ppdet.utils.logger import setup_logger
logger = setup_logger('train')


def get_test_images(infer_dir, infer_img):
    """
    Get image path list in TEST mode
    """
    assert infer_img is not None or infer_dir is not None, \
        "--infer_img or --infer_dir should be set"
    assert infer_img is None or os.path.isfile(infer_img), \
            "{} is not a file".format(infer_img)
    assert infer_dir is None or os.path.isdir(infer_dir), \
            "{} is not a directory".format(infer_dir)

    # infer_img has a higher priority
    if infer_img and os.path.isfile(infer_img):
        return [infer_img]

    images = set()
    infer_dir = os.path.abspath(infer_dir)
    assert os.path.isdir(infer_dir), \
        "infer_dir {} is not a directory".format(infer_dir)
    exts = ['jpg', 'jpeg', 'png', 'bmp']
    exts += [ext.upper() for ext in exts]
    for ext in exts:
        images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
    images = list(images)

    assert len(images) > 0, "no image found in {}".format(infer_dir)
    logger.info("Found {} inference images in total.".format(len(images)))

    return images


def run(FLAGS, cfg, img_path):
    # build trainer
    trainer = Trainer(cfg, mode='test')

    # load weights
    w = "D:/MultiTargetTracking/PaddleDetection/model/SA_final_model.pdparams"
    trainer.load_weights(w)
    # get inference images
    images = get_test_images(FLAGS.infer_dir, img_path)

    # inference
    bbox_res = trainer.predict(
        images,
        draw_threshold=FLAGS.draw_threshold,
        output_dir=FLAGS.output_dir,
        save_txt=FLAGS.save_txt)
    return bbox_res


def detect(args, img_path):
    FLAGS = args
    cfg = load_config(FLAGS.config)
    cfg['use_vdl'] = FLAGS.use_vdl
    cfg['vdl_log_dir'] = FLAGS.vdl_log_dir
    merge_config(FLAGS.opt)

    place = paddle.set_device('gpu' if cfg.use_gpu else 'cpu')

    if FLAGS.slim_config:
        cfg = build_slim_model(cfg, FLAGS.slim_config, mode='test')

    check_config(cfg)
    check_gpu(cfg.use_gpu)
    check_version()
    bbox_res = run(FLAGS, cfg, img_path)
    return bbox_res
