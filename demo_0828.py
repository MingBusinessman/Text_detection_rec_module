import difflib
import json
import os
import sys
import subprocess


import time
import subprocess
import os
import subprocess
import shutil
import cv2
import copy
import numpy as np
import time
import logging
from PIL import Image
import paddletools.infer.utility as utility
import paddletools.infer.predict_rec as predict_rec
import paddletools.infer.predict_det as predict_det
import paddletools.infer.predict_cls as predict_cls
from ppocr.utils.utility import get_image_file_list, check_and_read_gif
from ppocr.utils.logging import get_logger
from paddletools.infer.utility import draw_ocr_box_txt, get_rotate_crop_image
logger = get_logger()

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

class TextSystem(object):
    def __init__(self, args):
        if not args.show_log:
            logger.setLevel(logging.INFO)

        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)

    def print_draw_crop_rec_res(self, img_crop_list, rec_res):
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite("./output/img_crop_%d.jpg" % bno, img_crop_list[bno])
            logger.info(bno, rec_res[bno])

    def __call__(self, img, cls=True):
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)

        logger.debug("dt_boxes num : {}, elapse : {}".format(
            len(dt_boxes), elapse))
        if dt_boxes is None:
            return None, None
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls and cls:
            img_crop_list, angle_list, elapse = self.text_classifier(
                img_crop_list)
            logger.debug("cls num  : {}, elapse : {}".format(
                len(img_crop_list), elapse))

        rec_res, elapse = self.text_recognizer(img_crop_list)
        logger.debug("rec_res num  : {}, elapse : {}".format(
            len(rec_res), elapse))
        # self.print_draw_crop_rec_res(img_crop_list, rec_res)
        filter_boxes, filter_rec_res = [], []
        for box, rec_reuslt in zip(dt_boxes, rec_res):
            text, score = rec_reuslt
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_reuslt)
        return filter_boxes, filter_rec_res


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes


#目录文件不存在则自动创建,存在则清空并创建
def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.mkdir(path)

#视频提取功能
def video_extract(source_video,to_path,speed):
    to_path = to_path+'%05d.jpg'
    #strcmd = 'ffmpeg -i "%s" -filter:v "select=not(mod(n\,%d)),setpts=N/(25*TB)" -qscale:v 1 "%s"'%(source_video,speed,to_path)
    strcmd = 'ffmpeg -i "%s" -r %d "%s"'%(source_video,speed,to_path)
    # strcmd = "ffmpeg", "-i", filename,"-r","1", dest
    #print(strcmd)
    subprocess.call(strcmd, shell=True)

#视频抽帧处理
def deal_process(src_path,to_base_path,speed):
    if not os.path.exists(src_path):
        return
    if os.path.isdir(src_path):
        for dir in os.listdir(src_path):
            path = os.path.join(src_path,dir)
            deal_process(path,to_base_path,speed)
    else:
        video_name = os.path.splitext(os.path.basename(src_path))[0]
        to_path = os.path.join(to_base_path,video_name)+'/'
        # to_path = os.path.join(to_base_path,video_name+'.jpg')
        check_dir(to_path)
        # to_path = to_path+video_name+'_'
        video_extract(src_path,to_path,speed)

def str_similar(s1, s2):
    seq = difflib.SequenceMatcher(lambda x:x in " ", s1, s2)
    ratio = seq.ratio()
    return ratio

#文本识别主函数
def main(src_path, frame_dir, image_dir, speed, args):

    logger.info("extracting video frames......")
    deal_process(src_path, frame_dir, speed) #抽帧，src_path视频路径，image_dir为帧序列路径

    #一些初始化设置
    image_file_list = get_image_file_list(image_dir)
    image_file_list = image_file_list[args.process_id::args.total_process_num]
    text_sys = TextSystem(args)
    is_visualize = True
    font_path = args.vis_font_path
    drop_score = args.drop_score

    # warm up 10 times
    if args.warmup:
        img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
        for i in range(10):
            res = text_sys(img)

    total_time = 0
    cpu_mem, gpu_mem, gpu_util = 0, 0, 0
    _st = time.time()
    count = 0

    temp_text = ''
    for idx, image_file in enumerate(image_file_list):

        #将视频帧序列的文件名转为json中的时间
        img, flag = check_and_read_gif(image_file)
        seconds = int(os.path.basename(image_file).split('.')[0]) - 1
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        video_time = '{0}:{1:02d}:{2:02d}'.format(h, m, s)

        #文本检测识别
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        starttime = time.time()
        dt_boxes, rec_res = text_sys(img)
        elapse = time.time() - starttime
        total_time += elapse
        text_result = ''
        logger.info(
            str(idx) + "  Predict time of %s: %.3fs" % (image_file, elapse))
        for text, score in rec_res:

            # logger.info("{}, {:.3f}".format(text, score))
            text_result = text_result + text + '\n'

        logger.info("{}".format(text_result))

        #根据字符串相似度，去除相同帧
        ratio =str_similar(temp_text, text_result)
        if ratio < 0.35 :
            temp_text = text_result
            new_text_result = {video_time: text_result}

            # filename = '{}_result.json'.format(src_path.split('.')[0])
            filename = 'inference_results/jsons/{}_result.json'.format(os.path.basename(src_path).split('.')[0])
            logger.info('The filename is {}'.format(filename))
            with open(filename, 'a') as file_obj:
                json.dump(new_text_result, file_obj, indent=4)

        #保存识别结果的可视化图片
        if is_visualize:
            image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            boxes = dt_boxes
            txts = [rec_res[i][0] for i in range(len(rec_res))]
            scores = [rec_res[i][1] for i in range(len(rec_res))]

            draw_img = draw_ocr_box_txt(
                image,
                boxes,
                txts,
                scores,
                drop_score=drop_score,
                font_path=font_path)
            draw_img_save = "./inference_results/"
            if not os.path.exists(draw_img_save):
                os.makedirs(draw_img_save)
            if flag:
                image_file = image_file[:-3] + "png"
            if ratio < 0.35 :

                cv2.imwrite(
                    os.path.join(draw_img_save, os.path.basename(image_file)),
                    draw_img[:, :, ::-1])
                logger.info("The visualized image saved in {}".format(
                    os.path.join(draw_img_save, os.path.basename(image_file))))

    logger.info("The predict total time is {}".format(time.time() - _st))
    logger.info("\nThe predict total time is {}".format(total_time))
    logger.info(text_result)






if __name__ == "__main__":
    args = utility.parse_args()
    this_module_start = time.time()
    if args.use_mp:
        p_list = []
        total_process_num = args.total_process_num
        for process_id in range(total_process_num):
            cmd = [sys.executable, "-u"] + sys.argv + [
                "--process_id={}".format(process_id),
                "--use_mp={}".format(False)
            ]
            p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stdout)
            p_list.append(p)
        for p in p_list:
            p.wait()
    else:
        src_path = 'test_data/video/test1.mp4'  # 原始视频目录
        frame_dir = 'test_data/frames/'  # 抽帧存放目录
        image_dir = 'test_data/frames/{}'.format(os.path.basename(src_path).split('.')[0])
        speed = 1  # 视频抽帧间隔   1帧/秒
        main(src_path, frame_dir, image_dir, speed, args)
        this_module_end = time.time()
        print("This module cost " + str(this_module_end - this_module_start) + "s")