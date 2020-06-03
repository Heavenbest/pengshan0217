# -*- coding: utf-8 -*-
import ConfigParser
import datetime
import json
import os
import threading
import timeit

import cv2
import numpy as np

import classifier
import detector
import log
import pbcvt
import skvideo.io
from encoder import Encoder
import time

class VideoDetector(threading.Thread):
    def __init__(self, camera_info):
        super(VideoDetector, self).__init__()
        self.initialize(camera_info)

    #def __del__(self):
     #   if hasattr(self, 'output'):
    #        self.output.close()
        # self.after_vread()

    def initialize(self, camera_info):
        self.detect_net = detector.load_net(
            'resources/yolov2_sx_oldbest.cfg',
            'resources/yolov2-voc-stone_520_500000.weights', 0)

        self.class_net = classifier.load_net(
            'resources/1th.prototxt',
            'resources/1th.caffemodel',
            'resources/1th.binaryproto')

        self.two_level = camera_info[u'two_level_classify']
        self.one_level_score = camera_info[u'one_level_score']
        self.two_level_score = camera_info[u'two_level_score']
        self.exposure = camera_info[u'exposure']
        self.send_coal = camera_info[u'send_coal']
        if self.two_level:
            self.class_special_net = classifier.load_net(
                'resources/2th.prototxt',
                'resources/2th.caffemodel',
                'resources/2th.binaryproto')

        self.sample_interval = camera_info[u'sample_interval']
        self.output_result = camera_info[u'output_result']
        self.show_result = ('SSH_CLIENT' not in os.environ)
        self.crop_x_min = camera_info[u'crop_x_min']
        self.crop_x_max = camera_info[u'crop_x_max']
        self.crop_y_min = camera_info[u'crop_y_min']
        self.crop_y_max = camera_info[u'crop_y_max']
        self.logger = log.create_logger(
            datetime.date.today().strftime("%Y-%m-%d"))

        if self.output_result:
            self.output = skvideo.io.FFmpegWriter(
                camera_info[u'output_path'].encode('ascii'),
                inputdict={
                    '-r': '1'
                },
                outputdict={
                    '-r': '1'
                })


        self.image_count = {}


    def read(self):
        if not hasattr(self, 'video'):
            self.video = pbcvt.Baumer()
            self.video.initBaumer(self.exposure)

        im = self.video.captureImage()
        while im is None:
            self.logger.info('Retry')
            self.video.stopBaumer()
            self.video.initBaumer()
            im = self.video.captureImage()

        im = im[self.crop_y_min: self.crop_y_max,
                self.crop_x_min: self.crop_x_max]

        return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
  
    def save_photo(self, image,filename,num):
        date = datetime.date.today().strftime('%Y-%m-%d')
        directory = os.path.join('/home/abc/Tracker_0706++/images/data_shuangxin', 'photo', date)
        if not os.path.exists(directory):
            os.makedirs(directory, 0755)

        now = time.time()
        local_time = time.localtime(now)
        date_format_localtime = time.strftime('%H_%M_%S_%ms', local_time)
        format_localtime = date_format_localtime[:(date_format_localtime.find('s'))]
        cv2.imwrite(os.path.join(
            directory, '%s_%s_%d.png' % (format_localtime,filename,num)), image)

        return True
 
    def save_anchor(self, folder,im):
        if folder not in self.image_count:
            self.image_count[folder] = 0

        #if self.image_count[folder] >= 10000:
        #    return False

        self.image_count[folder] += 1

        date = datetime.date.today().strftime('%Y-%m-%d')
        directory = os.path.join('images',folder, date)
        if not os.path.exists(directory):
            os.makedirs(directory, 0755)

        #now = time.time()
        #local_time = time.localtime(now)
        #date_format_localtime = time.strftime('%H_%M_%S_%ms', local_time)
        #format_localtime = date_format_localtime[:(date_format_localtime.find('s'))]
        #cv2.imwrite(os.path.join(
        #    directory, '%s.png' % (format_localtime)), image)
        cv2.imwrite(os.path.join(
            directory, '%d.png' % (self.image_count[folder])), im)

        return True

    def save(self, folder, im, score):     #ci chu xiu gai (self,folder,im,score)
        if folder not in self.image_count:
            self.image_count[folder] = 0

        #if self.image_count[folder] >= 10000:
        #     return False

        self.image_count[folder] += 1

        date = datetime.date.today().strftime('%Y-%m-%d')
        directory = os.path.join('/home/abc/Tracker_0706++/images/data_shuangxin', folder, date)
        if not os.path.exists(directory):
            os.makedirs(directory, 0755)

        cv2.imwrite(os.path.join(
            directory, '%d_%s.png' % (self.image_count[folder], score)), im)

        return True


    def run(self):
            classifier.set_mode_gpu()
       # while True:

            start = timeit.default_timer()
            
            filepath='/home/abc/shuangxin_image/22'

            num = 0
            for root, dirs, files in os.walk(filepath):
                for file in files:
                    num+=1
                    if num%1000==0:
                        print("num=%d"%num)
                    filename=file.split('.')[0]
                    im=cv2.imread(os.path.join(filepath,file))

                    #im = im_[:,50:1780]
                    height, width = im.shape[:2]
                    cv2.namedWindow('read image', 0)
                    cv2.resizeWindow('read image', 640, int(640 * height / width))
                    cv2.imshow('read image', im)
                    #im = self.read()
                    start = timeit.default_timer()

                    detections = detector.detect(self.detect_net, im)

                    self.coals = []
                    im_display = np.copy(im)
                    i=0
                    for x_min, y_min, x_max, y_max in detections:
                        i+=1
                        block_crop = im[y_min:y_max, x_min:x_max]
                        self.save_photo(block_crop,filename,i)
                        is_coal = False
                        score = classifier.classify(self.class_net, block_crop)
                        score_str = '%.2f' % score
                        if score > self.one_level_score:
                            if self.two_level:
                                s = classifier.classify(self.class_special_net, block_crop)
                                score_str += '_%.2f' % s
                            if not self.two_level or s > self.two_level_score:
                                is_coal = True
                                im_display = cv2.rectangle(im_display, (x_min, y_min), (x_max, y_max), (163, 73, 164), 3)
                                im_display = cv2.putText(im_display, 'COAL: %.2f' % score,
                                       (x_min + 5, y_min + 15),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                                if self.send_coal:
                                    self.coals.append((
                                       .5 * (x_min + x_max) / width,
                                       .5 * (y_min + y_max) / height,
                                       1. * (x_max - x_min) / width,
                                       1. * (y_max - y_min) / height))
                        if not is_coal:
                            im_display = cv2.rectangle(im_display, (x_min, y_min), (x_max, y_max),(0, 255, 128), 3)
                            im_display = cv2.putText(
                                    im_display, 'STONE: %.2f' % score,
                                    (x_min + 5, y_min + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                            if not self.send_coal:
                                self.coals.append((
                                   .5 * (x_min + x_max) / width,
                                   .5 * (y_min + y_max) / height,
                                   1. * (x_max - x_min) / width,
                                   1. * (y_max - y_min) / height))

                        #self.save('COAL' if is_coal else 'STONE',  block_crop, score_str)

                        cv2.namedWindow('anchor_box', 0)
                        cv2.resizeWindow('anchor_box', 640, int(640 * height / width))
                        cv2.imshow('anchor_box', im_display)
                        if cv2.waitKey(1) == 40:
                             break
                    self.save_anchor('anchor_box',im_display)
                    endlo = timeit.default_timer()
                    delay = endlo - start
                    print('Delay: %.4fs' % delay)

            """                     
            if height > width:
                width = 1500 * width / height
                height = 1500
            else:
                height = 1500 * height / width
                width = 1500
            im_display = cv2.resize(im_display, (width, height))
            if self.show_result:
                try:
                    cv2.imshow('AI Pickup', im_display)
                except Exception, e:
                    print e
                    self.show_result = False

            if self.output_result and self.frame_idx < 50000:
                self.output.writeFrame(im_display)

            if self.show_result and cv2.waitKey(1) == 27:
                break

            end = timeit.default_timer()
            delay = end - start

            self.logger.info('Delay: %.2fs' % delay)
            """

if __name__ == '__main__':
    config = ConfigParser.ConfigParser()
    config.read('app.config')
    camera_info = json.loads(config.get('cameras', 'camera_info'))
    VideoDetector(camera_info[0]).start()
