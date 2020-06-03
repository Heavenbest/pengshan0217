# -*- coding: UTF-8 -*-
import os
import datetime
import cv2
import time
import numpy as np
import operator
#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')


from xml.dom.minidom import parse

def save_photo(image,category,image_count):
    date = datetime.date.today().strftime('%Y-%m-%d')
    directory = os.path.join( 'photo', date, category)
    if not os.path.exists(directory):
        os.makedirs(directory, 0755)

    now = time.time()
    local_time = time.localtime(now)
    date_format_localtime = time.strftime('%H_%M_%S_%ms', local_time)
    format_localtime = date_format_localtime[:(date_format_localtime.find('s'))]
    cv2.imwrite(os.path.join(
        directory, '%s_%d.png' % (format_localtime,image_count)), image)

    return True

def save(folder, im, image_count={}):
    if folder not in image_count:
        image_count[folder] = 0

    #if image_count[folder] >= 20000:
     #   image_count[folder] = 0

    image_count[folder] += 1

    date = datetime.date.today().strftime('%Y-%m-%d')
    directory = os.path.join('images', folder, date)
    if not os.path.exists(directory):
        os.makedirs(directory, 0o755)

    now = time.time()
    local_time = time.localtime(now)
    date_format_localtime = time.strftime('%H_%M_%S_%ms', local_time)
    format_localtime = date_format_localtime[:(date_format_localtime.find('s'))]
    cv2.imwrite(os.path.join(
        directory, '%s_%d.png' % (format_localtime,image_count[folder])), im)

    return True


def cutout_photo(filepath):
    for root, dirs, files in os.walk(filepath):
        image_count = 0
        for file in files:
            if file.endswith('.xml'):
                filename=file.split('.')[0]
                xml_file=os.path.join(os.path.abspath(filepath),filename+'.xml')
                jpg_file = os.path.join(os.path.abspath(filepath), filename + '.jpg')
                img = cv2.imread(jpg_file)
                height, width = img.shape[:2]
                cv2.namedWindow('read image', 0)
                cv2.resizeWindow('read image', 430, int(430 * height / width))
                cv2.imshow('read image', img)
                im_display = np.copy(img)

                doc = parse(xml_file)
                root = doc.documentElement
                objects = root.getElementsByTagName('object')
                for object in objects:
                    name_dom = object.getElementsByTagName("name")[0]
                    str_name = name_dom.childNodes[0].data

                    xmin_dom = object.getElementsByTagName("xmin")[0]
                    str_xmin = xmin_dom.childNodes[0].data
                    xmin = int(str_xmin)

                    ymin_dom = object.getElementsByTagName("ymin")[0]
                    str_ymin = ymin_dom.childNodes[0].data
                    ymin = int(str_ymin)

                    xmax_dom = object.getElementsByTagName("xmax")[0]
                    str_xmax = xmax_dom.childNodes[0].data
                    xmax = int(str_xmax)

                    ymax_dom = object.getElementsByTagName("ymax")[0]
                    str_ymax = ymax_dom.childNodes[0].data
                    ymax = int(str_ymax)
                    # print(xmin, ymin, xmax, ymax)
                    block_crop = img[ymin:ymax, xmin:xmax]
                    #if image_count >= 20000:
                    #    image_count = 0
                    image_count += 1
                    save_photo(block_crop, str_name,image_count)
                    if str_name == 'stone':
                        im_display = cv2.rectangle(im_display, (xmin, ymin), (xmax, ymax), (0, 255, 128), 3)
                    else:
                        im_display = cv2.rectangle(im_display, (xmin, ymin), (xmax, ymax), (163, 73, 164), 3)
                save('anchor_box', im_display)
                cv2.namedWindow('anchor_box', 0)
                cv2.resizeWindow('anchor_box', 430, int(430 * height / width))
                cv2.imshow('anchor_box', im_display)

                if cv2.waitKey(1) == 27:
                    break

def find_file(fileDir,filename):
    out=0
    for root,dirs,files in os.walk(fileDir):
        for file in files:
            if operator.eq(file,filename):
                out=1
                break

    return out


def search_xml(fileDir):
    i=0
    for root,dirs,files in os.walk(fileDir):
        for file in files:
            if file.endswith('.xml'):
                i += 1
                if i % 1000 == 0:
                    print("search xml file number: %d" % i)
                filename=file.split('.')[0]
                jpgfile=filename+'.jpg'
                out=find_file(fileDir,jpgfile)
                if out==0:
                    out_name = filename + '.xml'
                    print("xml file not find:", out_name)
                    os.remove(fileDir + file)


def search_jpg(fileDir):
    i=0
    for root,dirs,files in os.walk(fileDir):
        for file in files:
            if file.endswith('.jpg'):
                i += 1
                if i % 1000 == 0:
                    print("search jpg file number: %d" % i)
                filename=file.split('.')[0]
                xmlfile=filename+'.xml'
                out=find_file(fileDir,xmlfile)
                if out==0:
                    out_name=filename+'.jpg'
                    print("jpg file not find:",out_name)
                    os.remove(fileDir+file)


if __name__ == '__main__':
    #fileDir='C:\\pycharm_work\\xml\\stone_coal\\'
    #search_jpg(fileDir)
    #search_xml(fileDir)

    filepath='/media/abc/0012-D687/双欣图片/shuangxin/jpgs'
    cutout_photo(filepath)
