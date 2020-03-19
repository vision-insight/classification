# -*- coding: utf-8 -*-
import cv2
import os
import pathlib
import numpy as np
from multiprocessing import Pool
from threading import Thread
import multiprocessing
import random

import datetime
import re
import sys
import shutil


config_file = os.path.join(os.path.dirname(__file__), "config.cfg")

class duplicate_image_removeal:
    def __init__(self, config_file = config_file):
        self.root_dir = None
        self.new_root_dir = None
        self.thre = None
        self.basic_set_num = None
        self.core_ratio = None
        self.input_image = None
        self.RENAME = None
        self.delete_subfolders = None
        self.image_paths = None
        self.core_used = None
        self.init_file_num = 0
        self.round = 0
        self.subgroup_num = 0

    def config_parse(self,):
        """
            Read config from the config file
        """
        with open(config_file) as f:
            content = f.readlines()
        for line in content:
            line = line.strip()
            line = line.split('#')[0]
            line = [i for i in line.split(" ") if i != ""]
            if line == []:
                continue
            elif line[0] == "root_dir":
                self.root_dir = line[1]
                print("[INFO] image root dir : ", self.root_dir)
            elif line[0] == "thre":
                self.thre = float(line[1])
                print("[INFO] thre : ", self.thre)
            elif line[0] == "basic_set_num":
                self.basic_set_num = int(line[1])
                print("[INFO] basic_set_num : ", self.basic_set_num)
            elif line[0] == "core_ratio":
                self.core_ratio = float(line[1])
                print("[INFO] core ratio : ", self.core_ratio)
            elif line[0] == "rename":
                self.RENAME = True if line[1] == "True" else False
                print("[INFO] rename : ", self.RENAME)
            elif line[0] == "delete_subfolders":
                self.delete_subfolders = True if line[1] == "True" else False
                print("[INFO] delete subfolders : ", self.delete_subfolders)
            elif line[0] == "stop_thre":
                self.stop_thre = float(line[1])
                print("[INFO] stop threshold : ", self.stop_thre)
        self.update_image_paths()

    def main(self):
        """
            main function where the application starts
        """
        print("*"*20)
        self.config_parse()
        print("*"*20)
        self.apply()

    def update_image_paths(self,):
        self.image_paths = []
        if not os.path.isdir(self.root_dir):
            raise Exception("root folder %s not exists or is not a dir" % self.root_dir)
        self.image_paths = sorted(self.get_all_files(self.root_dir, file_type='jpg'), reverse = False)

        # self.image_paths = self.get_all_files(self.root_dir, file_type='jpg')
        random.shuffle(self.image_paths)

    def apply(self):
        self.core_used = int(os.cpu_count()*self.core_ratio)
        print("[INFO] core used : ", self.core_used )
        self.init_file_num = len(self.image_paths)
        while True:
            init_num = len(self.image_paths)
            print("[INFO] %d images to be proccessed" % init_num)
            # Split the images into subgroups
            self.subgroup_num = int(round(init_num/self.basic_set_num))
            # If the subgroup num is large than 1
            if self.subgroup_num >= 1:
                print("[INFO] subgroup num is %d, with each group of %d images (roughly)" % (self.subgroup_num, self.basic_set_num))
                self.run(init_num)
            else:
                self.round += 1
                init_num = len(self.image_paths)
                print("[INFO] too few images to be divided into 1 set" )
                print("[INFO] start round (last) : %d " % self.round)
                self._deduplicate(self.image_paths)
                print("[INFO] processing done")
                self.update_image_paths()
                final_num = len(self.image_paths)
                break

            self.update_image_paths()
            final_num = len(self.image_paths)
            if abs(init_num - final_num)/final_num < self.stop_thre:
                break
            print('[INFO] images deleted in round %d : %d ' % (self.round, init_num - final_num))
        print("[INFO] deduplicate finished: %d --> %d, %d deleted ! " % (self.init_file_num, final_num, self.init_file_num - final_num ))
        if self.RENAME:
            self.rename()

    def run(self, init_num):

        self.round += 1
        print("[INFO] image quantity in round %d : %d " %(self.round, init_num))
        img_path_sets = np.array_split(self.image_paths, self.subgroup_num)
        multiprocessing.freeze_support()
        print('[INFO] start multiprocessing, %d core used ' % self.core_used)
        pool = Pool(self.core_used)
        for img_path_set in img_path_sets:
            pool.apply_async(self._deduplicate, args = (img_path_set.tolist(),))
        pool.close()
        pool.join()
        print('[INFO] multiprocessing done')


    def remove_subfolders(self,):
        for temp_dir in pathlib.Path(self.new_root_dir).iterdir() :
            if temp_dir.is_dir():
                self.remove(temp_dir)


    def rename(self,):
        if not self.root_dir.endswith('/'):
            self.root_dir = self.root_dir + "/"
        self.new_root_dir = self.root_dir[:-1] + "_" + self.time_stamp()  + "_dedup_%d" % len(self.image_paths) +'/'
        os.rename(self.root_dir, self.new_root_dir)

        if self.delete_subfolders:
            self.image_paths = self.get_all_files(self.new_root_dir)
            for image_path in self.image_paths:
                try:
                    if os.path.relpath(os.path.dirname(image_path)) == os.path.relpath(self.new_root_dir):
                        continue
                    shutil.move(image_path, self.new_root_dir)
                except Exception as e:
                    print('fadf', str(e))
            self.remove_subfolders()

    def _deduplicate(self, img_paths):
        print('[INFO] file num : ', len(img_paths), ' in process : ', os.getpid())
        while True:
            if img_paths == [] : break
            input_image = imread(img_paths.pop(0), as_gray=True)

            black_list = [] # !!!should not remove the items of a list when the list is in looping.
            for img_path in img_paths:
                score = self.get_hash_score(input_image, img_path)
                if score >= self.thre:
                    black_list.append(img_path)
                    self.remove(img_path)
            for i in black_list : img_paths.remove(i)

            # if img_paths == [] : break

    def get_hash_score(self, img1, img2, hash_type='phash'):
        # img_1 = img1 if isinstance(img1, np.ndarray) else imread(img1, as_gray=True)
        img2 = img2 if isinstance(img2, np.ndarray) else imread(img2, as_gray=True)
        try:
            if hash_type == 'phash':
                h1, h2 = map(cv2.img_hash.pHash, (img1, img2,))
            elif hash_type == 'ahash':
                h1, h2 = map(cv2.img_hash.averageHash, (img1, img2,))
            score = self.hash_score(h1, h2)
        except Exception as e:
            print(e)
            print(img2)
            return 0
        return score

    def get_all_files(self, root_dir, file_type = 'jpg'):
        return [str(i) for i in pathlib.Path(root_dir).rglob('*.' + file_type)]

    def time_stamp(self, time_type = "date_only"):
        if time_type == "date_only":
            return datetime.datetime.now().strftime('%Y%m%d')
        else:
            return datetime.datetime.now().strftime('%Y%m%d_%H%M%S%f')[:-3]

    def remove(self, file_path):
        def remove_file(file_path):
            os.remove(file_path)

        def remove_dir(file_path):
            for i in pathlib.Path(file_path).rglob("*"):
                if not os.path.isfile(i):
                    continue
                remove_file(str(i))
            subfolders = [str(i) for i in pathlib.Path(file_path).rglob("*/")][::-1] + [file_path]
            for subfolder in subfolders:
                os.rmdir(subfolder)
        if os.path.isfile(file_path):
            Thread(target = remove_file, args = (file_path,)).start()
        elif os.path.isdir(file_path):
            Thread(target = remove_dir, args = (file_path,)).start()

    def hash_score(self, hash1, hash2):
        hash_dist = sum([self.ham_dist(v1, v2) for v1, v2 in zip(hash1[0], hash2[0])])
        score = 1 - hash_dist / 64
        return score

    def ham_dist(self, x, y):
        """
        :type x: int
        :type y: int
        :rtype: int
        """
        return bin(x ^ y).count('1')




def imread(file_path, as_gray=False, resize=False, width=False, height=False):
    image = cv2.imread(filename=file_path, flags=cv2.IMREAD_GRAYSCALE if as_gray == True else cv2.IMREAD_COLOR)

    if resize == True:
        image = image_resize(image, width=width, height=height)

    if not as_gray:
        if image.shape[-1] == 4:
            return image[:, :, :-1][:, :, ::-1]
        else:
            return image[:, :, ::-1]

    return image


def image_resize(img, width=False, height=False):
    h, w = img.shape[:2]
    if width != False and height == False:
        new_h = int_round(width / w * h)
        return cv2.resize(img, dsize=(width, new_h))

    if width == False and height != False:
        new_w = int_round(height / h * w)
        return cv2.resize(img, dsize=(new_w, height))

    return cv2.resize(img, dsize=(width, height))


## data format
def int_floor(i):
    if type(i) != list:
        return int(np.floor(i))
    else:
        return [int(np.floor(k)) for k in i]


def int_round(i):
    if type(i) != list:
        return int(np.round(i))
    else:
        return [int(np.round(k)) for k in i]


def int_ceil(i):
    if type(i) != list:
        return int(np.ceil(i))
    else:
        return [int(np.ceil(k)) for k in i]


if __name__ == "__main__":
    di = duplicate_image_removeal()
    di.main()



