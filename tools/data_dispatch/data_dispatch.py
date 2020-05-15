import os
import pathlib
import shutil
import random
from PIL import Image
from tqdm import tqdm
from threading import Thread
import threading
import time
import numpy as np
from multiprocessing import Process, Queue, Manager, Pool
import multiprocessing as mp


random.seed(10)

sem=threading.Semaphore(10024)

class data_dispatch:
    def __init__(self, config_file):
        self.cfg_file = config_file
    
    def config_reader(self):
        with open(self.cfg_file,encoding='UTF-8') as f:
            content = f.readlines() #encoding='UTF-8'
        for line in content:
            temp = [i for i in line.replace('\t','  ').split("#")[0].strip().split(' ') \
                     if    i != '']
            if temp == []:
                continue

            if "source_dir" in temp:
                self.source_dir = temp[1] #self.ip_pattern.findall(temp)[0][0]
                print(f"[CONFIG INFO] source folder : {self.source_dir}")
                continue

            elif "dest_dir" in temp:
                self.dest_dir = temp[1]
                print(f"[CONFIG INFO] dest folder: {self.dest_dir}")
                continue

            elif "file_num" in temp:
                self.file_num = int(temp[1])
                print(f"[CONFIG INFO] file_num : {self.file_num}")
                continue


    def dispatch(self):
        self.config_reader()
        image_paths = [str(i) for i in pathlib.Path(self.source_dir).rglob("*.jpg")]
        print(f"[INFO] file num : {len(image_paths)}")
        sub_group_num = len(image_paths)//self.file_num
        print(f"[INFO] divided into {sub_group_num} subgroups")
        sub_paths = np.array_split(image_paths, sub_group_num)

        pool = Pool(4)
        for index, i in enumerate(sub_paths):
            pool.apply_async(self.file_move, args=(index, i)) 
        pool.close()
        pool.join()

    def file_move(self, index, paths):
        dest_dir = os.path.join(self.dest_dir, "folder_" + str(index))
        mkdir(dest_dir)
        for i in paths:
            file_name = os.path.basename(i)
            dest_path = os.path.join(dest_dir, file_name)
            if os.path.exists(dest_path):
                continue
            else:
                shutil.copy(i, dest_dir)


def mkdir(*path):
    for p in path:
        if not os.path.exists(p):
            os.makedirs(p)


if __name__ == "__main__":

    config_file = r"C:\Users\Administrator\Desktop\config.cfg"
    
    ds = data_dispatch(config_file = config_file)
    ds.dispatch()

