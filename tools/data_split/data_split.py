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

random.seed(10)

sem=threading.Semaphore(10024)

class data_split:
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

            elif "train_ratio" in temp:
                train_ratio = float(temp[1])
                print(f"[CONFIG INFO] train ratio : {train_ratio}")
                continue

            elif "valid_ratio" in temp:
                valid_ratio = float(temp[1])
                print(f"[CONFIG INFO] valid ratio : {valid_ratio}")
                continue

            elif "test_ratio" in temp:
                test_ratio = float(temp[1]) 
                print(f"[CONFIG INFO] test ratio : {test_ratio}")
                continue

            elif "shuffle" in temp:
                self.shuffle = True if temp[1] == "True" else False
                print(f"[CONFIG INFO] shuffle? : {self.shuffle}")
                continue

        self.ratio = {"train": train_ratio, "valid": valid_ratio, "test": test_ratio}
        assert train_ratio + valid_ratio + test_ratio == 1,\
                print('[INFO] sum of all ratio should be 1')



    #def split_parallel(self, label = "train"):
        
        


    def split(self):
        self.config_reader()
        
        dest_dirs = {}
        for temp in ["train", "valid", "test"]:
            if self.ratio[temp] != 0:
                dest_dirs[temp] = os.path.join(self.dest_dir, temp)
                mkdir(dest_dirs[temp])
        
        for class_name in tqdm(os.listdir(self.source_dir)):
            class_dir = os.path.join(self.source_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            images_per_class = [str(i) for i in pathlib.Path(class_dir).rglob("*.jpg")]
    
            image_sets = list_split(images_per_class, ratio = [self.ratio["train"], self.ratio["valid"], self.ratio["test"]])
    
            for temp, image_set in zip(["train", "valid", "test"], image_sets):
                if self.ratio[temp] != 0 and os.path.isdir(dest_dirs[temp]):
                    dest_class_dir = os.path.join(dest_dirs[temp], class_name)
                    mkdir(dest_class_dir)
                    for i in image_set:
                            #print(threading.current_thread().name,i)
                            #time.sleep(1)
                        #if len(threading.enumerate()) >= 1000:
                        #        time.sleep(15)
                        #Thread(target = shutil.copy, args=(i, dest_class_dir), daemon = True).start()
                        shutil.copy(i, dest_class_dir)


def mkdir(*path):
    for p in path:
        if not os.path.exists(p):
            os.makedirs(p)

def list_split(input_list, ratio = [0.5, 0.5], shuffle = False):
    assert sum(ratio) == 1, print("sum of ratio should be 1")
    input_list = sorted(input_list)
    len_list = len(input_list)

    split_position = []
    temp = 0
    for i in ratio:
        split_position.append(round(len_list*(temp+i)))
        temp += i

    return_list = np.split(input_list, split_position)
    output_list = []
    temp_sum = 0
    for i in return_list:
        if len(i) == 0:
            continue
        temp_sum += len(i)
        output_list.append(i.tolist())

    if temp_sum != len_list:
        raise Exception("output list is len than input_list")
    return output_list


if __name__ == "__main__":

    config_file = "./config.cfg"
    
    ds = data_split(config_file = config_file)
    ds.split()

