import os
import pathlib
import shutil
import random
from PIL import Image
from tqdm import tqdm


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
                 self.train_ratio = float(temp[1])
                 print(f"[CONFIG INFO] train ratio : {self.train_ratio}")
                 continue

             elif "valid_ratio" in temp:
                 self.valid_ratio = float(temp[1])
                 print(f"[CONFIG INFO] valid ratio : {self.valid_ratio}")
                 continue

             elif "test_ratio" in temp:
                 self.test_ratio = float(temp[1]) 
                 print(f"[CONFIG INFO] test ratio : {self.test_ratio}")
                 continue

             elif "shuffle" in temp:
                 self.shuffle = True if temp[1] == "True" else False
                 print(f"[CONFIG INFO] shuffle? : {self.shuffle}")
                 continue

    def split(self):
        self.config_reader()
        assert self.train_ratio + self.valid_ratio + self.test_ratio == 1,\
                print('[INFO] sum of all ratio should be 1')

        if self.train_ratio != 0:
            train_dir = os.path.join(self.dest_dir, 'train')
            mkdir(train_dir)
    
        if self.valid_ratio != 0:
            val_dir = os.path.join(self.dest_dir, 'valid')
            mkdir(val_dir)
    
        if self.test_ratio != 0:
            test_dir = os.path.join(self.dest_dir, 'test')
            mkdir(test_dir)
    
        for class_name in tqdm(os.listdir(self.source_dir)):
            class_dir = os.path.join(self.source_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            image_paths = [str(i) for i in pathlib.Path(class_dir).rglob("*.jpg")]
    
            train_images, val_images, test_images = \
                            list_split(image_paths, ratio = [self.train_ratio, self.valid_ratio, self.test_ratio])
    
            if self.train_ratio != 0 and os.path.isdir(train_dir):
                dest_class_dir = os.path.join(train_dir, class_name)
                mkdir(dest_class_dir)
                [shutil.copy(i, dest_class_dir) for i in train_images]
    
            if self.valid_ratio != 0 and os.path.isdir(valid_dir):
                dest_class_dir = os.path.join(val_dir, class_name)
                mkdir(dest_class_dir)
                [shutil.copy(i, dest_class_dir) for i in val_images]
    
            if self.test_ratio != 0 and os.path.isdir(test_dir):
                dest_class_dir = os.path.join(test_dir, class_name)
                mkdir(dest_class_dir)
                [shutil.copy(i, dest_class_dir) for i in test_images]



def mkdir(*path):
    for p in path:
        if not os.path.exists(p):
            os.makedirs(p)

def list_split(input_list, ratio = [0.5, 0.5], shuffle = False):
    assert sum(ratio) == 1, print('sum of ratio must equals to 1')
    if shuffle == True: random.shuffle(input_list)
    c_list = [ round(i*len(input_list)  ) for i in ratio ]
    new_list = []
    for i in range(len(c_list)):
        if i == 0:
            start = 0
        end = start + c_list[i]
        new_list.append(input_list[start : end])
        start = end
    return new_list


if __name__ == "__main__":

    config_file = "/media/D/lulei/classification/tools/data_split_config.cfg"
    
    ds = data_split(config_file = config_file)
    ds.split()

