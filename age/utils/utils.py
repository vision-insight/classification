import os
import shutil
import random

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


def data_split(image_root_folder, dest_folder = './', train_ratio = 0, val_ratio = 0, test_ratio = 0, shuffle = True):

    assert train_ratio + val_ratio + test_ratio == 1, print('[INFO] sum of all ratio should be 1')

    if train_ratio != 0:
        train_dir = os.path.join(dest_folder, 'train')
        mkdir(train_dir)

    if val_ratio != 0:
        val_dir = os.path.join(dest_folder, 'valid')
        mkdir(val_dir)

    if test_ratio != 0:
        test_dir = os.path.join(dest_folder, 'test')
        mkdir(test_dir)

    for class_name in os.listdir(image_root_folder):
        class_dir = os.path.join(image_root_folder, class_name)
        if not os.path.isdir(class_dir):
            continue
        image_paths = [ os.path.join(class_dir, file_path)
                       for file_path in os.listdir(class_dir)
                       if file_path[-3:] == 'jpg']

        train_images, val_images, test_images = \
                        list_split(image_paths, ratio = [train_ratio, val_ratio, test_ratio])

        if os.path.isdir(train_dir):
            dest_class_dir = os.path.join(train_dir, class_name)
            mkdir(dest_class_dir)
            [shutil.copy(i, dest_class_dir) for i in train_images]

        if os.path.isdir(val_dir):
            dest_class_dir = os.path.join(val_dir, class_name)
            mkdir(dest_class_dir)
            [shutil.copy(i, dest_class_dir) for i in val_images]

        if os.path.isdir(test_dir):
            dest_class_dir = os.path.join(test_dir, class_name)
            mkdir(dest_class_dir)
            [shutil.copy(i, dest_class_dir) for i in test_images]



def mkdir(*path):
    for p in path:
        if not os.path.exists(p):
            os.makedirs(p)
