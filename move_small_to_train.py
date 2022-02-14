import os
from PIL import Image
import shutil
import sys

from shutil import copyfile


def move_files(source_path, aim_dir):
    if os.path.exists(aim_dir):
        shutil.rmtree(aim_dir)
    os.makedirs(aim_dir)
    source_list = os.listdir(source_path)
    files = [os.path.join(source_path, _) for _ in source_list]
    for index, source_file in enumerate(files):
        (nameWithoutExtention, extention) = os.path.splitext(os.path.basename(source_file))
        label_name = nameWithoutExtention + extention
        copyfile(source_file, aim_dir + label_name)


if __name__ == '__main__':
    # 创建固定格式的数据文件目录
    wd = os.getcwd()
    # aim_dir1 = 'move_dir\\images\\'
    aim_dir1 = 'VOCdevkit/images/train/'
    # source_path1 = os.path.join(wd, "trainsplit\\images\\")
    source_path1 = os.path.join(wd, "VOCdevkit/trainsplit/images/")
    move_files(source_path1, aim_dir1)
    #     split_dir1 = 'VOCdevkit/trainsplit/images/'
    #     if os.path.exists(split_dir1):
    #         shutil.rmtree(split_dir1)
    #     os.makedirs(split_dir1)

    # aim_dir2 = 'move_dir\\labels\\'
    aim_dir2 = 'VOCdevkit/labels/train/'
    # source_path2 = os.path.join(wd, "trainsplit\\labelTxt\\")
    source_path2 = os.path.join(wd, "VOCdevkit/trainsplit/labelTxt/")
    move_files(source_path2, aim_dir2)
    #     split_dir2 = 'VOCdevkit/trainsplit/labelTxt/'
    #     if os.path.exists(split_dir2):
    #         shutil.rmtree(split_dir2)
    #     os.makedirs(split_dir2)

    aim_dir3 = 'VOCdevkit/images/val/'
    source_path3 = os.path.join(wd, "VOCdevkit/valsplit/images/")
    move_files(source_path3, aim_dir3)
    #     split_dir3 = 'VOCdevkit/valsplit/images/'
    #     if os.path.exists(split_dir3):
    #         shutil.rmtree(split_dir3)
    #     os.makedirs(split_dir3)

    aim_dir4 = 'VOCdevkit/labels/val/'
    source_path4 = os.path.join(wd, "VOCdevkit/valsplit/labelTxt/")
    move_files(source_path4, aim_dir4)
#     split_dir4 = 'VOCdevkit/valsplit/labelTxt/'
#     if os.path.exists(split_dir4):
#         shutil.rmtree(split_dir4)
#     os.makedirs(split_dir4)