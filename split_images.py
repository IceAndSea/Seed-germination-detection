'''
通过xyxy的24找到目标框，然后转化为xyxy，转化为xywh
'''
from multiprocessing import Pool
import cv2
import glob
import os.path as osp
import os
import torch
import numpy as np


# 中心点+宽高坐标格式转换成左上角右下角坐标格式
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[0] = x[0] - x[2] / 2  # top left x
    y[1] = x[1] - x[3] / 2  # top left y
    y[2] = x[0] + x[2] / 2  # bottom right x
    y[3] = x[1] + x[3] / 2  # bottom right y
    return y

# 中心点+宽高坐标格式转换成左上角右下角坐标格式
def xyxy2xywh(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[0] = (x[0] + x[2]) / 2
    y[1] = (x[1] + x[3]) / 2
    y[2] = x[2] - x[0]
    y[3] = x[3] - x[1]
    return y


class DOTAImageSplitTool(object):
    def __init__(self,
                 in_root,
                 out_root,
                 tile_overlap,
                 tile_shape,
                 num_process=8,
                 ):
        self.in_images_dir = osp.join(in_root, 'images/')
        self.in_labels_dir = osp.join(in_root, 'labelTxt/')
        self.out_images_dir = osp.join(out_root, 'images/')
        self.out_labels_dir = osp.join(out_root, 'labelTxt/')
        assert isinstance(tile_shape, tuple), f'argument "tile_shape" must be tuple but got {type(tile_shape)} instead!'
        assert isinstance(tile_overlap,
                          tuple), f'argument "tile_overlap" must be tuple but got {type(tile_overlap)} instead!'
        self.tile_overlap = tile_overlap
        self.tile_shape = tile_shape
        images = glob.glob(self.in_images_dir + '*.jpg')   #将图片的后缀名
        labels = glob.glob(self.in_labels_dir + '*.txt')
        image_ids = [*map(lambda x: osp.splitext(osp.split(x)[-1])[0], images)]
        label_ids = [*map(lambda x: osp.splitext(osp.split(x)[-1])[0], labels)]

        assert set(image_ids) == set(label_ids)
        self.image_ids = image_ids
        if not osp.isdir(out_root):
            os.mkdir(out_root)
        if not osp.isdir(self.out_images_dir):
            os.mkdir(self.out_images_dir)
        if not osp.isdir(self.out_labels_dir):
            os.mkdir(self.out_labels_dir)
        self.num_process = num_process

    def _parse_annotation_single(self, image_id):
        label_dir = osp.join(self.in_labels_dir, image_id + '.txt')
        with open(label_dir, 'r') as f:
            s = f.readlines()

        objects = []
        s = s[:]
        for si in s:
            bbox_info = si.split()
            assert len(bbox_info) == 5


            bbox_size = [float(bbox_info[3]), float(bbox_info[4])]

            center = float(bbox_info[1]), float(bbox_info[2])

            xywh = [float(bbox_info[1]), float(bbox_info[2]),float(bbox_info[3]), float(bbox_info[4])]
            xyxy=xywh2xyxy(xywh)

            objects.append({'bbox_size': bbox_size,
                            'label': bbox_info[0],
                            'center': center,
                            'xyxy':xyxy})
        return objects

    def _split_single(self, image_id):
        objs = self._parse_annotation_single(image_id)
        image_dir = osp.join(self.in_images_dir, image_id + '.jpg')################################
        img = cv2.imread(image_dir)
        h, w, _ = img.shape
        w_ovr, h_ovr = self.tile_overlap
        w_s, h_s = self.tile_shape

        for h_off in range(0, max(1, h - h_ovr), h_s - h_ovr):
            if h_off > 0:
                h_off = min(h - h_s, h_off)  # h_off + hs <= h if h_off > 0
            for w_off in range(0, max(1, w - w_ovr), w_s - w_ovr):
                if w_off > 0:
                    w_off = min(w - w_s, w_off)  # w_off + ws <= w if w_off > 0
                objs_tile = []

                for obj in objs:
                    # if w_off <= obj['center'][0]*w <= w_off + w_s - 1:
                    #     if h_off <= obj['center'][1]*h <= h_off + h_s - 1:
                    #         objs_tile.append(obj)
                    if w_off-24 <= obj['xyxy'][0]*w <= w_off + w_s +24 and w_off-24 <= obj['xyxy'][2]*w <= w_off + w_s +24:
                        if h_off-24 <= obj['xyxy'][1]*h <= h_off + h_s +24 and h_off-24 <= obj['xyxy'][3]*h <= h_off + h_s +24:
                            objs_tile.append(obj)
                #print(len(objs_tile))
                if len(objs_tile) > 0:
                    img_tile = img[h_off:h_off + h_s, w_off:w_off + w_s, :]
                    save_image_dir = osp.join(self.out_images_dir, f'{image_id}_{w_off}_{h_off}.jpg')
                    save_label_dir = osp.join(self.out_labels_dir, f'{image_id}_{w_off}_{h_off}.txt')
                    cv2.imwrite(save_image_dir, img_tile)
                    label_tile = []
                    for obj in objs_tile:
                        px, py = obj['center'][0]*w, obj['center'][1]*h

                        px = str((px - w_off)/w_s)
                        py = str((py - h_off)/h_s)

                        xyxy_up = [max(1,obj['xyxy'][0]*w- w_off),max(1,obj['xyxy'][1]*h- h_off),min(w_s-1,obj['xyxy'][2]*w- w_off),min(h_s-1,obj['xyxy'][3]*h- h_off)]
                        xywh_up = xyxy2xywh(xyxy_up)


                        obj_s = f'{obj["label"]} {str(xywh_up[0]/w_s)} {str(xywh_up[1]/h_s)} {str(xywh_up[2]/w_s)} {str(xywh_up[3]/h_s)}\n'
                        label_tile.append(obj_s)
                    with open(save_label_dir, 'w') as f:
                        f.writelines(label_tile)

    def split(self):
        with Pool(self.num_process) as p:
            p.map(self._split_single, self.image_ids)


if __name__ == '__main__':
    wd = os.getcwd()
    # aim_dir1 = 'move_dir\\images\\'
    # train_big = 'VOCdevkit/train/images/'
    # source_path1 = os.path.join(wd, "trainsplit\\images\\")
    train_big_path = os.path.join(wd, "VOCdevkit/train")
    train_small_path=os.path.join(wd, "VOCdevkit/trainsplit")

    trainsplit = DOTAImageSplitTool(train_big_path,
                                    train_small_path,
                                    tile_overlap=(160, 160),
                                    tile_shape=(640, 640))
    trainsplit.split()

    val_big_path = os.path.join(wd, "VOCdevkit/val")
    val_small_path = os.path.join(wd, "VOCdevkit/valsplit")
    valsplit = DOTAImageSplitTool(val_big_path,
                                  val_small_path,
                                  tile_overlap=(160, 160),
                                  tile_shape=(640, 640))
    valsplit.split()