
import glob
import os.path as osp
from pathlib import Path

import numpy as np
import torch
import cv2

from utils.general import (xywh2xyxy, box_iou, ap_per_class)


#读取一个txt文件
def read_files(label_path,label_id):
    label_dir = osp.join(label_path, label_id + '.txt')
    with open(label_dir, 'r') as f:
        s = f.readlines()

    objects = []
    s = s[:]
    for si in s:
        bbox_info = si.split()

        objects.append(bbox_info)

    return np.array(objects).astype(float)


def ap_count(prediction_dir,true_dir,image_path,out_dir,plots=True,verbose=True):

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    names = ['yes','no']
    nc= len(names)

    seen = 0
    p, r, f1, mp, mr, map50, map,ap50 = 0., 0., 0., 0., 0., 0., 0.,0.
    stats, ap, ap_class = [], [], []

    # print(true_dir)
    labels = glob.glob(true_dir + '/*.txt')
    label_ids = np.array([osp.splitext(osp.split(x)[-1])[0] for x in labels])

    # 对每一张图片进行处理
    for si, label_id in enumerate(label_ids):
        jpg = osp.join(image_path, label_id + '.jpg')
        print(jpg)
        img0 = cv2.imread(jpg)
        h, w, _ = img0.shape
        whwh=torch.Tensor([w,h,w,h]).to(device)

        targets = torch.from_numpy(read_files(true_dir,label_id))
        #print(targets)
        pred = torch.from_numpy(read_files(prediction_dir,label_id))

        labels = targets
        nl = len(labels)
        tcls = labels[:, 0].type(torch.IntTensor).tolist() if nl else []  # target class
        seen += 1

        if pred is None:
            if nl:
                stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
            continue

        # Assign all predictions as incorrect
        correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
        if nl:
            detected = []  # target indices
            tcls_tensor = labels[:, 0].type(torch.IntTensor)

            # target boxes
            tbox = xywh2xyxy(labels[:, 1:5])*whwh

            # Per target class
            for cls in torch.unique(tcls_tensor):
                ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                pi = (cls == pred[:, 0].type(torch.IntTensor)).nonzero(as_tuple=False).view(-1)  # target indices

                # Search for detections
                if pi.shape[0]:
                    # Prediction to target ious
                    pbox=xywh2xyxy(pred[pi, 1:5])*whwh
                    ious, i = box_iou(pbox, tbox[ti]).max(1)  # best ious, indices

                    # Append detections
                    detected_set = set()
                    for j in (ious > iouv[0]).nonzero(as_tuple=False):
                        d = ti[i[j]]  # detected target
                        if d.item() not in detected_set:
                            detected_set.add(d.item())
                            detected.append(d)
                            correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                            if len(detected) == nl:  # all targets already located in image
                                break

        # Append statistics (correct, conf, pcls, tcls)
        stats.append((correct.cpu(), pred[:, 5].cpu(), pred[:,0].type(torch.IntTensor).cpu(), tcls))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, fname=out_dir + '/precision-recall_curve.png')
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    results_file = str(out_dir + '/results.txt')
    # Print results
    s='%20s' %('class')
    s=s+'%12s'%('image_num')
    s=s+'%12s'%('object_num')
    s = s + '%12s' % ('p')
    s = s + '%12s' % ('r')
    s = s + '%12s' % ('ap50')
    s = s + '%12s' % ('ap')+'\n'
    pf = '%20s' + '%12.3g' * 6  # print form
    s=s+pf % ('all', seen, nt.sum(), mp, mr, map50, map)
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))


    results = ''
    # Print results per class
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            results=results+'\n'+pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i])
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
    with open(results_file, 'a') as f:
        f.write(s + results + '\n')  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)



    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, maps)


if __name__ == '__main__':
    pred_dir='inference/output/labelTxt'
    true_dir='VOCdevkit/val/labelTxt'
    output_dir='inference/output/results'
    image_path = 'VOCdevkit/val/images'
    ap_count(pred_dir,true_dir,image_path,output_dir)
