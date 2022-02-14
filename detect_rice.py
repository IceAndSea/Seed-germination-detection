import argparse
import os
import shutil
import time
from pathlib import Path
import numpy
import xlwt
import xlrd
from xlutils.copy import copy
import os.path as osp

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from torchvision.ops import nms

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages,LoadImages_files
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized

def crop_xyxy2ori_xyxy(pred,x_shift,y_shift):
    ori_pred=[]
    pred=pred.cpu().numpy().tolist()
    for det in pred:
        x1,y1,x2,y2,conf,cls=det
        ori_x1,ori_y1=x1+x_shift,y1+y_shift
        ori_x2, ori_y2 = x2 + x_shift, y2 + y_shift
        ori_pred.append([ori_x1,ori_y1,ori_x2,ori_y2,conf,cls])
    return ori_pred

def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz,overlap = \
        opt.save_dir, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size,opt.overlap
    webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')

    # Initialize
    set_logging()
    device = select_device(opt.device)
    if os.path.exists(out):  # output dir
        shutil.rmtree(out)  # delete dir
    os.makedirs(out)  # make new dir
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16


    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages_files(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    colors = numpy.zeros((len(names), 3))
    for indexx in range(len(names)):
        colors[indexx][indexx] = 255

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once



    save_dir = str(Path(out) / 'images')
    svae_txt_dir = str(Path(out) / 'labelTxt')
    if os.path.exists(save_dir):  # output dir
        shutil.rmtree(save_dir)  # delete dir
    os.makedirs(save_dir)  # make new dir
    if save_txt:
        if os.path.exists(svae_txt_dir):  # output dir
            shutil.rmtree(svae_txt_dir)  # delete dir
        os.makedirs(svae_txt_dir)  # make new dir


    for path, img, im0s, vid_cap in dataset:
        H, W, C = im0s.shape
        step_h, step_w = (imgsz - overlap), (imgsz - overlap)
        ori_preds = []
        gn=torch.Tensor([W,H,W,H])
        for x_shift in range(0,max(1,W-overlap),step_w):
            if x_shift > 0:
                x_shift = min(W - imgsz, x_shift)  # w_off + ws <= w if w_off > 0
            for y_shift in range(0,max(1,H-overlap),step_h):
                if y_shift > 0:
                    y_shift = min(H - imgsz, y_shift)  # h_off + hs <= h if h_off > 0
                img_part = img[:, y_shift:y_shift + imgsz, x_shift:x_shift + imgsz]
                img_part = torch.from_numpy(img_part).to(device)
                img_part = img_part.half() if half else img_part.float()  # uint8 to fp16/32
                img_part /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img_part.ndimension() == 3:
                    img_part = img_part.unsqueeze(0)
                # Inference
                t1 = time_synchronized()
                pred = model(img_part, augment=opt.augment)[0]

                # Apply NMS
                pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                           agnostic=opt.agnostic_nms)
                t2 = time_synchronized()
                # xx=pred[0]
                # print('1111111111111111111')
                # print(x_shift)
                # print(y_shift)
                # print(pred)
                # for i, det in enumerate(pred):
                #     ori_pred = crop_xyxy2ori_xyxy(det, x_shift, y_shift)  ####
                #     ori_preds += ori_pred

                if pred[0] is not None and len(pred[0]):
                    ori_pred = crop_xyxy2ori_xyxy(pred[0], x_shift, y_shift)  ####
                    ori_preds += ori_pred

        ori_preds = numpy.array(ori_preds)
        ori_preds = torch.from_numpy(ori_preds).to(device)
        boxes, scores = ori_preds[:, :4].clone(), ori_preds[:, 4]
        index_filter = nms(boxes, scores, opt.iou_thres)
        ori_preds = ori_preds[index_filter]


        # Process detections
        # for i, det in enumerate(ori_preds):  # detections per image
        # if webcam:  # batch_size >= 1
        #     p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
        # else:
        p, s, im0 = path, '', im0s

        save_path = str(Path(save_dir) / Path(p).name)
        txt_path = str(Path(svae_txt_dir) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
        s += '%gx%g ' % img.shape[1:]  # print string
        # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh



        if ori_preds is not None and len(ori_preds):
            # Rescale boxes from img_size to im0 size
            # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            # det= [pd.cpu().numpy().tolist() for pd in ori_preds]
            det=ori_preds

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, names[int(c)])  # add to string



            # Write results
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # normalized xywh
                xywh_save=(xyxy2xywh(torch.tensor(xyxy).view(1, 4))/gn).view(-1).tolist()  # normalized xywh
                if max(xywh[2],xywh[3])>60 and min(xywh[2],xywh[3])>30 and xywh[2]*xywh[3]>1100:


                    if save_txt:  # Write to file

                        line = (cls, *xywh_save,conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line) + '\n') % line)

                    if save_img or view_img:  # Add bbox to image
                        # label = '%s %.2f' % (names[int(cls)], conf)
                        label ='%.2f' % (conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)


        # Print time (inference + NMS)
        print('%sDone. (%.3fs)' % (s, t2 - t1))


        # Stream results
        if view_img:
            cv2.imshow(p, im0)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration

        # Save results (image with detections)
        if save_img:
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
            else:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer

                    fourcc = 'mp4v'  # output video codec
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                vid_writer.write(im0)


    if save_txt or save_img:
        print('Results saved to %s' % Path(out))

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--overlap', type=int, default=160, help='sub image overlap size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
