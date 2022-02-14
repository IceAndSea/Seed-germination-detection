# Seed-germination-detection
Seed germination rate statistics

####train
```
python train.py --data data/voc-seed.yaml --cfg models/yolov5s-seed-transformer-small.yaml --weights weights/yolov5s.pt --batch-size 16 --epochs 100
```

###detect
```
python detect_rice.py --source ./inference/images --img-size 640 --weights weights/best.pt --conf 0.4 --iou-thres 0.4 --agnostic-nms
```

