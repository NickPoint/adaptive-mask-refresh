# Adaptive mask refresh for asynchronous edge–server video segmentation

A client sees video frames but segmentation runs less often, on a server, and its results arrive late. The question is whether the client can keep useful object masks in between, and when it should request a fresh one.

## Research question

How much mask accuracy is lost when segmentation runs only every N frames, how much of that loss does optical-flow propagation recover, and can a rule that uses only client-side information decide when to refresh better than a fixed interval?

## Setup

- **Data:** KITTI MOTS [dataset](https://www.vision.rwth-aachen.de/page/mots), zip [download](http://www.cvlibs.net/download.php?file=data_tracking_image_2.zip), [annotations](https://www.vision.rwth-aachen.de/media/resource_files/instances_txt.zip) in rle format
- **Segmentation:** pretrained Ultralytics YOLO segmentation model.
- **Propagation:** OpenCV dense optical flow.

## How to run

```
pip install -r requirements.txt
```

Then open `baselines.ipynb`. To run the baselines set `RUN_BASELINES = True` in the first cell. The per-frame results are written to `results/baselines.pkl`.

## Choices made

- **Split:** the standard KITTI MOTS train/val split. The 12 train sequences are used for development, the 9 val sequences are held out.
- **Classes:** COCO `person` and `car` only. The trucks, busses and other means of transport are not annotated in KITTI MOTS data either.
- **Optical flow:** Here I use OpenCV DIS.
- **IoU:** intersection and union are summed over all frames before dividing, with ignore regions excluded. Frames where both masks are empty therefore add nothing.
- **Hardware:** segmentation runs on the GPU and optical flow on the CPU.

## Baseline results

The baselines, their results and the plots are in the notebook `baselines.ipynb`. In short, holding the last mask loses accuracy very fast (IoU 0.895 with segmentation on every frame, 0.62 when I segment every 5th frame), and optical-flow propagation wins most of it back (0.84 at the same interval).
