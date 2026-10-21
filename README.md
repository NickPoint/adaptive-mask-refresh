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

## The client server setup

Here the client (`client.py`) and the segmenter (`segmenter.py`) are separate processes. The client replays the frames at 10 fps, sends every 5th frame as a JPEG and does not wait for the answer. In between it holds the last mask or moves it with optical flow. A mask becomes usable a fixed delay after its request, so it describes an old frame. With flow the client moves it over the frames that passed since, using the flows it already computed for them.

The results and the plots are in the notebook `delay.ipynb`.

### Comparison with the baselines

The mask age is the number of frames between the frame a mask was segmented from and the frame it is shown on. In the baselines, with segmentation on every 5th frame, a mask is shown at ages 0, 1, 2, 3 and 4, so on average it is 2 frames old. In the client-server setup the answer can be used on the next frame at the earliest, so the youngest mask is 1 frame old, and every 100 ms of delay makes the masks one frame older.

IoU on the development sequences, segmentation on every 5th frame:

| | mean mask age | hold last mask | optical-flow propagation |
|---|---|---|---|
| baselines | 2 | 0.62 | 0.84 |
| delay 0 ms | 3 | 0.53 | 0.81 |
| delay 150 ms | 4 | 0.48 | 0.78 |
| delay 350 ms | 6 | 0.41 | 0.73 |
| delay 550 ms | 8 | 0.37 | 0.68 |

The same results by mask age, all delays together:

| mask age (frames) | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| hold, baselines | 0.90 | 0.71 | 0.59 | 0.51 | 0.46 | – | – | – | – |
| hold, with delay | – | 0.72 | 0.59 | 0.51 | 0.46 | 0.43 | 0.41 | 0.37 | 0.35 |
| flow, baselines | 0.90 | 0.87 | 0.84 | 0.81 | 0.78 | – | – | – | – |
| flow, with delay | – | 0.87 | 0.84 | 0.81 | 0.78 | 0.76 | 0.73 | 0.69 | 0.65 |

Some discoveries:

- **The delay makes the masks older, and older masks are worse.** Every 100 ms of delay adds one frame to the mean mask age, from 3 frames at 0 ms to 8 at 550 ms. Over the same range the IoU falls from 0.53 to 0.37 for holding and from 0.81 to 0.68 for flow.
- **Even without delay the masks are one frame older than in the baselines.** The answer can be used on the next frame at the earliest, so the mean age is 3 instead of 2. This is why 0 ms gives 0.81 instead of 0.84 for flow and 0.53 instead of 0.62 for holding.
- **The age is the whole effect of the delay.** At the same age the baselines and the client-server runs differ by at most 0.003 IoU, for ages 1 to 4. So the delay hurts only by making the masks older, and the JPEG compression of the frames costs almost nothing.
