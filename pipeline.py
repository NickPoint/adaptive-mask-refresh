import os
import time

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask

DATA_DIR = "kitti_mots"
KEYS = ["seq", "time_frame"]

# Standard KITTI MOTS split (train.seqmap / val.seqmap in mots_tools)
DEV_SEQS = ["0000", "0001", "0003", "0004", "0005", "0009",
            "0011", "0012", "0015", "0017", "0019", "0020"]
HELDOUT_SEQS = ["0002", "0006", "0007", "0008", "0010",
                "0013", "0014", "0016", "0018"]

# COCO classes counted as a KITTI MOTS object: person, car
CLASSES = [0, 2]

FPS = 10 # KITTI frame rate

SEGMENTER_ADDRESS = ("localhost", 6000)


def merge_per_frame(rows, name):
    return pd.DataFrame(
        [(seq, frame, mask.merge(g.tolist())) for (seq, frame), g in rows.groupby(KEYS)["rle_obj"]],
        columns=KEYS + [name],
    )


def load_frames(seqs):
    """One row per image, with the combined ground-truth and ignore masks as RLE (NaN when a frame has none)"""
    frames, annots = [], []
    for seq in seqs:
        image_dir = f"{DATA_DIR}/training/image_02/{seq}"
        names = sorted(os.listdir(image_dir))
        frames.append(pd.DataFrame({
            "seq": seq,
            "time_frame": [int(os.path.splitext(n)[0]) for n in names],
            "image_path": [f"{image_dir}/{n}" for n in names],
        }))

        a = pd.read_csv(f"{DATA_DIR}/instances_txt/{seq}.txt", delimiter=" ",
                        names=["time_frame", "id", "class_id", "img_height", "img_width", "rle"])
        a.insert(0, "seq", seq)
        annots.append(a)

    frames = pd.concat(frames, ignore_index=True)
    annots = pd.concat(annots, ignore_index=True)

    annots["rle_obj"] = annots.apply(
        lambda x: {"size": [int(x["img_height"]), int(x["img_width"])], "counts": x["rle"].encode()},
        axis=1,
    )

    is_object = annots["class_id"].isin([1, 2])
    gt = merge_per_frame(annots[is_object], "gt_object")
    ignore = merge_per_frame(annots[~is_object], "ignore_object")

    return frames.merge(gt, on=KEYS, how="left").merge(ignore, on=KEYS, how="left")


def iou(pred, gt, ignore):
    valid = ~ignore
    intersection = (pred & gt & valid).sum()
    union = ((pred | gt) & valid).sum()
    return intersection, union


def to_mask(rle, shape):
    if isinstance(rle, dict):
        return mask.decode(rle).astype(bool)
    return np.zeros(shape, dtype=bool)


def to_rle(m):
    return mask.encode(np.asfortranarray(m.astype(np.uint8)))


def segment(model, img):
    result = model.predict(img, classes=CLASSES, retina_masks=True, verbose=False)[0]
    if result.masks is None:
        return np.zeros(img.shape[:2], dtype=bool)
    return result.masks.data.any(dim=0).cpu().numpy().astype(bool)


_dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)


def backward_flow(prev_img, img):
    """Dense flow from img back to prev_img: for every pixel of the new frame, where that pixel was before"""
    return _dis.calc(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY), None)


def warp(m, flow):
    ys, xs = np.indices(m.shape, dtype=np.float32)
    warped = cv2.remap(m.astype(np.uint8), xs + flow[..., 0], ys + flow[..., 1], cv2.INTER_NEAREST)
    return warped.astype(bool)


def flow_warp(m, prev_img, img):
    """Move the mask of prev_img onto img with dense optical flow."""
    return warp(m, backward_flow(prev_img, img))


def run_sequence(seq_frames, model, interval=1, propagate=None, fps=None):
    """Segment every `interval` frames. In between propogate mask with optical flow or hold it.

    With `fps` set, frames are released at that rate in real time, as a camera would deliver them.
    """
    rows = []
    pred = prev_img = None
    start = time.perf_counter()
    for i, row in enumerate(seq_frames.sort_values("time_frame").itertuples()):
        if fps is not None:
            time.sleep(max(0.0, start + i / fps - time.perf_counter()))
        img = cv2.imread(row.image_path)
        shape = img.shape[:2]
        mask_age = i % interval

        t0 = time.perf_counter()
        if mask_age == 0:
            pred = segment(model, img)
        elif propagate is not None:
            pred = propagate(pred, prev_img, img)
        step_time = time.perf_counter() - t0
        prev_img = img

        intersection, union = iou(pred, to_mask(row.gt_object, shape), to_mask(row.ignore_object, shape))
        rows.append({
            "seq": row.seq,
            "time_frame": row.time_frame,
            "mask_age": mask_age,
            "intersection": intersection,
            "union": union,
            "step_time": step_time,
            "pred_rle": to_rle(pred),
        })
    return pd.DataFrame(rows)


def summarize(results):
    """IoU pooled over all frames, mean step times in ms, and the share of frames slower than the frame period"""
    refresh = results["mask_age"] == 0
    return pd.Series({
        "iou": results["intersection"].sum() / results["union"].sum(),
        "segment_ms": results.loc[refresh, "step_time"].mean() * 1e3,
        "propagate_ms": results.loc[~refresh, "step_time"].mean() * 1e3,
        "mean_ms": results["step_time"].mean() * 1e3,
        "late": (results["step_time"] > 1 / FPS).mean(),
        "segment_calls": refresh.sum(),
    })


def error_overlay(img, pred, gt, ignore):
    """Green: correct, red: false positive, blue: missed, yellow: ignored. Returns a BGR image"""
    vis = img.copy()
    valid = ~ignore
    layers = [
        (pred & gt & valid, (0, 255, 0)),
        (pred & ~gt & valid, (0, 0, 255)),
        (~pred & gt & valid, (255, 0, 0)),
        (ignore, (0, 255, 255)),
    ]
    for m, colour in layers:
        vis[m] = 0.5 * vis[m] + 0.5 * np.array(colour)
    return vis
