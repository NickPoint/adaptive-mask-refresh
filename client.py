import os
import pickle
import subprocess
import sys
import time
from multiprocessing.connection import Client

import cv2
import numpy as np
import pandas as pd

import pipeline as p

JPEG_QUALITY = 90
INTERVAL = 5
DELAYS = [0.0, 0.15, 0.35, 0.55]  # seconds
OUT_DIR = "results"


def run_remote(seq_frames, conn, interval, use_flow, delay, fps=p.FPS):
    rows = []
    pred = source = prev_img = None
    flows = {}    # frame index -> flow back to the previous frame, kept for catching up late masks and not run optical flow again on old frames
    sent = {}     # frame index -> time its request was sent
    waiting = []  # (time the mask becomes usable, frame index, RLE)
    received = 0
    start = time.perf_counter()
    for i, row in enumerate(seq_frames.sort_values("time_frame").itertuples()):
        time.sleep(max(0.0, start + i / fps - time.perf_counter()))
        img = cv2.imread(row.image_path)
        shape = img.shape[:2]

        t0 = time.perf_counter()
        bytes_up = bytes_down = 0
        latency = np.nan

        if i % interval == 0:
            jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1].tobytes()
            request = pickle.dumps((i, jpeg))
            conn.send_bytes(request)
            sent[i] = time.perf_counter()
            bytes_up = len(request)

        while conn.poll():
            response = conn.recv_bytes()
            received += 1
            bytes_down += len(response)
            j, rle, _ = pickle.loads(response)
            waiting.append((max(time.perf_counter(), sent[j] + delay), j, rle))

        flow = None
        if use_flow and prev_img is not None:
            flow = flows[i] = p.backward_flow(prev_img, img)
        prev_img = img

        now = time.perf_counter()
        usable = [w for w in waiting if w[0] <= now]
        if usable:
            _, source, rle = max(usable, key=lambda w: w[1])
            waiting = [w for w in waiting if w[1] > source]
            pred = p.to_mask(rle, shape)
            if use_flow:
                for k in range(source + 1, i + 1): # here source is the frame index and i + 1 depicts current frame of client
                    pred = p.warp(pred, flows[k]) # so we need to propogate i - source frames. This is done by already precomputed flow of pixels so we do not need to store old frames and run optical flow again.
                flows = {k: f for k, f in flows.items() if k > source}
            latency = now - sent[source]
        elif flow is not None and pred is not None:
            pred = p.warp(pred, flow)
        step_time = time.perf_counter() - t0

        current = pred if pred is not None else np.zeros(shape, dtype=bool)
        intersection, union = p.iou(current, p.to_mask(row.gt_object, shape), p.to_mask(row.ignore_object, shape))
        rows.append({
            "seq": row.seq,
            "time_frame": row.time_frame,
            "mask_age": np.nan if source is None else i - source,
            "intersection": intersection,
            "union": union,
            "step_time": step_time,
            "latency": latency,
            "bytes_up": bytes_up,
            "bytes_down": bytes_down,
        })

    for _ in range(len(sent) - received):  # late answers must not leak into the next sequence
        conn.recv_bytes()
    return pd.DataFrame(rows)


def summarize(results):
    seconds = len(results) / p.FPS
    return pd.Series({
        "iou": results["intersection"].sum() / results["union"].sum(),
        "mask_age": results["mask_age"].mean(),
        "latency_ms": results["latency"].mean() * 1e3,
        "requests": (results["bytes_up"] > 0).sum(),
        "up_kbit_s": results["bytes_up"].sum() * 8 / seconds / 1e3,
        "down_kbit_s": results["bytes_down"].sum() * 8 / seconds / 1e3,
        "client_ms": results["step_time"].mean() * 1e3,
        "late": (results["step_time"] > 1 / p.FPS).mean(),
    })


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    frames = p.load_frames(p.DEV_SEQS)

    server = subprocess.Popen([sys.executable, "segmenter.py"], stdout=subprocess.PIPE, text=True)
    try:
        while (line := server.stdout.readline()).strip() != "ready":
            if not line:
                raise RuntimeError("segmenter process exited before it was ready")
        runs = []
        with Client(p.SEGMENTER_ADDRESS) as conn:
            for method, use_flow in [("hold", False), ("flow", True)]:
                for delay in DELAYS:
                    print(f"{method}, delay {delay * 1e3:.0f} ms", flush=True)
                    for _, seq_frames in frames.groupby("seq"):
                        run = run_remote(seq_frames, conn, INTERVAL, use_flow, delay)
                        run.insert(0, "delay_ms", int(delay * 1e3))
                        run.insert(0, "method", method)
                        runs.append(run)
    finally:
        server.terminate()

    results = pd.concat(runs, ignore_index=True)
    results.to_csv(f"{OUT_DIR}/delay_frames.csv", index=False)


if __name__ == "__main__":
    main()
