"""Live demo: plays a video in a loop, asks the segmenter for masks when the policy triggers and shows the
propagated mask, with live charts of the mask accuracy, age, latency and boundary residual.

    python demo.py kitti_mots/training/image_02/0019        # a KITTI MOTS sequence, with ground truth
    python demo.py video.mp4                                 # any video, without ground truth
    SEGMENTER_AUTHKEY=$(cat ~/.segmenter_key) python demo.py kitti_mots/training/image_02/0019 --remote

Keys: space pauses, g shows the ground truth, b the band where the boundary residual is measured,
r the whole residual map, q or Esc quits.
"""
import argparse
import os
import pickle
import secrets
import subprocess
import sys
import time
from collections import deque
from multiprocessing.connection import Client

import cv2
import numpy as np

import pipeline as p
from client import JPEG_QUALITY
from policy_client import Link, Policy, boundary_band, boundary_residual, residual_map, threshold

HISTORY = 300      # frames shown in the charts, 30 s at 10 fps
CHART_H = 70       # height of one chart in px
RESIDUAL_MAX = 40  # residual in gray levels that gets the hottest colour

# BGR
GREEN = (125, 170, 27)
ORANGE = (52, 104, 235)
BLUE = (214, 120, 42)
YELLOW = (90, 190, 230)
GREY = (150, 150, 150)
TEXT = (230, 230, 230)


def kitti_rows(folder):
    """Rows of p.load_frames for a KITTI MOTS training folder, or None for any other folder or a video"""
    folder = os.path.normpath(folder)
    image_dir = os.path.dirname(folder)
    split_dir = os.path.dirname(image_dir)
    root = os.path.dirname(split_dir)
    seq = os.path.basename(folder)
    if (os.path.basename(image_dir) != "image_02" or os.path.basename(split_dir) != "training"
            or not os.path.exists(os.path.join(root, "instances_txt", f"{seq}.txt"))):
        return None
    p.DATA_DIR = root
    return p.load_frames([seq]).sort_values("time_frame")


def read_frames(source, rows):
    """(image, ground-truth RLE, ignore RLE) for every frame of one pass; the RLEs are None without ground truth"""
    if rows is not None:
        for row in rows.itertuples():
            yield cv2.imread(row.image_path), row.gt_object, row.ignore_object
    elif os.path.isdir(source):
        for name in sorted(os.listdir(source)):
            img = cv2.imread(os.path.join(source, name))
            if img is not None:
                yield img, None, None
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"could not open {source}")
        while (frame := cap.read())[0]:
            yield frame[1], None, None
        cap.release()


def video_fps(source, default):
    if os.path.isdir(source):
        return default
    fps = cv2.VideoCapture(source).get(cv2.CAP_PROP_FPS)
    return fps if fps > 0 else default


def resized_mask(rle, original_shape, shape):
    m = p.to_mask(rle, original_shape).astype(np.uint8)
    return cv2.resize(m, shape[::-1], interpolation=cv2.INTER_NEAREST) > 0


def heat(residual, valid):
    out = cv2.applyColorMap((np.clip(residual / RESIDUAL_MAX, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    out[~valid] = 0
    return out


def outline(img, m, colour, thickness):
    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, colour, thickness, cv2.LINE_AA)


def draw_video(img, pred, gt, residual, valid, views, lines):
    show_residual = residual is not None and (views["residual"] or views["band"])
    hot = heat(residual, valid) if show_residual else None
    out = hot.copy() if views["residual"] and show_residual else img.copy()
    if pred is not None:
        out[pred] = (0.6 * out[pred] + 0.4 * np.array(GREEN)).astype(np.uint8)
        outline(out, pred, GREEN, 2)
        if views["band"] and show_residual:
            band = boundary_band(pred) & valid
            out[band] = hot[band]
    if views["gt"] and gt is not None:
        outline(out, gt, (255, 255, 255), 1)
    top = out[:12 + 22 * len(lines)]
    top[:] = (0.45 * top).astype(np.uint8)  # darker band behind the text
    for n, text in enumerate(lines):
        cv2.putText(out, text, (10, 22 + 22 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT, 1, cv2.LINE_AA)
    return out


def chart(width, title, value, values, top, colour, line=None, marks=()):
    """One strip chart of the last HISTORY values, newest on the right; NaN leaves a gap.
    line: a dashed horizontal line (a threshold), marks: (index, colour, row) ticks under the title."""
    img = np.full((CHART_H, width, 3), 28, np.uint8)
    cv2.line(img, (0, CHART_H - 1), (width, CHART_H - 1), (60, 60, 60), 1)
    n = len(values)
    xs = width - 1 - (n - 1 - np.arange(n)) * (width - 1) / (HISTORY - 1)

    def y(v):
        return CHART_H - 6 - np.clip(np.asarray(v, float) / top, 0, 1) * (CHART_H - 36)

    if line is not None:
        for x in range(0, width, 14):
            cv2.line(img, (x, int(y(line))), (x + 7, int(y(line))), GREY, 1)
    for k, c, row in marks:
        cv2.line(img, (int(xs[k]), 19 + 6 * row), (int(xs[k]), 23 + 6 * row), c, 2)
    ys = y(values)
    segment = []
    for k in range(n + 1):
        if k < n and np.isfinite(ys[k]):
            segment.append((xs[k], ys[k]))
        elif segment:
            cv2.polylines(img, [np.round(segment).astype(np.int32)], False, colour, 2, cv2.LINE_AA)
            segment = []
    cv2.putText(img, title, (8, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1, cv2.LINE_AA)
    (w, _), _ = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(img, value, (width - w - 8, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT, 1, cv2.LINE_AA)
    return img


def top_of(values, minimum):
    values = np.asarray(values, float)
    return max(minimum, np.nanmax(values) * 1.1) if np.isfinite(values).any() else minimum


def draw_charts(width, history, policy, loop_iou, fps):
    h = list(history)
    last = h[-1]
    sent = [k for k, f in enumerate(h) if f["sent"]]
    arrived = [k for k, f in enumerate(h) if f["arrived"]]
    charts = []
    if last["has_gt"]:
        charts.append(chart(width, "IoU with the ground truth (white outline)",
                            f"{last['iou']:.2f} now   {loop_iou:.2f} this loop" if np.isfinite(last["iou"]) else "-",
                            [f["iou"] for f in h], 1.0, GREEN))
    age = last["age"]
    charts.append(chart(width, "mask age (frames)   ticks: request sent (orange), mask arrived (green)",
                        f"{'-' if np.isnan(age) else int(age)}   {last['pending']} pending   "
                        f"{len(sent) / (len(h) / fps):.1f} requests/s",
                        [f["age"] for f in h], top_of([f["age"] for f in h], 30), BLUE,
                        marks=[(k, ORANGE, 0) for k in sent] + [(k, GREEN, 1) for k in arrived]))
    latencies = [f["latency"] for f in h]
    top = top_of(latencies, 1.0)
    charts.append(chart(width, f"latency of the shown mask (s), axis up to {top:.1f} s",
                        "-" if np.isnan(last["latency"]) else f"{last['latency'] * 1e3:.0f} ms",
                        latencies, top, ORANGE))
    tb = policy.t_boundary
    boundary = [f["boundary"] for f in h]
    charts.append(chart(width, "boundary residual since the mask's source frame" + (f", threshold {tb:g}" if tb else ""),
                        "-" if np.isnan(last["boundary"]) else f"{last['boundary']:.0f}",
                        boundary, top_of(boundary, tb * 1.5 if tb else 1.0), YELLOW, line=tb))
    return np.vstack(charts)


def play(source, rows, conn, policy, delay, link, fps, width, loop, title, save):
    """Plays until the window is closed (or once without loop). Returns the number of answers still on their way."""
    sent, extra = {}, {}  # frame index -> time its request was sent, extra time from the simulated link
    received = 0
    history = deque(maxlen=HISTORY)
    views = {"gt": True, "band": False, "residual": False}
    writer = None
    start = time.perf_counter()
    i = 0
    try:
        while True:
            # every pass over the video is a new scene: answers for frames before the cut are dropped
            cut = i
            pred = source_frame = prev_img = residual = valid = None
            flows, boundaries, pending, waiting = {}, {}, set(), []
            boundary_acc, latency, loop_inter, loop_union = 0.0, np.nan, 0, 0
            policy.last_sent = None
            link.start("0")
            for img, gt_rle, ignore_rle in read_frames(source, rows):
                original_shape = img.shape[:2]
                if width and img.shape[1] != width:
                    img = cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1])))
                shape = img.shape[:2]

                flow = None
                if prev_img is not None:
                    flow = flows[i] = p.backward_flow(prev_img, img)

                while conn.poll():
                    j, rle, _ = pickle.loads(conn.recv_bytes())
                    received += 1
                    if j >= cut:
                        waiting.append((max(time.perf_counter(), sent[j] + delay) + extra[j], j, rle))

                now = time.perf_counter()
                usable = [w for w in waiting if w[0] <= now and (source_frame is None or w[1] > source_frame)]
                if usable:
                    _, source_frame, rle = max(usable, key=lambda w: w[1])
                    waiting = [w for w in waiting if w[1] > source_frame]
                    pending = {s for s in pending if s > source_frame}
                    pred = p.to_mask(rle, shape)
                    for k in range(source_frame + 1, i + 1):
                        pred = p.warp(pred, flows[k])
                    flows = {k: f for k, f in flows.items() if k > source_frame}
                    latency = now - sent[source_frame]
                    boundaries = {k: b for k, b in boundaries.items() if k > source_frame}
                    boundary_acc = sum(boundaries.values())
                elif flow is not None and pred is not None:
                    pred = p.warp(pred, flow)

                if flow is not None:
                    residual, valid = residual_map(prev_img, img, flow)
                    if pred is not None:
                        boundaries[i] = boundary_residual(residual, valid, pred)
                        boundary_acc += boundaries[i]
                prev_img = img

                signals = {"boundary": boundary_acc if flow is not None else np.nan, "tiles": np.nan}
                send = policy.should_send(i, source_frame, pending, signals)
                if send:
                    jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1].tobytes()
                    request = pickle.dumps((i, jpeg))
                    conn.send_bytes(request)
                    sent[i] = time.perf_counter()
                    extra[i] = link.extra(sent[i], len(request))
                    pending.add(i)

                gt, frame_iou = None, np.nan
                if gt_rle is not None:
                    gt = resized_mask(gt_rle, original_shape, shape)
                    ignore = resized_mask(ignore_rle, original_shape, shape)
                    current = pred if pred is not None else np.zeros(shape, dtype=bool)
                    inter, union = p.iou(current, gt, ignore)
                    loop_inter, loop_union = loop_inter + inter, loop_union + union
                    frame_iou = inter / union if union else np.nan

                history.append({
                    "has_gt": gt is not None,
                    "iou": frame_iou,
                    "age": np.nan if source_frame is None else i - source_frame,
                    "pending": len(pending),
                    "latency": latency,
                    "boundary": signals["boundary"],
                    "sent": send,
                    "arrived": bool(usable),
                })
                lines = [title, f"frame {i - cut}   {link.describe()}"]
                video = draw_video(img, pred, gt, residual, valid, views, lines)
                shown = np.vstack([video, draw_charts(shape[1], history, policy,
                                                      loop_inter / loop_union if loop_union else np.nan, fps)])
                if save and writer is None:
                    writer_size = shown.shape[1::-1]
                    writer = cv2.VideoWriter(save, cv2.VideoWriter_fourcc(*"mp4v"), fps, writer_size)
                if writer is not None and shown.shape[1::-1] == writer_size:
                    writer.write(shown)
                cv2.imshow(title, shown)

                wait = max(1, int((start + (i + 1) / fps - time.perf_counter()) * 1e3))
                paused = False
                while True:
                    key = cv2.waitKey(wait) & 0xFF
                    if key in (ord("q"), 27) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                        return len(sent) - received
                    if key == ord(" "):
                        paused = not paused
                    for name, k in [("gt", "g"), ("band", "b"), ("residual", "r")]:
                        if key == ord(k):
                            views[name] = not views[name]
                    if not paused:
                        break
                    wait = 50
                    start += 0.05  # the pause does not count as playing time
                i += 1
            if not loop:
                return len(sent) - received
    finally:
        if writer is not None:
            writer.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", help="KITTI MOTS sequence folder, other folder of frames, or video file")
    parser.add_argument("--remote", action="store_true", help="connect to a running segmenter instead of starting one")
    parser.add_argument("--policy", choices=["fixed", "max_interval", "residual"], default="residual")
    parser.add_argument("--n", type=int, default=5, help="fixed: send every N frames")
    parser.add_argument("--k", type=int, default=30, help="max_interval and residual: largest mask age in frames")
    parser.add_argument("--t-boundary", type=threshold, default=100.0, help="residual: number or 'none'")
    parser.add_argument("--t-tiles", type=threshold, default=None, help="residual: number or 'none'")
    parser.add_argument("--delay", type=float, default=0.0, help="earliest time in seconds a mask is used after its request")
    parser.add_argument("--uplink-kbit", type=float, help="simulated uplink rate in kbit/s (default: not limited)")
    parser.add_argument("--net-delay", type=float, default=0.0, help="simulated extra network delay in seconds")
    parser.add_argument("--jitter", type=float, default=0.0, help="simulated random extra delay in seconds (std of a half-normal)")
    parser.add_argument("--seed", type=int, default=0, help="seed for the jitter")
    parser.add_argument("--fps", type=float, help="playback rate (default: the video's, or 10 for a folder)")
    parser.add_argument("--width", type=int, default=1242, help="resize frames to this width, 0 keeps them as they are")
    parser.add_argument("--once", action="store_true", help="play the video once instead of in a loop")
    parser.add_argument("--save", help="also write the shown frames to this video file")
    args = parser.parse_args()

    if args.policy == "fixed":
        policy = Policy("fixed", n=args.n)
    elif args.policy == "max_interval":
        policy = Policy("max_interval", k=args.k)
    else:
        policy = Policy("residual", k=args.k, t_boundary=args.t_boundary, t_tiles=args.t_tiles)
    link = Link(args.uplink_kbit, args.net_delay, args.jitter, args.seed)
    rows = kitti_rows(args.video)
    fps = args.fps or (p.FPS if rows is not None else video_fps(args.video, p.FPS))
    title = f"{policy.name} {policy.params}".strip()

    server = None
    if not args.remote:
        os.environ.setdefault("SEGMENTER_AUTHKEY", secrets.token_hex(16))
        server = subprocess.Popen([sys.executable, "segmenter.py"], stdout=subprocess.PIPE, text=True)
    try:
        while server and (line := server.stdout.readline()).strip() != "ready":
            if not line:
                raise RuntimeError("segmenter process exited before it was ready")
        with Client(p.SEGMENTER_ADDRESS, authkey=os.environ["SEGMENTER_AUTHKEY"].encode()) as conn:
            outstanding = play(args.video, rows, conn, policy, args.delay, link, fps, args.width,
                               not args.once, title, args.save)
            for _ in range(outstanding):  # so the segmenter does not write to a closed connection
                conn.recv_bytes()
    finally:
        cv2.destroyAllWindows()
        if server:
            server.terminate()


if __name__ == "__main__":
    main()
