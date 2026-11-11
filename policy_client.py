import argparse
import itertools
import os
import pickle
import secrets
import subprocess
import sys
import time
from multiprocessing.connection import Client

import cv2
import numpy as np
import pandas as pd

import pipeline as p
from client import DELAYS, JPEG_QUALITY, OUT_DIR, summarize

MIN_INTERVAL = 3      # frames between two requests (adaptive policies)
MAX_PENDING = 2       # requests sent and not yet adopted (adaptive policies)
BAND = 7              # width in px of the band around the mask boundary
TILE = 32             # tile size in px
TILE_RESIDUAL = 12.0  # a tile is changed when its mean residual (gray levels) is above this
MIN_VALID = 0.5       # tiles with less valid (inside the image after warping) pixels are ignored


class Policy:
    """Decides on every frame whether to ask the segmenter for a new mask."""

    def __init__(self, name, n=None, k=None, t_boundary=None, t_tiles=None):
        self.name, self.n, self.k, self.t_boundary, self.t_tiles = name, n, k, t_boundary, t_tiles
        self.last_sent = None

    @property
    def params(self):
        values = {"N": self.n, "K": self.k, "tb": self.t_boundary, "tt": self.t_tiles}
        return ",".join(f"{key}={v}" for key, v in values.items() if v is not None)

    def should_send(self, i, displayed_source, pending_sources, signals):
        """displayed_source: frame the shown mask was segmented from (None before the first mask).
        pending_sources: frames of requests that are sent and not yet adopted. signals: dict with boundary, tiles (NaN if unknown)."""
        send = self._decide(i, displayed_source, pending_sources, signals)
        if send:
            self.last_sent = i
        return send

    def _decide(self, i, displayed_source, pending_sources, signals):
        if self.name == "fixed":
            return i % self.n == 0
        if self.last_sent is not None and i - self.last_sent < MIN_INTERVAL:
            return False
        if len(pending_sources) >= MAX_PENDING:
            return False
        newer_in_flight = any(displayed_source is None or s > displayed_source for s in pending_sources)
        if newer_in_flight:
            return False
        if displayed_source is None:
            return True
        if i - displayed_source >= self.k:
            return True
        if self.name == "residual":
            return bool((self.t_boundary is not None and signals["boundary"] > self.t_boundary)
                        or (self.t_tiles is not None and signals["tiles"] > self.t_tiles))
        return False


def residual_map(prev_img, img, flow):
    """Abs difference between img and prev_img warped onto it, after removing each frame's mean brightness.
    Also returns which pixels have a source inside the image."""
    prev = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    cur = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    prev -= prev.mean()
    cur -= cur.mean()
    h, w = cur.shape
    ys, xs = np.indices(cur.shape, dtype=np.float32)
    sx, sy = xs + flow[..., 0], ys + flow[..., 1]
    warped = cv2.remap(prev, sx, sy, cv2.INTER_LINEAR)
    valid = (sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)
    return np.abs(cur - warped), valid


def boundary_band(m):
    """Pixels within about BAND / 2 px of the mask boundary, on both sides"""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BAND, BAND))
    m = m.astype(np.uint8)
    return (cv2.dilate(m, k) > 0) & (cv2.erode(m, k) == 0)


def boundary_residual(residual, valid, m):
    """Mean residual in a band around the mask boundary (0 when there is no usable band)"""
    band = boundary_band(m) & valid
    return float(residual[band].mean()) if band.any() else 0.0


def tile_fraction(residual, valid, m):
    """Share of the tiles without mask whose mean residual is above TILE_RESIDUAL"""
    h, w = residual.shape
    h, w = h // TILE * TILE, w // TILE * TILE

    def tiles(a):
        return a[:h, :w].reshape(h // TILE, TILE, w // TILE, TILE).sum(axis=(1, 3))

    count = tiles(valid.astype(np.float32))
    mean = tiles(np.where(valid, residual, 0.0)) / np.maximum(count, 1)
    outside = (tiles(m.astype(np.float32)) == 0) & (count >= MIN_VALID * TILE * TILE)
    if not outside.any():
        return 0.0
    return float((mean[outside] > TILE_RESIDUAL).mean())


class Link:
    """Simulated network on top of the real one: a limited uplink (requests queue behind each other),
    a fixed extra delay and a random extra delay (half-normal, so it only adds)."""

    def __init__(self, uplink_kbit=None, delay=0.0, jitter=0.0, seed=0):
        self.uplink_kbit, self.delay, self.jitter, self.seed = uplink_kbit, delay, jitter, seed

    def describe(self):
        if not (self.uplink_kbit or self.delay or self.jitter):
            return "network as it is"
        parts = [f"upload {self.uplink_kbit:g} kbit/s"] if self.uplink_kbit else []
        if self.delay or self.jitter:
            parts.append(f"+{self.delay * 1e3:.0f} ms" + (f" + jitter {self.jitter * 1e3:.0f} ms" if self.jitter else ""))
        return "simulated " + ", ".join(parts)

    def start(self, seq):
        self.rng = np.random.default_rng([self.seed, int(seq)])
        self.free = 0.0  # time the uplink has sent everything queued so far

    def extra(self, now, size):
        """Time an answer to a request of `size` bytes, sent at `now`, arrives later than on the real network."""
        done = now
        if self.uplink_kbit:
            done = max(now, self.free) + size * 8 / (self.uplink_kbit * 1e3)
            self.free = done
        return done - now + self.delay + abs(self.rng.normal(0.0, self.jitter))


def run_policy(seq_frames, conn, policy, delay, link=None, fps=p.FPS):
    link = link or Link()
    link.start(seq_frames["seq"].iloc[0])
    rows = []
    pred = source = prev_img = None
    flows = {}    # frame index -> flow back to the previous frame
    sent = {}     # frame index -> time its request was sent
    extra = {}    # frame index -> extra time from the simulated link
    pending = []  # frames of requests that are sent and not adopted yet
    waiting = []  # (time the mask becomes usable, frame index, RLE)
    received = 0
    boundary_acc = 0.0
    boundaries = {}  # frame index -> boundary residual of that frame, to sum it again from a late mask's source frame
    policy.last_sent = None
    start = time.perf_counter()
    for i, row in enumerate(seq_frames.sort_values("time_frame").itertuples()):
        time.sleep(max(0.0, start + i / fps - time.perf_counter()))
        img = cv2.imread(row.image_path)
        shape = img.shape[:2]

        t0 = time.perf_counter()
        bytes_up = bytes_down = 0
        latency = np.nan

        flow = None
        if prev_img is not None:  # computed once, used for the trigger and for the propagation
            flow = flows[i] = p.backward_flow(prev_img, img)

        while conn.poll():
            response = conn.recv_bytes()
            received += 1
            bytes_down += len(response)
            j, rle, _ = pickle.loads(response)
            waiting.append((max(time.perf_counter(), sent[j] + delay) + extra[j], j, rle))

        now = time.perf_counter()
        usable = [w for w in waiting if w[0] <= now and (source is None or w[1] > source)]
        if usable:
            _, source, rle = max(usable, key=lambda w: w[1])
            waiting = [w for w in waiting if w[1] > source]
            pending = [s for s in pending if s > source]
            pred = p.to_mask(rle, shape)
            for k in range(source + 1, i + 1):
                pred = p.warp(pred, flows[k])
            flows = {k: f for k, f in flows.items() if k > source}
            latency = now - sent[source]
            # the new mask has drifted since its source frame; the frames in between were measured around the old mask
            boundaries = {k: b for k, b in boundaries.items() if k > source}
            boundary_acc = sum(boundaries.values())
        elif flow is not None and pred is not None:
            pred = p.warp(pred, flow)

        t1 = time.perf_counter()
        signals = {"boundary": np.nan, "tiles": np.nan}
        if flow is not None:
            residual, valid = residual_map(prev_img, img, flow)
            current = pred if pred is not None else np.zeros(shape, dtype=bool)
            signals["boundary"] = boundary_residual(residual, valid, current)
            signals["tiles"] = tile_fraction(residual, valid, current)
            boundary_acc += signals["boundary"]
            boundaries[i] = signals["boundary"]
        trigger_time = time.perf_counter() - t1

        decision = dict(signals, boundary=boundary_acc if flow is not None else np.nan)
        send = policy.should_send(i, source, pending, decision)
        if send:
            jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1].tobytes()
            request = pickle.dumps((i, jpeg))
            conn.send_bytes(request)
            sent[i] = time.perf_counter()
            extra[i] = link.extra(sent[i], len(request))
            pending.append(i)
            bytes_up = len(request)
        prev_img = img
        step_time = time.perf_counter() - t0

        current = pred if pred is not None else np.zeros(shape, dtype=bool)
        intersection, union = p.iou(current, p.to_mask(row.gt_object, shape), p.to_mask(row.ignore_object, shape))
        rows.append({
            "seq": row.seq,
            "time_frame": row.time_frame,
            "policy": policy.name,
            "params": policy.params,
            "delay_ms": int(delay * 1e3),
            "uplink_kbit": link.uplink_kbit or np.nan,
            "net_delay_ms": int(link.delay * 1e3),
            "jitter_ms": int(link.jitter * 1e3),
            "mask_age": np.nan if source is None else i - source,
            "intersection": intersection,
            "union": union,
            "residual_boundary": signals["boundary"],
            "residual_boundary_acc": decision["boundary"],
            "residual_tiles": signals["tiles"],
            "sent": send,
            "pending": len(pending),
            "trigger_time": trigger_time,
            "step_time": step_time,
            "latency": latency,
            "bytes_up": bytes_up,
            "bytes_down": bytes_down,
        })

    for _ in range(len(sent) - received):  # late answers must not leak into the next sequence
        conn.recv_bytes()
    return pd.DataFrame(rows)


def make_policies(names, ns, ks, t_boundaries, t_tiles):
    policies = []
    if "fixed" in names:
        policies += [Policy("fixed", n=n) for n in ns]
    if "max_interval" in names:
        policies += [Policy("max_interval", k=k) for k in ks]
    if "residual" in names:
        for k, tb, tt in itertools.product(ks, t_boundaries, t_tiles):
            if tb is not None or tt is not None:  # with both off it is max_interval
                policies.append(Policy("residual", k=k, t_boundary=tb, t_tiles=tt))
    return policies


def threshold(text):
    return None if text == "none" else float(text)


def summary_table(results):
    def one(g):
        s = summarize(g)
        s["requests_s"] = s["requests"] / (len(g) / p.FPS)
        s["trigger_ms"] = g["trigger_time"].mean() * 1e3
        return s[["iou", "mask_age", "requests_s", "up_kbit_s", "trigger_ms"]]
    return results.groupby(["policy", "params", "delay_ms"]).apply(one)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", action="store_true", help="connect to a running segmenter instead of starting one")
    parser.add_argument("--split", choices=["dev", "heldout"], default="dev")
    parser.add_argument("--policies", nargs="+", choices=["fixed", "max_interval", "residual"],
                        default=["fixed", "max_interval", "residual"])
    parser.add_argument("--n", nargs="+", type=int, default=[5, 10], help="fixed: send every N frames")
    parser.add_argument("--k", nargs="+", type=int, default=[10, 20, 30], help="max_interval and residual: largest mask age")
    parser.add_argument("--t-boundary", nargs="+", type=threshold, default=[None, 30.0], help="residual: numbers or 'none'")
    parser.add_argument("--t-tiles", nargs="+", type=threshold, default=[None, 0.15], help="residual: numbers or 'none'")
    parser.add_argument("--delays", nargs="+", type=float, default=DELAYS, help="seconds")
    parser.add_argument("--uplink-kbit", type=float, help="simulated uplink rate in kbit/s (default: not limited)")
    parser.add_argument("--net-delay", type=float, default=0.0, help="simulated extra network delay in seconds, added to every answer")
    parser.add_argument("--jitter", type=float, default=0.0, help="simulated random extra delay in seconds (std of a half-normal)")
    parser.add_argument("--seed", type=int, default=0, help="seed for the jitter")
    parser.add_argument("--out", help="CSV for the per-frame results (default: results/policy_frames_<split>.csv)")
    parser.add_argument("--overwrite", action="store_true", help="replace --out if it already exists")
    args = parser.parse_args()

    out = args.out or f"{OUT_DIR}/policy_frames_{args.split}.csv"
    if os.path.exists(out) and not args.overwrite:
        sys.exit(f"{out} already exists, choose another --out or pass --overwrite")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    frames = p.load_frames(p.DEV_SEQS if args.split == "dev" else p.HELDOUT_SEQS)
    link = Link(args.uplink_kbit, args.net_delay, args.jitter, args.seed)
    policies = make_policies(args.policies, args.n, args.k, args.t_boundary, args.t_tiles)

    server = None
    if not args.remote:
        os.environ.setdefault("SEGMENTER_AUTHKEY", secrets.token_hex(16))
        server = subprocess.Popen([sys.executable, "segmenter.py"], stdout=subprocess.PIPE, text=True)
    try:
        while server and (line := server.stdout.readline()).strip() != "ready":
            if not line:
                raise RuntimeError("segmenter process exited before it was ready")
        runs = []
        with Client(p.SEGMENTER_ADDRESS, authkey=os.environ["SEGMENTER_AUTHKEY"].encode()) as conn:
            passes = list(itertools.product(policies, args.delays))
            for n, (policy, delay) in enumerate(passes, 1):
                for _, seq_frames in frames.groupby("seq"):
                    runs.append(run_policy(seq_frames, conn, policy, delay, link))
                # save after every pass, so a dropped connection loses at most the current one
                results = pd.concat(runs, ignore_index=True)
                results.to_csv(out, index=False)
                done = results[(results["policy"] == policy.name) & (results["params"] == policy.params)
                               & (results["delay_ms"] == int(delay * 1e3))]
                s = summarize(done)
                print(f"[{n}/{len(passes)}] {policy.name} {policy.params}, delay {delay * 1e3:.0f} ms: "
                      f"IoU {s['iou']:.3f}, {s['requests'] / (len(done) / p.FPS):.2f} requests/s", flush=True)
    finally:
        if server:
            server.terminate()

    print(f"saved to {out}")
    print(summary_table(results).round(3).to_string())


if __name__ == "__main__":
    main()
