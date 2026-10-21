"""Segmentation server: receives JPEG frames from the client and returns the combined mask as RLE."""
import pickle
import time
from multiprocessing.connection import Listener

import cv2
import numpy as np
from ultralytics import YOLO

import pipeline as p


def serve(conn, model):
    while True:
        try:
            frame_id, jpeg = pickle.loads(conn.recv_bytes())
        except EOFError:
            return
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        t0 = time.perf_counter()
        rle = p.to_rle(p.segment(model, img))
        conn.send_bytes(pickle.dumps((frame_id, rle, time.perf_counter() - t0)))


def main():
    model = YOLO("yolo26m-seg.pt", task="segment")
    p.segment(model, np.zeros((375, 1242, 3), np.uint8))  # warm-up
    with Listener(p.SEGMENTER_ADDRESS) as listener:
        print("ready", flush=True)
        while True:
            with listener.accept() as conn:
                serve(conn, model)


if __name__ == "__main__":
    main()
