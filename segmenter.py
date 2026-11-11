"""Segmentation server: receives JPEG frames from the client and returns the combined mask as RLE."""
import os
import pickle
import socket
import time
from multiprocessing import AuthenticationError
from multiprocessing.connection import Listener

import cv2
import numpy as np
from ultralytics import YOLO

import pipeline as p


def serve(conn, model):
    while True:
        try:
            frame_id, jpeg = pickle.loads(conn.recv_bytes())
        except (EOFError, ConnectionResetError):
            return
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        t0 = time.perf_counter()
        rle = p.to_rle(p.segment(model, img))
        try:
            conn.send_bytes(pickle.dumps((frame_id, rle, time.perf_counter() - t0)))
        except (BrokenPipeError, ConnectionResetError):  # client went away, wait for the next one
            return


def main():
    model = YOLO("yolo26m-seg.pt", task="segment")
    p.segment(model, np.zeros((375, 1242, 3), np.uint8))  # warm-up
    authkey = os.environ["SEGMENTER_AUTHKEY"].encode()  # messages are pickled, so only accept authenticated clients
    with Listener(p.SEGMENTER_ADDRESS, authkey=authkey) as listener:
        print(f"listening on {socket.gethostname()}:{p.SEGMENTER_ADDRESS[1]}", flush=True)
        print("ready", flush=True)
        while True:
            try:
                conn = listener.accept()
            except (AuthenticationError, EOFError, ConnectionResetError) as e:  # wrong key or dropped during the handshake
                print(f"rejected a client: {e!r}", flush=True)
                continue
            with conn:
                serve(conn, model)


if __name__ == "__main__":
    main()
