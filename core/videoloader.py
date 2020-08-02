import os
import cv2
import sys
import time
import platform
import logging
import numpy as np
from multiprocessing import Queue as pQueue
from threading import Thread
from queue import Queue, LifoQueue

try:
    from armv7l.openvino.inference_engine import IECore, IEPlugin
except:
    from openvino.inference_engine import IECore, IEPlugin

logger = logging.getLogger('debug')

class VideoLoader:
    def __init__(self, path, batchSize=10, queueSize=0):
        try:
            if path == "webcam":
                self.stream = cv2.VideoCapture(0)
            else:
                self.stream = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
            logger.info("Loaded stream: " + path)
        except:
            logger.error("Cannot open stream: " + path, exc_info=True)
        
        self.batchSize = batchSize
        self.datalen = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        leftover = 0
        if self.datalen % batchSize:
            leftover = 1
        if self.datalen == -1:
            self.num_batches = 1
        else:
            self.num_batches = self.datalen // batchSize + leftover
        
        self.Q = Queue(maxsize=queueSize)
        
    def start(self):
        self.t = Thread(target=self.update(), args=())
        self.t.daemon = True
        self.t.start()
        self.t.join()
        
        self.stream.release()
        return self
    
    def update(self):
        for i in range(self.num_batches):
            grabbed = True
            
            while grabbed:
                (grabbed, frame) = self.stream.read()

                if not grabbed:
                    logging.error("Reached end of stream")
                    return

                self.Q.put(frame)

    def getFrame(self):
        # return next frame in the queue
        if self.Q.empty():
            return None
        else:
            return self.Q.get()
        
        