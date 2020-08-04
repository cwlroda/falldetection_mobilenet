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
    def __init__(self, stream, batchSize=10, queueSize=0):        
        self.stream = stream
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
        self.online = True
        
    def start(self):
        self.t = Thread(target=self.update(), args=())
        self.t.daemon = True
        self.t.start()
        self.t.join()
        
        return self
    
    def update(self):
        for i in range(self.num_batches):            
            while True:
                (self.grabbed, frame) = self.stream.read()

                if self.grabbed:
                    self.droppedFrames = 0
                else:
                    self.droppedFrames += 1
                    
                    if self.droppedFrames > 60:
                        self.online = False
                
                if not self.stream.isOpened() or not self.online:
                    self.online = False
                    return

                self.Q.put(frame)
                # time.sleep(0.5)

    def getFrame(self):
        # return next frame in the queue
        if self.Q.empty():
            return None
        else:
            return self.Q.get()

    def isOnline(self):
        return self.online