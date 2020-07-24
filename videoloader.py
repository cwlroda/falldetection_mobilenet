import os
import cv2
import sys
import time
import platform
import numpy as np
from multiprocessing import Queue as pQueue
from threading import Thread
from queue import Queue, LifoQueue

try:
    from armv7l.openvino.inference_engine import IECore, IEPlugin
except:
    from openvino.inference_engine import IECore, IEPlugin

class VideoLoader:
    def __init__(self, path, batchSize=1, queueSize=300):
        self.path = path
        self.stream = cv2.VideoCapture(self.path)
        assert self.stream.isOpened(), 'Cannot capture source'
        self.stopped = False
        
        self.batchSize = batchSize
        self.datalen = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        leftover = 0
        if (self.datalen) % batchSize:
            leftover = 1
        self.num_batches = self.datalen // batchSize + leftover
        
        self.Q = Queue(maxsize=queueSize)
        
    def start(self):
        t = Thread(target=self.update(), args=())
        t.daemon = True
        t.start()

        return self
    
    def update(self):
        for i in range(self.num_batches):
            grabbed = True

            while grabbed:
                (grabbed, frame) = self.stream.read() 
                
                if not grabbed:
                    return
                
                self.Q.put(frame)

    def getitem(self):
        # return next frame in the queue
        if self.Q.empty():
            return None
        else:
            return self.Q.get()
        
        