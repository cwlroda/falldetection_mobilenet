import sys
import os
import io
import time
import re
import cv2
import logging
import numpy as np
from queue import Queue, LifoQueue

from core.videoloader import VideoLoader
from core.detectionloader import DetectionLoader

logger = logging.getLogger('debug')

class MultiStreamLoader:
    def __init__(self, model, RTSP_list):
        self.model = model
        self.RTSP_list = RTSP_list
        self.streams = []
        
    def loadStreams(self):
        for dict in self.RTSP_list:
            ref_RTSPHandler = None
            str_RTSPURL = None
            
            if "RTSPURL" in dict:
                str_RTSPURL = dict["RTSPURL"]
            logger.info("Loading stream: " + str_RTSPURL)
            
            t1 = time.perf_counter()
            ref_RTSPHandler = RTSPHandler(self.model, dict, str_RTSPURL)
            t2 = time.perf_counter()
            
            elapsedTime = t2-t1
            logger.info("Time elapsed: " + str(elapsedTime))
            
            self.streams.append(ref_RTSPHandler)
            logger.info("Finished running " + str_RTSPURL)
        
        return self.streams
    

class RTSPHandler:
    def __init__(self, model, dict, RTSPURL=None):
        self.model = model
        self.RTSPdict = dict
        self.RTSPURL = RTSPURL
        self.outframes = []
        
        try:
            self.data = VideoLoader(self.RTSPURL, batchSize=1, queueSize=0)
            logger.info("Reading frames")
            self.data.start()
            self.start()
        except:
            import traceback
            traceback.print_exc()
    
    def start(self):
        try:
            logger.info("Performing inference")
            detection = DetectionLoader(self.model, self.data, queueSize=0)
            self.outframes = detection.start()
            logger.info("Inference done")
            
        except:
            import traceback
            traceback.print_exc()
            
    def getFrames(self):
        return self.outframes
    
    