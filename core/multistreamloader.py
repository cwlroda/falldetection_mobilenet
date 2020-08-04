import sys
import os
import io
import time
import re
import cv2
import logging
import numpy as np
from threading import Thread
from queue import Queue, LifoQueue

from core.videoloader import VideoLoader

logger = logging.getLogger('debug')

class MultiStreamLoader:
    def __init__(self, model, RTSP_list):
        self.model = model
        self.RTSP_list = RTSP_list
        self.streams = []
        
    def generateStreams(self):
        for RTSPdict in self.RTSP_list:
            ref_RTSPHandler = None
            str_RTSPURL = None
            ID = None
            
            if "RTSPURL" in RTSPdict:
                str_RTSPURL = RTSPdict["RTSPURL"]
            if "ID" in RTSPdict:
                ID = RTSPdict["ID"]
            logger.info("Loading stream: " + str_RTSPURL)
            
            t1 = time.perf_counter()
            ref_RTSPHandler = RTSPHandler(self.model, RTSPdict, str_RTSPURL, ID)
            t2 = time.perf_counter()
            
            elapsedTime = t2-t1
            logger.info("Time elapsed: " + str(elapsedTime))
            
            self.streams.append(ref_RTSPHandler)
            logger.info("Finished running " + str_RTSPURL)
        
        return self.streams
    
    def getStreams(self):
        return self.streams
    
    def getFrames(self):
        frames = []
        for stream in self.streams:
            frame = stream.getFrame()
            
            if frame is not None:
                frames.append(frame)
                
        return frames

class RTSPHandler:
    def __init__(self, model, dict, RTSPURL=None, ID=None):
        self.model = model
        self.RTSPdict = dict
        self.RTSPURL = RTSPURL
        self.ID = ID
        self.Q = Queue(maxsize=0)
        self.frame = None
        self.droppedFrames = 0
        
        self.online = True
        self.makeConnection()
    
    def makeConnection(self):
        try:
            self.stream = cv2.VideoCapture(self.RTSPURL, cv2.CAP_FFMPEG)
            
            if self.stream.isOpened():
                logger.info("Loaded stream: " + self.RTSPURL)
                self.t = Thread(target=self.update, args=())
                self.t.daemon = True
                self.t.start()
                logger.info("RTSP thread started")
                
        except:
            logger.error("Cannot open stream: " + self.RTSPURL, exc_info=True)
    
    def reconnect(self):
        self.stream.release()
        self.frame = None
        self.droppedFrames = 0
        self.stream = cv2.VideoCapture(self.RTSPURL, cv2.CAP_FFMPEG)
        
        if self.stream.isOpened():
            logger.info("Reconnected to stream: " + self.RTSPURL)
            self.online = True
            return True
        else:
            logger.error("Cannot reconnect to stream: " + self.RTSPURL, exc_info=True)
            return False
        
    def update(self):
        while True:
            if self.stream.isOpened():
                (self.grabbed, self.frame) = self.stream.read()

                if self.grabbed:
                    self.droppedFrames = 0
                else:
                    self.droppedFrames += 1
                    
                    if self.droppedFrames > 60:
                        self.online = False
                
            while not self.stream.isOpened() or not self.online:
                logger.info("Reconnecting to stream: " + self.RTSPURL)
                self.online = self.reconnect()
    
    def getFrame(self):
        return (self.frame, self.ID)
    
    