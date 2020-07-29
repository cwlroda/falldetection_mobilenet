import sys
import os
import io
import time
import re
import cv2
import numpy as np
import argparse
from queue import Queue, LifoQueue

from videoloader import VideoLoader
from detectionloader import DetectionLoader

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

camera_width = 1080
camera_height = 720

parser = argparse.ArgumentParser()
# parser.add_argument("-d", "--device", help="Specify the target device to infer on; CPU, GPU, MYRIAD is acceptable. (Default=CPU)", default="CPU", type=str)
# parser.add_argument("-b", "--boost", help="Setting it to True will make it run faster instead of sacrificing accuracy. (Default=False)", default=False, type=bool)
# parser.add_argument("-m", "--mode", help="Mode 0: Image display with multithreading. Mode 1: Video display without multithreading. (Default=0)", default=0, type=int)
parser.add_argument("-v", "--video", help="Specify video file, if any, to perform pose estimation (Default=Webcam)", default='webcam', type=str)
parser.add_argument("-o", "--output_dir", help="Specify output directory. (Default=\{CURR_DIR\}/output/)", default="output", type=str)
parser.add_argument("-bs", "--batch_size", help="Specify batch size for processing. (Default=50)", default=50, type=int)
args = parser.parse_args()

# Save output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output/output.mp4', fourcc, 30, (432, 368))
    
try:    
    t1 = time.perf_counter()
    
    data = VideoLoader(args.video, batchSize=1, queueSize=0)
    data.start()
    detection = DetectionLoader(model, data, queueSize=0)
    detection.start()
    
    t2 = time.perf_counter()
    
    while True:
        frame = detection.getFrame()
        
        if frame is None:
            break
        
        ''' cv2.namedWindow("USB Camera", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("USB Camera", frame)
        out.write(frame)
        
        if cv2.waitKey(30)&0xFF == ord('q'):
            break '''
    
    elapsedTime = t2-t1
    print("Time elapsed: " + str(elapsedTime))
    

except:
    import traceback
    traceback.print_exc()

finally:
    cv2.destroyAllWindows()
    out.release()
    print("Finished")
    
    
class MultiStreamLoader:
    def __init__(self, model):
        self.model = model
        self.streams = []
        
    def loadStreams(self):
        pass
    
    