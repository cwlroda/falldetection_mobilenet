import sys
import os
import io
import time
import re
import argparse
import dataloader
import videoloader
import detectionloader

try:
    from armv7l.openvino.inference_engine import IECore, IEPlugin
except:
    from openvino.inference_engine import IECore, IEPlugin
    
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

try:
    model = dataloader.DataLoader()
    
    t1 = time.perf_counter()
    
    data = videoloader.VideoLoader(args.video, batchSize=50)
    data.start()
    detection = detectionloader.DetectionLoader(model, data)
    detection.start()
    
    t2 = time.perf_counter()
    elapsedTime = t2-t1
    print("Time elapsed: " + str(elapsedTime))

except:
    import traceback
    traceback.print_exc()

finally:
    print("Finished")