import os
import sys
import logging
import argparse

from config.configparser import ConfigParser
from core.dataloader import DataLoader
from core.multistreamloader import MultiStreamLoader
from core.imagewriter import ImageWriter
from core.logger import Logger
from core.detectionloader import DetectionLoader

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

configFile = "config/config.xml"

parser = argparse.ArgumentParser()
# parser.add_argument("-m", "--mode", help="Mode 0: Image display with multithreading. Mode 1: Video display without multithreading. (Default=0)", default=0, type=int)
parser.add_argument("-v", "--video", help="Specify video file, if any, to perform pose estimation (Default=Webcam)", default='webcam', type=str)
parser.add_argument("-o", "--output_dir", help="Specify output directory. (Default=\{CURR_DIR\}/output/)", default="output", type=str)
args = parser.parse_args()

try:
    logger = Logger('debug').setup()
    
    # load detection model
    model = DataLoader()
    
    # load configuration
    config = ConfigParser(configFile).getConfig()
    logger.info("Configuration loaded")
    
    # load image writer
    imgwriter = ImageWriter(config['FileOutput'], os.path.dirname(__file__))
    logger.info("Image writer loaded")
    
    # generate RTSP streams
    streams = MultiStreamLoader(model, config['RTSPAPI'])
    streams.generateStreams()
    
    inference = DetectionLoader(model, streams)
    inference.loadDetectors()
    
    while True:
        outframes = inference.getFrames()
        imgwriter.writeFrame(outframes)
        
        outframes.clear
        del outframes[:]
    
    logging.info("Frames successfully written to file")
    
except:
    import traceback
    traceback.print_exc()   

finally:
    logging.info("Finished")
    sys.exit(0)
        
        