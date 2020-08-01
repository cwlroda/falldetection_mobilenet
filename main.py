import os
import sys
import argparse

from config.configparser import ConfigParser
from core.dataloader import DataLoader
from core.multistreamloader import MultiStreamLoader
from core.imagewriter import ImageWriter

configFile = "config/config.xml"

parser = argparse.ArgumentParser()
# parser.add_argument("-m", "--mode", help="Mode 0: Image display with multithreading. Mode 1: Video display without multithreading. (Default=0)", default=0, type=int)
parser.add_argument("-v", "--video", help="Specify video file, if any, to perform pose estimation (Default=Webcam)", default='webcam', type=str)
parser.add_argument("-o", "--output_dir", help="Specify output directory. (Default=\{CURR_DIR\}/output/)", default="output", type=str)
args = parser.parse_args()

try:
    # load detection model
    model = DataLoader()
    
    # load configuration
    config = ConfigParser(configFile).getConfig()
    
    # load image writer
    imgwriter = ImageWriter(config['FileOutput'], os.path.dirname(__file__))
    
    # generate RTSP streams
    streams = MultiStreamLoader(model, config['RTSPAPI'])
    streams = streams.loadStreams()
    
    ID = 0
    
    for stream in streams:
        outframes = stream.getFrames()
        imgwriter.writeFrame(outframes, ID)
        ID += 1
    
except:
    import traceback
    traceback.print_exc()   

finally:
    print("Finished")
    sys.exit(0)
        
        