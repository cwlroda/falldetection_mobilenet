import os
import sys

from config.configparser import ConfigParser
from core.dataloader import DataLoader
from core.multistreamloader import MultiStreamLoader
from core.imagewriter import ImageWriter

configFile = "config/config.xml"

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
        
        