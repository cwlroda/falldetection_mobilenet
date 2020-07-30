import os
import sys

from config.configparser import ConfigParser
from core.dataloader import DataLoader
from core.multistreamloader import MultiStreamLoader

configFile = "config/config.xml"

try:
    # load detection model
    model = DataLoader()
    
    # load configuration
    config = ConfigParser(configFile).getConfig()
    
    # load image writer
    
    
    # generate RTSP streams
    streams = MultiStreamLoader(model, config['RTSPAPI'])
    streams.loadStreams()
    
except:
    import traceback
    traceback.print_exc()   

finally:
    print("Finished")
    sys.exit(0)
        
        