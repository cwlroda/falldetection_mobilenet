import os

from config.configparser import ConfigParser
from core.dataloader import DataLoader
from core.multistreamloader import MultistreamLoader

try:
    configFile = "config/config.xml"
    
    # load detection model
    model = DataLoader()
    
    # load configuration
    config = ConfigParser(configFile).getConfig()
    
    # load image writer
    
    # generate RTSP streams
    