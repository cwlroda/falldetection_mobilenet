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

class DataLoader:
    def __init__(self):
        model_xml = "models/train/test/openvino/mobilenet_v2_0.5_224/FP32/frozen-model.xml"
            
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        net = IECore().read_network(model=model_xml, weights=model_bin)
        self.input_blob = next(iter(net.inputs))
        self.exec_net = IECore().load_network(network=net, device_name="CPU")
        inputs = net.inputs["image"]

        self.h = inputs.shape[2] #368
        self.w = inputs.shape[3] #432

    def geth(self):
        return self.h
    
    def getw(self):
        return self.w
    
    def get_execnet(self):
        return self.exec_net
    
    def get_inputblob(self):
        return self.input_blob

