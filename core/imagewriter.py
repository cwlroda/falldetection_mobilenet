import os
import cv2
import logging
from datetime import datetime

logger = logging.getLogger('debug')

class ImageWriter:
    def __init__(self, output, maindir):
        self.output_dict = output[0]
        self.output_dir = os.path.join(maindir, self.output_dict["Dir"])
        self.fileconv = self.output_dict["FileName"]
        self.filename = None
    
    def writeFrame(self, frames):
        for frame in frames:
            try:
                (img, self.ID, self.fallcount) = frame
                
                if img is not None:
                    self.filename = self.getFileName()
                    cv2.imwrite(self.filename, img)
                # logger.info("Frame written: " + filename)
            except:
                logger.error("Error writing frame: " + self.filename, exc_info=True)
    
    def getFileName(self):
        str_filename = "".join(self.fileconv)
        str_datetime = datetime.today().strftime('%Y%m%d_%H%M%S')
        str_filename = str_filename.replace("{yyyymmdd}_{HHMMSS}", str_datetime)
        str_filename = str_filename.replace("{streamID}", str(self.ID))
        str_filename = str_filename.replace("{algoName}", "MobileNetV2")
        str_filename = str_filename.replace("{fallcount}", str(self.fallcount))
        str_filename = os.path.join(self.output_dir, str_filename)
    
        return str_filename
    
    