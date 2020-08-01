import os
import cv2
from datetime import datetime

class ImageWriter:
    def __init__(self, output, maindir):
        self.output_dict = output[0]
        self.output_dir = os.path.join(maindir, self.output_dict["Dir"])
        self.fileconv = self.output_dict["FileName"]
    
    def writeFrame(self, frames, ID):
        for frame in frames:
            img, fallcount = frame
            filename = self.getFileName(fallcount, ID)
            cv2.imwrite(filename, img)
    
    def getFileName(self, fallcount, ID):
        str_filename = "".join(self.fileconv)
        str_datetime = datetime.today().strftime('%Y%m%d_%H%M%S')
        str_filename = str_filename.replace("{yyyymmdd}_{HHMMSS}", str_datetime)
        str_filename = str_filename.replace("{streamID}", str(ID))
        str_filename = str_filename.replace("{algoName}", "MobileNetV2")
        str_filename = str_filename.replace("{fallcount}", str(fallcount))
        str_filename = os.path.join(self.output_dir, str_filename)
    
        return str_filename
    
    