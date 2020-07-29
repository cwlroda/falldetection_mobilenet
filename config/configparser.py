from xml.dom import minidom
import xmltodict

class ConfigParser:
    def __init__(self, configstr):
        self.configFile = configstr
        self.config = {}
        
    def getConfig(self):
        self.extractConfig("config.Source.RTSPAPI")
        self.extractConfig("config.Output.FileOutput")
        
        return self.config
    
    def getDictValue(self, dict, keyName):
        return dict[keyName]
    
    def extractConfig(self):
        configItemList = []
        doc = xmltodict.parse(open(self.configFile).read())
        
        pathItems = configItemName.split(".")
        refDict = doc
        
        for path in pathItems:
            refDict = self.getDictValue(refDict, path)
        
        if isinstance(refDict. list):
            configItemList = refDict
        else:
            configItemList.append(refDict)
            
        self.config[pathItems[-1]] = configItemList