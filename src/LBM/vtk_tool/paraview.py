import xml.etree.ElementTree as ET
import os

class PVDWriter:
    def __init__(self,path="output",name="default_name"):
        self.file_infos = []
        self.exportPath = path
        self.name = name
        self.version = 0.1
        self.byte_order = "LittleEndian"
    
    def addVTK(self,time,filename):
        self.file_infos.append((time,filename))
    def _createXML(self):
        """
        创建XML树结构
        
        Returns:
            根元素
        """
        # 创建根元素 VTKFile
        vtkfile = ET.Element("VTKFile")
        vtkfile.set("type", "Collection")
        vtkfile.set("version", str(self.version))
        vtkfile.set("byte_order", self.byte_order)
        
        # 创建Collection元素
        collection = ET.SubElement(vtkfile, "Collection")
        
        # 添加所有DataSet元素
        for timestep, file_path in self.file_infos:
            dataset = ET.SubElement(collection, "DataSet")
            dataset.set("timestep", str(timestep))
            dataset.set("group", "")
            dataset.set("part", "0")
            dataset.set("file", file_path)
            dataset.set("name", self.name)
        return vtkfile
    def writePVD(self): 
        root = self._createXML()
    
    # 生成XML字符串（包含声明）
    # 方法A：使用minidom添加声明
        from xml.dom import minidom
    
    # 先将ElementTree转换为字符串
        rough_string = ET.tostring(root, encoding='utf-8')
    
    # 用minidom解析并美化（会自动添加XML声明）
        reparsed = minidom.parseString(rough_string)
    
    # 生成美化后的XML（包含声明）
        pretty_xml = reparsed.toprettyxml(indent="    ", encoding="utf-8")
    
    # 解码为字符串
        xml_string = pretty_xml.decode("utf-8")
    
    # 写入文件
        with open(os.path.join(self.exportPath,self.name+".pvd"),"w",encoding="utf8") as f:
            f.write(xml_string)