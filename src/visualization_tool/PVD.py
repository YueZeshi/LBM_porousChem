"""PVD 文件生成工具。

提供简洁的 API 将多个 VTK 时间步组织为一个 `.pvd` 集合文件，
便于 ParaView/PyVista 等工具按时间序列加载。
"""
import xml.etree.ElementTree as ET
import os

class PVDWriter:
    """PVD 写入器，将时间步与 VTK 文件路径写入 `name.pvd`。

    Parameters
    ----------
    path : str, default "output"
        导出目录。
    name : str, default "default_name"
        集合文件名（不含扩展名）。实际输出为 `path/name.pvd`。
    """
    def __init__(self,path="result",name="default_name"):
        self.file_infos = []
        self.exportPath = path
        self.name = name
        self.version = 0.1
        self.byte_order = "LittleEndian"
    
    def addVTK(self,time,filename):
        """登记一个时间步及其对应的 VTK 文件。

        Parameters
        ----------
        time : float
            物理时间（或任意时间标量），写入 `timestep` 属性。
        filename : str
            相对或绝对文件路径；写入 `file` 属性。通常为 `.vtr` 或 `.vts`。
        """
        self.file_infos.append((time,filename))
    def _createXML(self):
        """创建 PVD 的 XML 树结构并返回根元素。"""
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
        """生成并写入 `.pvd` 文件到 `exportPath`。

        行为：
        - 根据已登记的时间步/文件生成 `Collection/DataSet` 列表
        - 使用 `minidom` 美化并写入 `path/name.pvd`
        """
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