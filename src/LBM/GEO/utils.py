"""
obselete utils for geometry generation
"""
import numpy as np
class Geo2D:
    @classmethod
    def circle(cls,cx,cy,r,x,y): 
        """
        返回(x,y)格点中包含(cx,cy)为圆心，r为半径的圆的有效面积和有效长度 用于2DLBM初始化
        """
        def D(p1,p2):
            return np.linalg.norm(np.array(p1)-np.array(p2))
        def Angle(p1,p2):
            v1 = np.array(p1)-np.array((cx,cy))
            v2 = np.array(p2)-np.array((cx,cy))
            cos_angle = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
            angle = np.arccos(np.clip(cos_angle,-1,1))
            return angle
        s_eff = 0.0
        l_eff = 0.0
        center = (cx,cy)
        point_list = [(x-0.5,y-0.5),(x+0.5,y-0.5),(x+0.5,y+0.5),(x-0.5,y+0.5)]
        d_list = [np.linalg.norm(np.array(p)-np.array(center)) for p in point_list]
        num_point_in = sum([1 if d<=r else 0 for d in d_list])
        index_alter = [0,1,2,3]
        def alter_index():
            for i in range(4):
                index_alter[i]+=1
                index_alter[i]%=4
        if num_point_in==0:# mode 0
            s_eff = 0.0
            l_eff = 0.0
        if num_point_in==4:# mode 4
            s_eff = 1.0
            l_eff = 0.0
        if num_point_in==1:# mode 1
            while(True): 
                if d_list[index_alter[0]]<=r:
                    # 求交点
                    ## 01
                    if point_list[index_alter[0]][0]==point_list[index_alter[1]][0]:# 垂直
                        x0 = point_list[index_alter[0]][0]
                        y0 = cy + (r**2-(x0-cx)**2)**0.5
                        if (y0-point_list[index_alter[0]][1])*(y0-point_list[index_alter[1]][1])>0:
                            y0 = cy - (r**2-(x0-cx)**2)**0.5
                    elif point_list[index_alter[0]][1]==point_list[index_alter[1]][1]:# 水平
                        y0 = point_list[index_alter[0]][1]
                        x0 = cx + (r**2-(y0-cy)**2)**0.5
                        if (x0-point_list[index_alter[0]][0])*(x0-point_list[index_alter[1]][0])>0:
                            x0 = cx - (r**2-(y0-cy)**2)**0.5
                    intersection1 = (x0,y0)
                    ## 03
                    if point_list[index_alter[0]][0]==point_list[index_alter[3]][0]:# 垂直
                        x0 = point_list[index_alter[0]][0]
                        y0 = cy + (r**2-(x0-cx)**2)**0.5
                        if (y0-point_list[index_alter[0]][1])*(y0-point_list[index_alter[3]][1])>0:
                            y0 = cy - (r**2-(x0-cx)**2)**0.5
                    elif point_list[index_alter[0]][1]==point_list[index_alter[3]][1]:# 水平
                        y0 = point_list[index_alter[0]][1]
                        x0 = cx + (r**2-(y0-cy)**2)**0.5
                        if (x0-point_list[index_alter[0]][0])*(x0-point_list[index_alter[3]][0])>0:
                            x0 = cx - (r**2-(y0-cy)**2)**0.5
                    intersection2 = (x0,y0)
                    # 求角度
                    angle = Angle(intersection1,intersection2)
                    # 求长度
                    l_eff = r*angle
                    # 求面积
                    v0 = np.array(point_list[index_alter[0]])-np.array(center)
                    v1 = np.array(intersection1)-np.array(center)
                    v2 = np.array(intersection2)-np.array(center)
                    s_eff = r**2*angle/2\
                                - np.abs(np.cross(v0,v1))/2\
                                - np.abs(np.cross(v0,v2))/2
                    break
                else:
                    alter_index()
                    continue
        if num_point_in==2:# mode 2
            while(True): 
                if d_list[index_alter[0]]<=r and d_list[index_alter[1]]<=r:
                    # 求交点
                    ## 03
                    if point_list[index_alter[0]][0]==point_list[index_alter[3]][0]:# 垂直
                        x0 = point_list[index_alter[0]][0]
                        y0 = cy + (r**2-(x0-cx)**2)**0.5
                        if (y0-point_list[index_alter[0]][1])*(y0-point_list[index_alter[3]][1])>0:
                            y0 = cy - (r**2-(x0-cx)**2)**0.5
                    elif point_list[index_alter[0]][1]==point_list[index_alter[3]][1]:# 水平
                        y0 = point_list[index_alter[0]][1]
                        x0 = cx + (r**2-(y0-cy)**2)**0.5
                        if (x0-point_list[index_alter[0]][0])*(y0-point_list[index_alter[3]][0])>0:
                            x0 = cx - (r**2-(y0-cy)**2)**0.5
                    intersection1 = (x0,y0)
                    ## 12
                    if point_list[index_alter[1]][0]==point_list[index_alter[2]][0]:# 垂直
                        x1 = point_list[index_alter[1]][0]
                        y1 = cy + (r**2-(x1-cx)**2)**0.5
                        if (y1-point_list[index_alter[1]][1])*(y1-point_list[index_alter[2]][1])>0:
                            y1 = cy - (r**2-(x1-cx)**2)**0.5
                    elif point_list[index_alter[1]][1]==point_list[index_alter[2]][1]:# 水平
                        y1 = point_list[index_alter[1]][1]
                        x1 = cx + (r**2-(y1-cy)**2)**0.5
                        if (x1-point_list[index_alter[1]][0])*(y1-point_list[index_alter[2]][0])>0:
                            x1 = cx - (r**2-(y1-cy)**2)**0.5
                    intersection2 = (x1,y1)
                    # 求角度
                    angle = Angle(intersection1,intersection2)
                    # 求长度
                    l_eff = r*angle
                    # 求面积
                    v0 = np.array(point_list[index_alter[0]])-np.array(center)
                    v0_ = np.array(point_list[index_alter[1]])-np.array(center)
                    v1 = np.array(intersection1)-np.array(center) # 0侧交点
                    v2 = np.array(intersection2)-np.array(center) # 0_侧交点
                    s_eff = r**2*angle/2\
                            - np.abs(np.cross(v2,v1))/2\
                                + D(v1+v2,v0+v0_)/2
                    break
                else: 
                    alter_index()
                    continue
        if num_point_in==3:# mode 3
                while(True): 
                    if d_list[index_alter[0]]<=r and d_list[index_alter[1]]<=r and d_list[index_alter[2]]<=r:
                        # 求交点
                        ## 03
                        if point_list[index_alter[0]][0]==point_list[index_alter[3]][0]:# 垂直
                            x0 = point_list[index_alter[0]][0]
                            y0 = cy + (r**2-(x0-cx)**2)**0.5
                            if (y0-point_list[index_alter[0]][1])*(y0-point_list[index_alter[3]][1])>0:
                                y0 = cy - (r**2-(x0-cx)**2)**0.5
                        elif point_list[index_alter[0]][1]==point_list[index_alter[3]][1]:# 水平
                            y0 = point_list[index_alter[0]][1]
                            x0 = cx + (r**2-(y0-cy)**2)**0.5
                            if (x0-point_list[index_alter[0]][0])*(x0-point_list[index_alter[3]][0])>0:
                                x0 = cx - (r**2-(y0-cy)**2)**0.5
                        intersection1 = (x0,y0)
                        ## 23
                        if point_list[index_alter[3]][0]==point_list[index_alter[2]][0]:# 垂直
                            x1 = point_list[index_alter[3]][0]
                            y1 = cy + (r**2-(x1-cx)**2)**0.5
                            if (y1-point_list[index_alter[3]][1])*(y1-point_list[index_alter[2]][1])>0:
                                y1 = cy - (r**2-(x1-cx)**2)**0.5
                        elif point_list[index_alter[3]][1]==point_list[index_alter[2]][1]:# 水平
                            y1 = point_list[index_alter[3]][1]
                            x1 = cx + (r**2-(y1-cy)**2)**0.5
                            if (x1-point_list[index_alter[3]][0])*(x1-point_list[index_alter[2]][0])>0:
                                x1 = cx - (r**2-(y1-cy)**2)**0.5
                        intersection2 = (x1,y1)
                        # 求角度
                        angle = Angle(intersection1,intersection2)\
                        # 求长度
                        l_eff = r*angle
                        # 求面积
                        v0 = np.array(point_list[index_alter[3]])-np.array(center)# 对侧
                        v1 = np.array(intersection1)-np.array(center) # 03侧交点
                        v2 = np.array(intersection2)-np.array(center) # 23侧交点
                        s_eff = 1+r**2*angle/2\
                                    - np.abs(np.cross(v0,v1))/2\
                                    - np.abs(np.cross(v0,v2))/2
                        break
                    else:
                        alter_index()
                        continue
        return s_eff,l_eff
class Geo3D:
    LINE_SAMPLE = 100
    CUBE_SAMPLE = 100
    @classmethod    
    def cube_intersection(cls,center,length,shape:str,*params):
        # 计算由shape（名字）定义的几何体，params为该几何体参数与以center为中心，length为边长的立方体的交集，相交体积和相交截面积
        # 使用Monte Carlo方法计算
        v = 0
        s = 0
        if shape.lower()=="sphere":
            sph_center = params[0]
            sph_radius = params[1]
            def sphere(point):
                d = np.linalg.norm(np.array(point)-np.array(sph_center))
                if d<=sph_radius:
                    return True
                else:
                    return False
            v,s = cls.cube_intersection_func(center,length,sphere)
        elif shape.lower() == "cylinder":
            cyl_center = params[0]
            cyl_up = params[1]
            cyl_radius = params[2]
            cyl_height = params[3]
            def cylinder(point):
                v = np.array(point)-np.array(cyl_center)
                up = np.array(cyl_up)/np.linalg.norm(np.array(cyl_up))
                z = np.dot(v,up)
                if z>=-cyl_height/2 and z<=cyl_height/2:
                    if np.linalg.norm(v)**2-z**2<=cyl_radius**2:
                        return True
                    else:
                        return False
                else:
                    return False      
            v,s = cls.cube_intersection_func(center,length,cylinder)     
        return v,s
    @classmethod
    def cube_intersection_func(cls,center,length,shape):
        # 计算由shape函数定义的几何体与以center为中心，length为边长的立方体的交集，相交体积和相交截面积
        # 使用Monte Carlo方法计算
        v = 0
        s = 0
        # 计算体积
        num_total = cls.CUBE_SAMPLE
        num_in = 0
        for _ in range(num_total):
            point = np.array(center)-length/2 + length*np.random.rand(3)
            if shape(point):
                num_in+=1
        v = (float(num_in)/num_total)*(length**3)
        # 计算面积
        return v,s
    @classmethod
    def line_intersection_func(cls,p1,p2,shape):
        intersected = False
        p_intersection = np.zeros(3)
        return intersected,p_intersection