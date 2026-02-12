from LBM.LBM2D import LBM2DSolver

# 纯流场
# 平面
# 斜面
# 小曲率凸面
# 大曲率凸面
# 小曲率凹面
# 大曲率凹面
# 颗粒
class OriginalDataGenerator:
    def __init__(self,solver:LBM2DSolver,path:str):
        self.solver = solver
        self.path = path
    def generate(self):
        pass
    

