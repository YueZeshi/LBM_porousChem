import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1. 设置随机种子保证可重复性
torch.manual_seed(42)

# 2. 创建一个简单的神经网络模型
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 5)  # 输入10维，输出5维
        self.fc2 = nn.Linear(5, 1)   # 输出1维
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 3. 创建模型、损失函数和优化器
model = SimpleNet()
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降

# 4. 创建一些虚拟数据
# 100个样本，每个样本10个特征
x_train = torch.randn(100, 10)
# 目标值：简单的线性关系加上一些噪声
y_train = x_train.sum(dim=1, keepdim=True) * 0.5 + torch.randn(100, 1) * 0.1

# 创建数据集和数据加载器
dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# 5. 训练循环
print("开始训练...")
print(f"使用设备: {'cuda' if torch.cuda.is_available() else 'cpu'}")

for epoch in range(10):  # 训练10个epoch
    total_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        # 前向传播
        output = model(data)
        loss = criterion(output, target)
        
        # 反向传播
        optimizer.zero_grad()  # 清空梯度
        loss.backward()        # 计算梯度
        optimizer.step()       # 更新参数
        
        total_loss += loss.item()
    
    # 打印每个epoch的平均损失
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/10, 平均损失: {avg_loss:.4f}")

# 6. 测试模型
print("\n测试模型...")
model.eval()  # 设置为评估模式
with torch.no_grad():  # 不计算梯度
    test_input = torch.randn(5, 10)  # 5个测试样本
    predictions = model(test_input)
    print("测试输入形状:", test_input.shape)
    print("预测结果形状:", predictions.shape)
    print("前3个预测值:", predictions[:3].squeeze().numpy())

# 7. 查看模型参数
print("\n模型结构:")
print(model)

print("\n第一层权重形状:", model.fc1.weight.shape)
print("第一层偏置形状:", model.fc1.bias.shape)

# 8. 保存和加载模型
torch.save(model.state_dict(), 'simple_model.pth')
print("\n模型已保存到 simple_model.pth")

# 加载模型
loaded_model = SimpleNet()
loaded_model.load_state_dict(torch.load('simple_model.pth'))
print("模型加载成功!")