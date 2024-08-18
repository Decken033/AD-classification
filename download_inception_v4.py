import timm
import torch

# 创建 Inception v4 模型并下载预训练权重
model = timm.create_model('inception_v4', pretrained=True)

# 保存模型权重到本地文件
torch.save(model.state_dict(), 'inception_v4.pth')

print("Model downloaded and saved to 'inception_v4.pth'")
