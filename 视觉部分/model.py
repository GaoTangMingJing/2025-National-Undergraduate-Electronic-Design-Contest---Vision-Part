import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import argparse
import cv2

# ---------- 模型结构 ----------
class StrongerCNN(nn.Module):
    def __init__(self):
        super(StrongerCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 12 * 12, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ---------- 图像预处理 ----------
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ---------- 加载模型 ----------
device = torch.device("cpu")  # 树莓派一般不带GPU
model = StrongerCNN().to(device)
model.load_state_dict(torch.load("digit_model_strong_aug_best.pth", map_location=device))
model.eval()

# ---------- 推理函数 ----------
def predict_image(img):
    # 直接使用OpenCV转换颜色空间（BGR转RGB）
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 转换为PIL图像格式并应用transform
    img = transform(Image.fromarray(img)).unsqueeze(0).to(device)  # [1, 3, 96, 96]
    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, dim=1).item()
    return pred

# ---------- 主函数 ----------
if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--image", type=str, required=True, help="路径：要识别的图像文件")
    # args = parser.parse_args()
    path = r"./num2.png"
    img = cv2.imread(path)
    for i in range(10):

        prediction = predict_image(img)
        print(f"预测结果: {prediction}")
