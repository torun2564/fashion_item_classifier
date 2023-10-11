
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

classes_ja = ["Tシャツ/トップ", "ズボン", "プルオーバー", "ドレス", "コート", "サンダル", "ワイシャツ", "スニーカー", "バッグ", "アンクルブーツ"]
classes_en = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

n_class = len(classes_ja)
img_size = 28

# 画像認識モデル
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.sq1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), #畳み込み層（チャンネル数、フィルタ数、フィルタサイズ）、学習要素はフィルタの値
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2,2)
        )
        
        self.sq2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2,2)
        )
        
        self.sq3 = nn.Sequential(
            nn.Linear(256*7*7, 1024), #全結合層 (入力：フィルタ数256 x 画像サイズ4x4)
            nn.Dropout(p=0.2), #ドロップアウト（p=ドロップアウト率）
            nn.Linear(1024, 512),
            nn.Dropout(p=0.2),
            nn.Linear(512, 10),#出力層（10種の分類）
        )
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        x = self.sq1(x)
        x = self.sq2(x)
        x = self.sq3(self.flatten(x))
        return x
    
net = Net()

# 訓練済みパラメータの読み込みと設定
net.load_state_dict(torch.load("model_cnn.pth", map_location=torch.device("cpu")))

def predict(img):
    img = img.convert("L") # モノクロに変換
    img = img.resize((img_size, img_size)) # サイズ変換
    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.0),(1.0))
                                   ])
    img = transform(img)
    x = img.reshape(1, 1, img_size, img_size)
    
    # 予測
    net.eval()
    y = net(x)
    
    # 結果を返す
    y_pred = F.softmax(torch.squeeze(y))
    sorted_prob, sorted_indices = torch.sort(y_pred, descending=True) # 予測確率を降順にソート
    
    return [(classes_ja[idx], classes_en[idx], prob.item()) for idx, prob in zip(sorted_indices, sorted_prob)]
        
