# CNN分类图像—CIFAR10数据集
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


def load_data():
    # 定义数据预处理：转换为Tensor并归一化到[0,1]
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # 加载训练集
    trainset = datasets.CIFAR10(
        root='data',  # 数据存储路径
        train=True,  # 加载训练集
        download=True,  # 如果不存在则下载
        transform=transform  # 应用数据转换
    )

    # 使用DataLoader一次性加载所有数据
    trainloader = DataLoader(trainset, batch_size=len(trainset), shuffle=False)
    xs, ys = next(iter(trainloader))

    # 将张量转换为列表以保持接口一致
    xs_list = [x for x in xs]  # 每个x的形状为[3,32,32]
    ys_list = ys.tolist()  # 标签转为列表

    return xs_list, ys_list


xs, ys = load_data()

# 输出验证
print(len(xs), len(ys))  # 50000, 50000
print(xs[0].shape)  # torch.Size([3, 32, 32])
print(ys[0])  # 标签值 (0-9)

#定义数据集
class Dataset(torch.utils.data.Dataset):

    def __len__(self):
        return len(xs)

    def __getitem__(self, i):
        return xs[i], ys[i]


dataset = Dataset()

x, y = dataset[0]

len(dataset), x.shape, y
#数据加载器
loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=8,
                                     shuffle=True,
                                     drop_last=True)

x, y = next(iter(loader))

len(loader), x.shape, y

#cnn神经网络
class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()

        #520的卷积层
        self.cnn1 = torch.nn.Conv2d(in_channels=3,
                                    out_channels=16,
                                    kernel_size=5,
                                    stride=2,
                                    padding=0)

        #311的卷积层
        self.cnn2 = torch.nn.Conv2d(in_channels=16,
                                    out_channels=32,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)

        #710的卷积层
        self.cnn3 = torch.nn.Conv2d(in_channels=32,
                                    out_channels=128,
                                    kernel_size=7,
                                    stride=1,
                                    padding=0)

        #池化层
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        #激活函数
        self.relu = torch.nn.ReLU()

        #线性输出层
        self.fc = torch.nn.Linear(in_features=128, out_features=10)

    def forward(self, x):

        #第一次卷积,形状变化可以推演
        #[8, 3, 32, 32] -> [8, 16, 14, 14]
        x = self.cnn1(x)
        x = self.relu(x)

        #第二次卷积,因为是311的卷积,所以尺寸不变
        #[8, 16, 14, 14] -> [8, 32, 14, 14]
        x = self.cnn2(x)
        x = self.relu(x)

        #池化,尺寸减半
        #[8, 32, 14, 14] -> [8, 32, 7, 7]
        x = self.pool(x)

        #第三次卷积,因为核心是7,所以只有一步计算
        #[8, 32, 7, 7] -> [8, 128, 1, 1]
        x = self.cnn3(x)
        x = self.relu(x)

        #展平,便于线性计算,也相当于把图像变成向量
        #[8, 128, 1, 1] -> [8, 128]
        x = x.flatten(start_dim=1)

        #线性计算输出
        #[8, 128] -> [8, 10]
        return self.fc(x)


model = Model()

model(torch.randn(8, 3, 32, 32)).shape

#训练
def train():
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fun = torch.nn.CrossEntropyLoss()
    model.train()

    for epoch in range(5):
        for i, (x, y) in enumerate(loader):
            out = model(x)
            loss = loss_fun(out, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % 2000 == 0:
                acc = (out.argmax(dim=1) == y).sum().item() / len(y)
                print(epoch, i, loss.item(), acc)

        torch.save(model, "D:\\Software\\PyCharm\\Model\\model3.model")


train()

#测试
@torch.no_grad()
def test():
    model = torch.load("D:\\Software\\PyCharm\\Model\\model3.model")
    model.eval()

    correct = 0
    total = 0
    for i in range(100):
        x, y = next(iter(loader))

        out = model(x).argmax(dim=1)

        correct += (out == y).sum().item()
        total += len(y)

    print(correct / total)


test()
