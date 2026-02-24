import os
import glob
import cv2
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import math
import numpy as np

def apply_mosaic(image, n_mosaics=1):
    """
    在一张图片上随机生成马赛克区域，并返回马赛克图和掩码。
    生成更真实、更多样化的马赛克效果
    """
    img = image.copy()
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if w < 20 or h < 20:
        return img, mask

    # 随机1-4个马赛克块
    n_mosaics = random.randint(1, 4)
    
    for _ in range(n_mosaics):
        # 马赛克区域：10% 到 40% 的图像尺寸（更多样化）
        size_w = random.randint(int(w * 0.10), int(w * 0.40))
        size_h = random.randint(int(h * 0.10), int(h * 0.40))
        
        # 确保坐标不会越界
        x = random.randint(0, max(0, w - size_w))
        y = random.randint(0, max(0, h - size_h))
        
        # 获取感兴趣区域 (ROI)
        roi = img[y:y+size_h, x:x+size_w]
        if roi.size == 0:
            continue
        
        # 随机选择马赛克类型
        mosaic_type = random.choice(['pixelate', 'blur', 'mixed'])
        
        if mosaic_type == 'pixelate':
            # 像素化马赛克：缩小到 5% - 15%（更精细）
            scale = random.uniform(0.05, 0.15)
            small_w = max(1, int(size_w * scale))
            small_h = max(1, int(size_h * scale))
            
            # 先缩小然后再用最近邻插值放大
            small_roi = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
            mosaic_roi = cv2.resize(small_roi, (size_w, size_h), interpolation=cv2.INTER_NEAREST)
            
        elif mosaic_type == 'blur':
            # 高斯模糊马赛克
            kernel_size = random.choice([15, 21, 31, 41, 51])
            mosaic_roi = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
            
        else:  # mixed
            # 混合效果：先像素化再轻微模糊
            scale = random.uniform(0.08, 0.18)
            small_w = max(1, int(size_w * scale))
            small_h = max(1, int(size_h * scale))
            small_roi = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
            mosaic_roi = cv2.resize(small_roi, (size_w, size_h), interpolation=cv2.INTER_NEAREST)
            # 添加轻微模糊
            kernel_size = random.choice([3, 5, 7])
            mosaic_roi = cv2.GaussianBlur(mosaic_roi, (kernel_size, kernel_size), 0)
        
        img[y:y+size_h, x:x+size_w] = mosaic_roi
        mask[y:y+size_h, x:x+size_w] = 1
        
    return img, mask
    return img, mask

def rename_input_images(input_dir):
    """
    标准化 input 文件夹中的文件名，防止因为空格、特殊字符或不同后缀导致的冲突。
    将所有图片重命名为 00001.jpg, 00002.png 等格式。
    """
    valid_exts = ('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG', '.bmp', '.BMP')
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]
    files.sort() # 排序确保顺序一致
    
    print(f"正在标准化 {len(files)} 个输入文件名...")
    
    # 建立一个临时重命名映射，防止在重命名过程中产生冲突
    temp_files = []
    for i, filename in enumerate(files):
        ext = os.path.splitext(filename)[1]
        new_name = f"training_img_{i+1:05d}{ext}"
        
        old_path = os.path.join(input_dir, filename)
        new_path = os.path.join(input_dir, new_name)
        
        # 如果新旧名字已经一样，跳过
        if filename == new_name:
            continue
            
        # 先重命名到一个绝对唯一的临时名字
        temp_name = f"temp_rename_{random.randint(10000, 99999)}_{i}{ext}"
        temp_path = os.path.join(input_dir, temp_name)
        os.rename(old_path, temp_path)
        temp_files.append((temp_path, new_path))
    
    # 从临时名字改为目标标准化名字
    for temp_path, new_path in temp_files:
        if os.path.exists(new_path):
            # 极罕见情况：如果目标路径已存在（可能是本次运行新产生的），则加个随机后缀
            base, ext = os.path.splitext(new_path)
            new_path = f"{base}_{random.randint(100, 999)}{ext}"
        os.rename(temp_path, new_path)
    
    print("文件名标准化完成。")

def process_data(input_dir, output_dir, mask_dir):
    """
    从input_dir读取原图，打上马赛克后保存到output_dir，掩码保存到mask_dir。
    """
    # 首先标准化输入文件名，解决特殊字符和重名潜在隐患
    rename_input_images(input_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    # 重新扫描标准化后的文件
    valid_exts = ('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG', '.bmp', '.BMP')
    image_paths = []
    for ext in valid_exts:
        image_paths.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
    
    if not image_paths:
        print(f"警告：在 {input_dir} 下没有找到有效的图片文件。")
        return False
        
    print(f"正在为 {len(image_paths)} 张图片各生成 8 张不同的马赛克副本...")
    for img_path in image_paths:
        base_name, ext = os.path.splitext(os.path.basename(img_path))
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {base_name}{ext}，已跳过。")
            continue
            
        # 针对当前文件生成 8 张不同马赛克位置和大小的变体图片
        for i in range(8):
            mosaic_img, mask = apply_mosaic(img, n_mosaics=1)
            
            out_path = os.path.join(output_dir, f"{base_name}_{i}{ext}")
            mask_path = os.path.join(mask_dir, f"{base_name}_{i}.png")
            
            cv2.imwrite(out_path, mosaic_img)
            cv2.imwrite(mask_path, mask * 255)  # 保存为0-255的图像
        
    print("数据预处理成功完成！所有 8 倍扩增的图片已准备就绪。")
    return True

class PerceptualLoss(nn.Module):
    """感知损失 - 使用VGG特征来保持语义信息"""
    def __init__(self):
        super().__init__()
        # 使用预训练的VGG16提取特征 - 修复警告
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*list(vgg[:4]))   # relu1_2
        self.slice2 = nn.Sequential(*list(vgg[4:9]))  # relu2_2
        self.slice3 = nn.Sequential(*list(vgg[9:16])) # relu3_3
        
        # 冻结VGG参数
        for param in self.parameters():
            param.requires_grad = False
        
        self.criterion = nn.L1Loss()
    
    def forward(self, pred, target):
        # 归一化到ImageNet的均值和标准差
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
        
        pred = (pred - mean) / std
        target = (target - mean) / std
        
        # 提取多层特征
        pred_f1 = self.slice1(pred)
        pred_f2 = self.slice2(pred_f1)
        pred_f3 = self.slice3(pred_f2)
        
        target_f1 = self.slice1(target)
        target_f2 = self.slice2(target_f1)
        target_f3 = self.slice3(target_f2)
        
        # 计算特征损失
        loss = self.criterion(pred_f1, target_f1) + \
               self.criterion(pred_f2, target_f2) + \
               self.criterion(pred_f3, target_f3)
        
        return loss

class MaskedLoss(nn.Module):
    """带掩码的损失函数 - 重点关注马赛克区域"""
    def __init__(self, base_criterion):
        super().__init__()
        self.base_criterion = base_criterion
    
    def forward(self, pred, target, mask):
        # 扩展掩码到3个通道
        if mask.size(1) == 1:
            mask = mask.repeat(1, 3, 1, 1)
        
        # 马赛克区域损失（权重更高）
        mosaic_loss = self.base_criterion(pred * mask, target * mask)
        
        # 非马赛克区域损失（保持一致性）
        non_mosaic_loss = self.base_criterion(pred * (1 - mask), target * (1 - mask))
        
        # 马赛克区域权重10倍
        return mosaic_loss * 10.0 + non_mosaic_loss * 1.0

class EdgeLoss(nn.Module):
    """边缘损失 - 强调边缘和细节恢复"""
    def __init__(self):
        super().__init__()
        # Sobel 算子用于边缘检测
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.sobel_x = nn.Parameter(sobel_x.repeat(3, 1, 1, 1), requires_grad=False)
        self.sobel_y = nn.Parameter(sobel_y.repeat(3, 1, 1, 1), requires_grad=False)
        self.criterion = nn.L1Loss()

    def forward(self, pred, target):
        # 计算梯度
        pred_grad_x = F.conv2d(pred, self.sobel_x, padding=1, groups=3)
        pred_grad_y = F.conv2d(pred, self.sobel_y, padding=1, groups=3)
        target_grad_x = F.conv2d(target, self.sobel_x, padding=1, groups=3)
        target_grad_y = F.conv2d(target, self.sobel_y, padding=1, groups=3)
        
        # 边缘损失
        loss_x = self.criterion(pred_grad_x, target_grad_x)
        loss_y = self.criterion(pred_grad_y, target_grad_y)
        
        return loss_x + loss_y

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        if img1.is_cuda:
            self.window = self.window.cuda(img1.device)
        self.window = self.window.type_as(img1)

        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size//2, groups=self.channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)

class GatedConv2d(nn.Module):
    """门控卷积 - 自动学习哪些特征应该通过"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, activation='lrelu'):
        super().__init__()
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.conv_mask = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.sigmoid = nn.Sigmoid()
        
        if activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            self.activation = None
    
    def forward(self, x):
        feature = self.conv_feature(x)
        if self.activation:
            feature = self.activation(feature)
        
        mask = self.conv_mask(x)
        mask = self.sigmoid(mask)
        
        # 门控机制：特征 * 门控掩码
        output = feature * mask
        return output

class GatedDeConv2d(nn.Module):
    """门控反卷积 - 用于上采样"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = GatedConv2d(in_channels, out_channels, kernel_size, stride, padding)
    
    def forward(self, x):
        # 使用双线性插值上采样
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv(x)
        return x

class ResidualBlock(nn.Module):
    """门控残差块"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = GatedConv2d(channels, channels, 3, 1, 1)
        self.conv2 = GatedConv2d(channels, channels, 3, 1, 1, activation=None)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        return out + residual

class DenseBlock(nn.Module):
    """密集块 - 用于特征提取"""
    def __init__(self, channels, growth_rate=32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, growth_rate, 3, padding=1)
        self.conv2 = nn.Conv2d(channels + growth_rate, growth_rate, 3, padding=1)
        self.conv3 = nn.Conv2d(channels + 2 * growth_rate, growth_rate, 3, padding=1)
        self.conv4 = nn.Conv2d(channels + 3 * growth_rate, growth_rate, 3, padding=1)
        self.conv5 = nn.Conv2d(channels + 4 * growth_rate, channels, 3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    """Residual in Residual Dense Block"""
    def __init__(self, channels, growth_rate=32):
        super().__init__()
        self.rdb1 = DenseBlock(channels, growth_rate)
        self.rdb2 = DenseBlock(channels, growth_rate)
        self.rdb3 = DenseBlock(channels, growth_rate)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class ResidualBlock(nn.Module):
    """轻量级残差块"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
    
    def forward(self, x):
        return x + self.conv(x) * 0.2

class AttentionBlock(nn.Module):
    """注意力模块 - 帮助模型关注重要特征"""
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch, channels, height, width = x.size()
        
        # 计算注意力
        query = self.query(x).view(batch, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch, -1, height * width)
        attention = F.softmax(torch.bmm(query, key), dim=-1)
        
        value = self.value(x).view(batch, -1, height * width)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch, channels, height, width)
        
        return self.gamma * out + x

class DoubleConv(nn.Module):
    """双卷积块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class InpaintingNetwork(nn.Module):
    """
    改进的U-Net架构，加入注意力机制
    """
    def __init__(self, in_channels=4):  # 3 (RGB) + 1 (mask)
        super().__init__()
        
        # 编码器
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # 瓶颈层 + 注意力
        self.bottleneck = DoubleConv(512, 1024)
        self.attention = AttentionBlock(1024)
        
        # 解码器
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)  # 1024 = 512 (from up4) + 512 (from enc4)
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)  # 512 = 256 (from up3) + 256 (from enc3)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)  # 256 = 128 (from up2) + 128 (from enc2)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)   # 128 = 64 (from up1) + 64 (from enc1)
        
        # 输出层
        self.output = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img, mask):
        # 拼接图像和掩码作为输入
        x = torch.cat([img, mask], dim=1)
        
        # 编码路径
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        
        # 瓶颈 + 注意力
        b = self.bottleneck(p4)
        b = self.attention(b)
        
        # 解码路径（带跳跃连接）
        u4 = self.up4(b)
        d4 = self.dec4(torch.cat([u4, e4], dim=1))
        
        u3 = self.up3(d4)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        
        # 输出
        out = self.output(d1)
        
        # 关键：只在马赛克区域替换，其他区域完全保持原样
        result = img * (1 - mask) + out * mask
        
        return result

# ----------------- 数据集 & 训练 -----------------

class MosaicRestorationDataset(Dataset):
    def __init__(self, input_dir, output_dir, mask_dir, img_size=(256, 256)):
        self.input_dir = input_dir    # 原图文件夹路径
        self.output_dir = output_dir  # 生成的马赛克图文件夹路径
        self.mask_dir = mask_dir      # 掩码文件夹路径
        
        # 寻找扩展名匹配的所有文件
        valid_ext = ('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG', '.bmp', '.BMP')
        self.mosaiced_names = [f for f in os.listdir(self.output_dir) if f.endswith(valid_ext)]
        self.img_size = img_size
        
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(), # 自动将值归一化到 [0, 1]
        ])

    def __len__(self):
        return len(self.mosaiced_names)

    def __getitem__(self, idx):
        mosaiced_name = self.mosaiced_names[idx]
        
        # 根据 output 的名字解析原图名字。例如: training_img_00001_0.jpg -> training_img_00001.jpg
        base_name, ext = os.path.splitext(mosaiced_name)
        orig_name = base_name.rsplit('_', 1)[0] + ext
        mask_name = base_name + '.png'
        
        orig_img_path = os.path.join(self.input_dir, orig_name)
        mosaiced_img_path = os.path.join(self.output_dir, mosaiced_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        orig_img = Image.open(orig_img_path).convert('RGB')
        mosaiced_img = Image.open(mosaiced_img_path).convert('RGB')
        mask_img = Image.open(mask_path).convert('L')  # 灰度图
        
        orig_tensor = self.transform(orig_img)
        mosaiced_tensor = self.transform(mosaiced_img)
        
        # 掩码转换
        mask_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
        ])
        mask_tensor = mask_transform(mask_img)
        
        # 确保掩码是二值的 (0 或 1)
        mask_tensor = (mask_tensor > 0.5).float()
        
        return mosaiced_tensor, orig_tensor, mask_tensor

def train_model(input_dir, output_dir, mask_dir, epochs=50, batch_size=4, lr=2e-4, final_model_path="final_model_dir/final_model.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"正在使用的训练设备: {device}")
    
    dataset = MosaicRestorationDataset(input_dir, output_dir, mask_dir)
    if len(dataset) == 0:
        print("数据集为空，跳过训练过程。")
        return
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    
    model = InpaintingNetwork().to(device)
    
    # 损失函数
    criterion_l1 = nn.L1Loss()
    criterion_perceptual = PerceptualLoss().to(device)
    
    # 使用 AdamW 优化器（更稳定）
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
    
    # 学习率调度器 - 使用余弦退火
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    print(f"开始训练，共计 {len(dataset)} 张图片，计划训练 {epochs} 轮...")
    
    # 检查第一个样本
    sample_mosaic, sample_orig, sample_mask = dataset[0]
    print(f"样本检查 - 马赛克图形状: {sample_mosaic.shape}, 掩码形状: {sample_mask.shape}")
    print(f"掩码覆盖率: {sample_mask.mean().item()*100:.2f}%")
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (mosaiced_imgs, orig_imgs, masks) in enumerate(dataloader):
            # 将数据传入指定的计算设备
            mosaiced_imgs = mosaiced_imgs.to(device)
            orig_imgs = orig_imgs.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(mosaiced_imgs, masks)
            
            # 检查输出是否有 NaN
            if torch.isnan(outputs).any():
                print(f"警告：检测到 NaN，跳过此批次")
                continue
            
            # 优化的损失函数 - 平衡全局和局部细节
            # 1. 马赛克区域的L1损失（主要）
            mask_expanded = masks.repeat(1, 3, 1, 1)
            loss_masked_l1 = criterion_l1(outputs * mask_expanded, orig_imgs * mask_expanded)
            
            # 2. 全图L1损失（保持整体一致性）
            loss_l1 = criterion_l1(outputs, orig_imgs)
            
            # 3. 感知损失（保持纹理和语义，权重提高）
            loss_perceptual = criterion_perceptual(outputs, orig_imgs)
            
            # 总损失 - 重点关注马赛克区域，同时保持感知质量
            loss = loss_masked_l1 * 10.0 + loss_l1 * 1.0 + loss_perceptual * 2.0
            
            # 梯度裁剪防止爆炸
            if loss.item() > 100:
                print(f"警告：损失过大 {loss.item():.2f}，跳过此批次")
                continue
            
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
                progress = (batch_idx + 1) / len(dataloader)
                bar_length = 30
                filled_length = int(bar_length * progress)
                bar = '=' * filled_length + '-' * (bar_length - filled_length)
                print(f"\r  [第 {epoch+1} 轮进展] [{bar}] {progress*100:.1f}% | Loss: {loss.item():.4f}", end="", flush=True)
                
        avg_loss = epoch_loss / len(dataloader)
        print(f"\n第 [{epoch+1}/{epochs}] 轮已完成, 平均损失: {avg_loss:.4f}, 学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        scheduler.step()
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            temp_model_dir = os.path.join(os.path.dirname(os.path.abspath(final_model_path)), "..", "models_temp")
            os.makedirs(temp_model_dir, exist_ok=True)
            best_model_path = os.path.join(temp_model_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ 保存最佳模型 (损失: {best_loss:.4f})")
        
        # 每10轮保存一次检查点
        if (epoch + 1) % 10 == 0:
            temp_model_dir = os.path.join(os.path.dirname(os.path.abspath(final_model_path)), "..", "models_temp")
            os.makedirs(temp_model_dir, exist_ok=True)
            epoch_model_path = os.path.join(temp_model_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), epoch_model_path)
            print(f"  ✓ 保存检查点: {epoch_model_path}")
        
    print("模型训练全部完成！")
    
    # 将最终的模型结果保存到另外的位置
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    torch.save(model.state_dict(), final_model_path)
    print(f"最终模型已成功保存至指定位置: {final_model_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(base_dir, "image", "input")
    output_folder = os.path.join(base_dir, "image", "output")
    mask_folder = os.path.join(base_dir, "image", "masks")
    
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(mask_folder, exist_ok=True)
    
    # 1. 预处理数据：将 input_folder 里的图片添加随机马赛克，输出到 output_folder
    has_data = process_data(input_folder, output_folder, mask_folder)
    
    # 2. 如果存在数据，开始训练大模型
    if has_data:
        # 指定最终模型保存位置为 models_fin
        final_model_dir = os.path.join(base_dir, "models_fin")
        final_model_pth = os.path.join(final_model_dir, "mosaic_restoration_model_final.pth")
        
        train_model(input_dir=input_folder, 
                    output_dir=output_folder, 
                    mask_dir=mask_folder,
                    epochs=120,  # 增加训练轮数
                    batch_size=4,
                    lr=2e-4,
                    final_model_path=final_model_pth)
    else:
        print("请检查是否存在图片问题，然后重新运行。")
