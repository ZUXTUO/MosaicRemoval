import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 导入基础组件
from train import *

class FrequencyLoss(nn.Module):
    """频域损失 - 强调高频细节（边缘、纹理）"""
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()
    
    def forward(self, pred, target):
        # FFT变换到频域
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)
        
        # 计算幅度谱
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        # 高频权重（中心是低频，边缘是高频）
        h, w = pred_mag.shape[-2:]
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        y, x = y.to(pred.device), x.to(pred.device)
        
        # 距离中心越远，权重越高（强调高频）
        center_y, center_x = h // 2, w // 2
        dist = torch.sqrt((y - center_y)**2 + (x - center_x)**2)
        weight = 1.0 + dist / dist.max()  # 1.0 到 2.0 的权重
        
        # 加权损失
        loss = self.criterion(pred_mag * weight, target_mag * weight)
        return loss

class TextureLoss(nn.Module):
    """纹理损失 - 使用Gram矩阵保持纹理"""
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.features = nn.Sequential(*list(vgg[:16]))  # 到relu3_3
        
        for param in self.parameters():
            param.requires_grad = False
        
        self.criterion = nn.L1Loss()
    
    def gram_matrix(self, x):
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)
    
    def forward(self, pred, target):
        # 归一化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
        
        pred = (pred - mean) / std
        target = (target - mean) / std
        
        # 提取特征
        pred_features = self.features(pred)
        target_features = self.features(target)
        
        # Gram矩阵
        pred_gram = self.gram_matrix(pred_features)
        target_gram = self.gram_matrix(target_features)
        
        return self.criterion(pred_gram, target_gram)

class Discriminator(nn.Module):
    """判别器 - 区分真实图像和生成图像"""
    def __init__(self, in_channels=3):
        super().__init__()
        
        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1)  # 输出真假概率
        )
    
    def forward(self, img):
        return self.model(img)

class SharpTrainer:
    """清晰度优先训练器"""
    def __init__(self, generator, discriminator, device):
        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        
        # 损失函数
        self.criterion_l1 = nn.L1Loss()
        self.criterion_perceptual = PerceptualLoss().to(device)
        self.criterion_frequency = FrequencyLoss().to(device)
        self.criterion_texture = TextureLoss().to(device)
        self.criterion_gan = nn.BCEWithLogitsLoss()
    
    def compute_generator_loss(self, outputs, orig_imgs, masks, fake_validity):
        """生成器损失"""
        mask_expanded = masks.repeat(1, 3, 1, 1)
        
        # 1. 对抗损失 - 让判别器认为是真的
        real_labels = torch.ones_like(fake_validity)
        loss_gan = self.criterion_gan(fake_validity, real_labels)
        
        # 2. 像素损失（降低权重，避免模糊）
        loss_l1_masked = self.criterion_l1(outputs * mask_expanded, orig_imgs * mask_expanded)
        
        # 3. 感知损失（提高权重）
        loss_perceptual = self.criterion_perceptual(outputs * mask_expanded, orig_imgs * mask_expanded)
        
        # 4. 频域损失（新增 - 强调高频细节）
        loss_frequency = self.criterion_frequency(outputs * mask_expanded, orig_imgs * mask_expanded)
        
        # 5. 纹理损失（新增 - 保持纹理）
        loss_texture = self.criterion_texture(outputs * mask_expanded, orig_imgs * mask_expanded)
        
        # 组合损失 - 新的权重配置
        total_loss = (
            loss_gan * 0.5 +              # 对抗损失（关键）
            loss_l1_masked * 1.0 +        # 像素损失（降低）
            loss_perceptual * 3.0 +       # 感知损失（提高）
            loss_frequency * 2.0 +        # 频域损失（新增）
            loss_texture * 1.5            # 纹理损失（新增）
        )
        
        return total_loss, {
            'gan': loss_gan.item(),
            'l1': loss_l1_masked.item(),
            'perceptual': loss_perceptual.item(),
            'frequency': loss_frequency.item(),
            'texture': loss_texture.item()
        }
    
    def compute_discriminator_loss(self, real_imgs, fake_imgs):
        """判别器损失"""
        # 真实图像
        real_validity = self.discriminator(real_imgs)
        real_labels = torch.ones_like(real_validity)
        loss_real = self.criterion_gan(real_validity, real_labels)
        
        # 生成图像
        fake_validity = self.discriminator(fake_imgs.detach())
        fake_labels = torch.zeros_like(fake_validity)
        loss_fake = self.criterion_gan(fake_validity, fake_labels)
        
        return (loss_real + loss_fake) / 2

def train_sharp(input_dir, output_dir, mask_dir, epochs=100, batch_size=4, 
                lr_g=1e-4, lr_d=1e-4, final_model_path="final_model_dir/final_model.pth"):
    """清晰度优先训练"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"正在使用的训练设备: {device}")
    
    # 数据集
    dataset = MosaicRestorationDataset(input_dir, output_dir, mask_dir)
    if len(dataset) == 0:
        print("数据集为空，跳过训练过程。")
        return
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                          num_workers=0, drop_last=True, pin_memory=True)
    
    # 模型
    generator = InpaintingNetwork().to(device)
    discriminator = Discriminator().to(device)
    
    # 训练器
    trainer = SharpTrainer(generator, discriminator, device)
    
    # 优化器
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
    
    # 学习率调度器
    scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_g, mode='min', factor=0.5, patience=10, verbose=True, min_lr=1e-7
    )
    
    print(f"开始清晰度优先训练（GAN模式），共计 {len(dataset)} 张图片，计划训练 {epochs} 轮...")
    print("策略：使用判别器 + 高频损失 + 纹理损失，对抗模糊")
    
    best_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 20
    
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        
        for batch_idx, (mosaiced_imgs, orig_imgs, masks) in enumerate(dataloader):
            mosaiced_imgs = mosaiced_imgs.to(device)
            orig_imgs = orig_imgs.to(device)
            masks = masks.to(device)
            
            mask_expanded = masks.repeat(1, 3, 1, 1)
            
            # ==================
            # 训练判别器
            # ==================
            optimizer_d.zero_grad()
            
            # 生成假图像
            fake_imgs = generator(mosaiced_imgs, masks)
            
            # 只在马赛克区域判别
            real_patches = orig_imgs * mask_expanded
            fake_patches = fake_imgs * mask_expanded
            
            # 判别器损失
            d_loss = trainer.compute_discriminator_loss(real_patches, fake_patches)
            d_loss.backward()
            optimizer_d.step()
            
            # ==================
            # 训练生成器
            # ==================
            optimizer_g.zero_grad()
            
            # 重新生成（需要梯度）
            fake_imgs = generator(mosaiced_imgs, masks)
            fake_patches = fake_imgs * mask_expanded
            
            # 判别器对生成图像的判断
            fake_validity = discriminator(fake_patches)
            
            # 生成器损失
            g_loss, loss_dict = trainer.compute_generator_loss(
                fake_imgs, orig_imgs, masks, fake_validity
            )
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizer_g.step()
            
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            
            # 进度条
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
                progress = (batch_idx + 1) / len(dataloader)
                bar_length = 30
                filled_length = int(bar_length * progress)
                bar = '=' * filled_length + '-' * (bar_length - filled_length)
                print(f"\r  [第 {epoch+1} 轮] [{bar}] {progress*100:.1f}% | "
                      f"G_Loss: {g_loss.item():.4f}, D_Loss: {d_loss.item():.4f}", 
                      end="", flush=True)
        
        avg_g_loss = epoch_g_loss / len(dataloader)
        avg_d_loss = epoch_d_loss / len(dataloader)
        
        print(f"\n第 [{epoch+1}/{epochs}] 轮已完成")
        print(f"  生成器损失: {avg_g_loss:.4f}, 判别器损失: {avg_d_loss:.4f}")
        print(f"  学习率: G={optimizer_g.param_groups[0]['lr']:.6f}, D={optimizer_d.param_groups[0]['lr']:.6f}")
        
        # 调整学习率
        scheduler_g.step(avg_g_loss)
        
        # 保存最佳模型
        if avg_g_loss < best_loss:
            best_loss = avg_g_loss
            patience_counter = 0
            temp_model_dir = os.path.join(os.path.dirname(os.path.abspath(final_model_path)), "..", "models_temp")
            os.makedirs(temp_model_dir, exist_ok=True)
            best_model_path = os.path.join(temp_model_dir, "best_model.pth")
            torch.save(generator.state_dict(), best_model_path)
            print(f"  ✓ 保存最佳生成器模型 (损失: {best_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  ⚠ 损失未改善 ({patience_counter}/{early_stop_patience})")
            
            if patience_counter >= early_stop_patience:
                print(f"\n早停触发！连续 {early_stop_patience} 轮未改善，停止训练。")
                break
        
        # 每10轮保存检查点
        if (epoch + 1) % 10 == 0:
            temp_model_dir = os.path.join(os.path.dirname(os.path.abspath(final_model_path)), "..", "models_temp")
            os.makedirs(temp_model_dir, exist_ok=True)
            epoch_model_path = os.path.join(temp_model_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(generator.state_dict(), epoch_model_path)
            print(f"  ✓ 保存检查点: {epoch_model_path}")
    
    print("\n模型训练全部完成！")
    
    # 保存最终模型
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    torch.save(generator.state_dict(), final_model_path)
    print(f"最终模型已成功保存至: {final_model_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(base_dir, "image", "input")
    output_folder = os.path.join(base_dir, "image", "output")
    mask_folder = os.path.join(base_dir, "image", "masks")
    
    # 检查预处理数据
    if not os.path.exists(output_folder) or len(os.listdir(output_folder)) == 0:
        print("正在预处理数据...")
        has_data = process_data(input_folder, output_folder, mask_folder)
        if not has_data:
            print("请检查是否存在图片问题，然后重新运行。")
            sys.exit(1)
    else:
        print("检测到已有预处理数据，跳过预处理步骤。")
    
    # 清晰度优先训练
    final_model_dir = os.path.join(base_dir, "models_fin")
    final_model_pth = os.path.join(final_model_dir, "mosaic_restoration_model_sharp.pth")
    
    train_sharp(
        input_dir=input_folder,
        output_dir=output_folder,
        mask_dir=mask_folder,
        epochs=100,
        batch_size=4,
        lr_g=1e-4,  # 生成器学习率
        lr_d=1e-4,  # 判别器学习率
        final_model_path=final_model_pth
    )
