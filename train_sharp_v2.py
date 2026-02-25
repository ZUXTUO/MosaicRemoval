import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models

# 导入基础组件
from train import *

class ImprovedFrequencyLoss(nn.Module):
    """改进的频域损失 - 更准确的高频强调"""
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
        
        # 高频权重 - 使用高通滤波器
        h, w = pred_mag.shape[-2:]
        
        # 创建高通滤波器（中心低频权重低，边缘高频权重高）
        cy, cx = h // 2, w // 2
        y = torch.arange(h, device=pred.device).float().view(-1, 1)
        x = torch.arange(w, device=pred.device).float().view(1, -1)
        
        # 归一化距离
        dist = torch.sqrt(((y - cy) / h) ** 2 + ((x - cx) / w) ** 2)
        
        # 高通滤波器：距离越远权重越高
        high_pass_weight = torch.clamp(dist * 3, 0, 2)  # 0到2的权重
        high_pass_weight = high_pass_weight.unsqueeze(0).unsqueeze(0)
        
        # 加权损失
        weighted_pred = pred_mag * high_pass_weight
        weighted_target = target_mag * high_pass_weight
        
        return self.criterion(weighted_pred, weighted_target)

class SharpnessLoss(nn.Module):
    """锐度损失 - 直接惩罚模糊"""
    def __init__(self):
        super().__init__()
        # Laplacian 算子用于检测锐度
        laplacian_kernel = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.laplacian = nn.Parameter(laplacian_kernel.repeat(3, 1, 1, 1), requires_grad=False)
        self.criterion = nn.L1Loss()
    
    def forward(self, pred, target):
        # 计算Laplacian响应（边缘强度）
        pred_sharp = F.conv2d(pred, self.laplacian, padding=1, groups=3)
        target_sharp = F.conv2d(target, self.laplacian, padding=1, groups=3)
        
        # 锐度损失：让预测图像的边缘响应接近目标
        return self.criterion(torch.abs(pred_sharp), torch.abs(target_sharp))

class ImprovedDiscriminator(nn.Module):
    """改进的判别器 - 使用谱归一化防止过强"""
    def __init__(self, in_channels=3):
        super().__init__()
        
        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.utils.spectral_norm(
                nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)
            )]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))  # 使用InstanceNorm代替BatchNorm
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1)
        )
    
    def forward(self, img):
        return self.model(img)

class SharpTrainerV2:
    """改进的清晰度训练器"""
    def __init__(self, generator, discriminator, device):
        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        
        # 损失函数
        self.criterion_l1 = nn.L1Loss()
        self.criterion_perceptual = PerceptualLoss().to(device)
        self.criterion_frequency = ImprovedFrequencyLoss().to(device)
        self.criterion_sharpness = SharpnessLoss().to(device)
        self.criterion_edge = EdgeLoss().to(device)
    
    def compute_gradient_penalty(self, real_imgs, fake_imgs):
        """计算梯度惩罚（WGAN-GP）"""
        batch_size = real_imgs.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)
        
        interpolates = (alpha * real_imgs + (1 - alpha) * fake_imgs).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates)
        
        fake = torch.ones(d_interpolates.size(), device=self.device, requires_grad=False)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    def compute_generator_loss(self, outputs, orig_imgs, masks, fake_validity):
        """生成器损失 - 改进的权重配置"""
        mask_expanded = masks.repeat(1, 3, 1, 1)
        
        # 1. 对抗损失 - 必须与WGAN-GP匹配，使用 -E[D(fake)] 而不是 MSE
        loss_gan = -torch.mean(fake_validity)
        
        # 2. 像素损失（马赛克区域）
        loss_l1_masked = self.criterion_l1(outputs * mask_expanded, orig_imgs * mask_expanded)
        
        # 3. 感知损失
        loss_perceptual = self.criterion_perceptual(outputs, orig_imgs)
        
        # 4. 频域损失
        loss_frequency = self.criterion_frequency(outputs, orig_imgs)
        
        # 5. 锐度损失
        loss_sharpness = self.criterion_sharpness(outputs, orig_imgs)
        
        # 6. 边缘损失
        loss_edge = self.criterion_edge(outputs, orig_imgs)
        
        # 改进的权重配置
        total_loss = (
            loss_gan * 0.3 +              # 对抗损失，降低权重以防止破坏色彩
            loss_l1_masked * 2.0 +        # 像素损失，提高权重确保颜色正确
            loss_perceptual * 2.5 +       # 感知损失
            loss_frequency * 1.5 +        # 频域损失
            loss_sharpness * 2.0 +        # 锐度损失
            loss_edge * 1.5               # 边缘损失
        )
        
        return total_loss, {
            'gan': loss_gan.item(),
            'l1': loss_l1_masked.item(),
            'perceptual': loss_perceptual.item(),
            'frequency': loss_frequency.item(),
            'sharpness': loss_sharpness.item(),
            'edge': loss_edge.item()
        }
    
    def compute_discriminator_loss(self, real_imgs, fake_imgs):
        """判别器损失 - 使用WGAN-GP"""
        # 真实图像
        real_validity = self.discriminator(real_imgs)
        
        # 生成图像
        fake_validity = self.discriminator(fake_imgs.detach())
        
        # WGAN损失
        loss_real = -torch.mean(real_validity)
        loss_fake = torch.mean(fake_validity)
        
        # 梯度惩罚
        gradient_penalty = self.compute_gradient_penalty(real_imgs, fake_imgs)
        
        # 总损失
        d_loss = loss_real + loss_fake + 10.0 * gradient_penalty
        
        return d_loss

def train_sharp_v2(input_dir, output_dir, mask_dir, epochs=100, batch_size=4, 
                   lr_g=1e-4, lr_d=4e-5, final_model_path="final_model_dir/final_model_sharp_v2.pth"):
    """改进的清晰度训练"""
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
    discriminator = ImprovedDiscriminator().to(device)
    
    # 训练器
    trainer = SharpTrainerV2(generator, discriminator, device)
    
    # 优化器 - 判别器学习率更低
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
    
    # 学习率调度器
    scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_g, mode='min', factor=0.5, patience=10, min_lr=1e-7
    )
    scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_d, mode='min', factor=0.5, patience=10, min_lr=1e-7
    )
    
    print(f"开始改进版清晰度训练，共计 {len(dataset)} 张图片，计划训练 {epochs} 轮...")
    print("改进：WGAN-GP + 锐度损失 + 改进频域损失 + 平衡的判别器")
    
    best_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 25
    
    # 判别器训练次数（前期多训练判别器）
    n_critic = 5
    
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
            # 训练判别器（多次）
            # ==================
            for _ in range(n_critic if epoch < 10 else 1):  # 前10轮多训练判别器
                optimizer_d.zero_grad()
                
                # 生成假图像
                with torch.no_grad():
                    fake_imgs = generator(mosaiced_imgs, masks)
                
                # 不再仅截取马赛克区域，全图鉴别能防止网络只在黑框边界钻空子
                real_patches = orig_imgs
                fake_patches = fake_imgs
                
                # 判别器损失
                d_loss = trainer.compute_discriminator_loss(real_patches, fake_patches)
                d_loss.backward()
                optimizer_d.step()
            
            # ==================
            # 训练生成器
            # ==================
            optimizer_g.zero_grad()
            
            # 生成图像
            fake_imgs = generator(mosaiced_imgs, masks)
            fake_patches = fake_imgs  # 使用全图
            
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
                      f"G: {g_loss.item():.4f}, D: {d_loss.item():.4f} | "
                      f"Sharp: {loss_dict['sharpness']:.4f}", 
                      end="", flush=True)
        
        avg_g_loss = epoch_g_loss / len(dataloader)
        avg_d_loss = epoch_d_loss / len(dataloader)
        
        print(f"\n第 [{epoch+1}/{epochs}] 轮已完成")
        print(f"  生成器损失: {avg_g_loss:.4f}, 判别器损失: {avg_d_loss:.4f}")
        print(f"  学习率: G={optimizer_g.param_groups[0]['lr']:.6f}, D={optimizer_d.param_groups[0]['lr']:.6f}")
        
        # 调整学习率
        scheduler_g.step(avg_g_loss)
        scheduler_d.step(avg_d_loss)
        
        # 保存最佳模型
        if avg_g_loss < best_loss:
            best_loss = avg_g_loss
            patience_counter = 0
            temp_model_dir = os.path.join(os.path.dirname(os.path.abspath(final_model_path)), "..", "models_temp")
            os.makedirs(temp_model_dir, exist_ok=True)
            best_model_path = os.path.join(temp_model_dir, "best_model_v2.pth")
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
            epoch_model_path = os.path.join(temp_model_dir, f"model_v2_epoch_{epoch+1}.pth")
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
    
    # 改进版清晰度训练
    final_model_dir = os.path.join(base_dir, "models_fin")
    final_model_pth = os.path.join(final_model_dir, "mosaic_restoration_model_sharp_v2.pth")
    
    train_sharp_v2(
        input_dir=input_folder,
        output_dir=output_folder,
        mask_dir=mask_folder,
        epochs=100,
        batch_size=4,
        lr_g=1e-4,   # 生成器学习率
        lr_d=4e-5,   # 判别器学习率（更低）
        final_model_path=final_model_pth
    )
