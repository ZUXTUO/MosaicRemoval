import os
import io
import torch
import torch.nn.functional as F
from flask import Flask, request, render_template_string, send_file, jsonify
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import numpy as np
import base64
import json

# 需要导入与训练时结构完全相同的模型定义
from train import InpaintingNetwork

app = Flask(__name__)

# 模型文件路径 - 使用最佳模型
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_temp", "best_model_v2.pth")

# 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化模型全局变量
model = None

def load_model():
    """加载之前训练好的 InpaintingNetwork 模型权重"""
    global model
    if model is None:
        model = InpaintingNetwork().to(device)
        model.eval() # 设置为评估模式
        
        if os.path.exists(MODEL_PATH):
            print(f"正在加载模型权重: {MODEL_PATH}")
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
            print("模型加载成功！")
        else:
            print(f"警告：找不到模型文件 {MODEL_PATH}。请先运行 train.py 训练模型。")

# 图像预处理和后处理
transform_input = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

transform_output = transforms.ToPILImage()

# ================= HTML 模板 =================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Image Restoration System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
            background: #f5f7fa;
            color: #2c3e50;
            min-height: 100vh;
            padding: 0;
        }
        .header {
            background: #ffffff;
            border-bottom: 1px solid #e1e8ed;
            padding: 1.5rem 2rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        }
        .header h1 {
            font-size: 1.5rem;
            font-weight: 600;
            color: #2c3e50;
            letter-spacing: -0.5px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }
        .card {
            background: white;
            border-radius: 8px;
            padding: 2rem;
            margin-bottom: 1.5rem;
            border: 1px solid #e1e8ed;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }
        .upload-section {
            text-align: center;
            padding: 3rem 2rem;
            border: 2px dashed #cbd5e0;
            border-radius: 8px;
            background: #f8fafc;
            transition: all 0.2s;
        }
        .upload-section:hover {
            border-color: #4a90e2;
            background: #f0f7ff;
        }
        .btn {
            background: #4a90e2;
            color: white;
            padding: 10px 24px;
            border: none;
            border-radius: 4px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            margin: 0 8px;
            display: inline-block;
        }
        .btn:hover {
            background: #357abd;
        }
        .btn:disabled {
            background: #cbd5e0;
            cursor: not-allowed;
        }
        .btn-secondary {
            background: #718096;
        }
        .btn-secondary:hover {
            background: #4a5568;
        }
        .canvas-container {
            display: none;
            margin: 2rem 0;
        }
        .canvas-wrapper {
            text-align: center;
            background: #f8fafc;
            padding: 2rem;
            border-radius: 8px;
            border: 1px solid #e1e8ed;
        }
        #canvas {
            border: 1px solid #cbd5e0;
            border-radius: 4px;
            cursor: crosshair;
            max-width: 100%;
            background: white;
        }
        .controls {
            margin-top: 1.5rem;
            padding-top: 1.5rem;
            border-top: 1px solid #e1e8ed;
        }
        .instructions {
            background: #fffbeb;
            border: 1px solid #fbbf24;
            border-radius: 6px;
            padding: 1.25rem;
            margin-bottom: 1.5rem;
            display: none;
        }
        .instructions h3 {
            font-size: 0.95rem;
            font-weight: 600;
            color: #92400e;
            margin-bottom: 0.75rem;
        }
        .instructions ul {
            margin-left: 1.5rem;
            color: #78350f;
            font-size: 0.9rem;
            line-height: 1.8;
        }
        .result-section {
            display: none;
        }
        .section-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 1.5rem;
            padding-bottom: 0.75rem;
            border-bottom: 2px solid #e1e8ed;
        }
        .image-compare {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin: 2rem 0;
        }
        .image-box {
            background: #f8fafc;
            padding: 1.5rem;
            border-radius: 8px;
            border: 1px solid #e1e8ed;
        }
        .image-box h3 {
            font-size: 0.95rem;
            font-weight: 600;
            color: #4a5568;
            margin-bottom: 1rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .image-box img {
            max-width: 100%;
            border-radius: 4px;
            border: 1px solid #cbd5e0;
            display: block;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 3rem;
        }
        .spinner {
            border: 3px solid #e1e8ed;
            border-top: 3px solid #4a90e2;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 0.8s linear infinite;
            margin: 0 auto 1rem;
        }
        .loading-text {
            color: #718096;
            font-size: 0.9rem;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .upload-hint {
            margin-top: 1rem;
            color: #718096;
            font-size: 0.85rem;
        }
        @media (max-width: 768px) {
            .container { padding: 1rem; }
            .card { padding: 1.5rem; }
            .image-compare { grid-template-columns: 1fr; gap: 1.5rem; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>AI Image Restoration System</h1>
    </div>

    <div class="container">
        <div class="card">
            <div class="upload-section">
                <input type="file" id="fileInput" accept="image/*" style="display:none">
                <button class="btn" onclick="document.getElementById('fileInput').click()">
                    Select Image
                </button>
                <p class="upload-hint">Supported formats: JPG, PNG, BMP</p>
            </div>
        </div>

        <div class="instructions" id="instructions">
            <h3>Instructions</h3>
            <ul>
                <li>Click and drag on the image to select mosaic regions</li>
                <li>Multiple regions can be selected</li>
                <li>Click "Clear Selection" to reset</li>
                <li>Click "Process" when ready</li>
            </ul>
        </div>

        <div class="canvas-container" id="canvasContainer">
            <div class="card">
                <div class="canvas-wrapper">
                    <canvas id="canvas"></canvas>
                </div>
                <div class="controls">
                    <button class="btn btn-secondary" onclick="clearSelection()">Clear Selection</button>
                    <button class="btn" onclick="processImage()" id="processBtn">Process Image</button>
                </div>
            </div>
        </div>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p class="loading-text">Processing image, please wait...</p>
        </div>

        <div class="result-section" id="resultSection">
            <div class="card">
                <h2 class="section-title">Results</h2>
                <div class="image-compare">
                    <div class="image-box">
                        <h3>Original</h3>
                        <img id="origImage" src="" alt="Original">
                    </div>
                    <div class="image-box">
                        <h3>Restored</h3>
                        <img id="resultImage" src="" alt="Restored">
                    </div>
                </div>
                <div style="text-align: center;">
                    <a id="downloadLink" href="#" download="restored_image.jpg" class="btn">
                        Download Result
                    </a>
                </div>
            </div>
        </div>
    </div>

    <script>
        let canvas, ctx, img, isDrawing = false;
        let startX, startY;
        let rectangles = [];

        document.getElementById('fileInput').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = function(event) {
                img = new Image();
                img.onload = function() {
                    setupCanvas();
                    document.getElementById('instructions').style.display = 'block';
                    document.getElementById('canvasContainer').style.display = 'block';
                    document.getElementById('resultSection').style.display = 'none';
                };
                img.src = event.target.result;
            };
            reader.readAsDataURL(file);
        });

        function setupCanvas() {
            canvas = document.getElementById('canvas');
            ctx = canvas.getContext('2d');
            
            // 设置canvas尺寸
            const maxWidth = 800;
            const scale = Math.min(1, maxWidth / img.width);
            canvas.width = img.width * scale;
            canvas.height = img.height * scale;
            
            drawImage();
            
            canvas.addEventListener('mousedown', startDrawing);
            canvas.addEventListener('mousemove', draw);
            canvas.addEventListener('mouseup', stopDrawing);
            canvas.addEventListener('mouseout', stopDrawing);
        }

        function drawImage() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            
            // 绘制所有矩形
            ctx.strokeStyle = 'red';
            ctx.lineWidth = 3;
            ctx.fillStyle = 'rgba(255, 0, 0, 0.2)';
            rectangles.forEach(rect => {
                ctx.fillRect(rect.x, rect.y, rect.width, rect.height);
                ctx.strokeRect(rect.x, rect.y, rect.width, rect.height);
            });
        }

        function startDrawing(e) {
            isDrawing = true;
            const rect = canvas.getBoundingClientRect();
            startX = e.clientX - rect.left;
            startY = e.clientY - rect.top;
        }

        function draw(e) {
            if (!isDrawing) return;
            
            const rect = canvas.getBoundingClientRect();
            const currentX = e.clientX - rect.left;
            const currentY = e.clientY - rect.top;
            
            drawImage();
            
            // 绘制当前正在画的矩形
            const width = currentX - startX;
            const height = currentY - startY;
            ctx.strokeStyle = 'red';
            ctx.lineWidth = 3;
            ctx.fillStyle = 'rgba(255, 0, 0, 0.2)';
            ctx.fillRect(startX, startY, width, height);
            ctx.strokeRect(startX, startY, width, height);
        }

        function stopDrawing(e) {
            if (!isDrawing) return;
            isDrawing = false;
            
            const rect = canvas.getBoundingClientRect();
            const endX = e.clientX - rect.left;
            const endY = e.clientY - rect.top;
            
            const width = endX - startX;
            const height = endY - startY;
            
            // 只保存有效的矩形（面积大于100像素）
            if (Math.abs(width) > 10 && Math.abs(height) > 10) {
                rectangles.push({
                    x: Math.min(startX, endX),
                    y: Math.min(startY, endY),
                    width: Math.abs(width),
                    height: Math.abs(height)
                });
                drawImage();
            }
        }

        function clearSelection() {
            rectangles = [];
            drawImage();
        }

        async function processImage() {
            if (rectangles.length === 0) {
                alert('请先框选马赛克区域！');
                return;
            }

            document.getElementById('loading').style.display = 'block';
            document.getElementById('processBtn').disabled = true;

            try {
                // 创建掩码
                const maskCanvas = document.createElement('canvas');
                maskCanvas.width = img.width;
                maskCanvas.height = img.height;
                const maskCtx = maskCanvas.getContext('2d');
                
                // 填充黑色背景
                maskCtx.fillStyle = 'black';
                maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
                
                // 在选中的区域填充白色
                maskCtx.fillStyle = 'white';
                const scaleX = img.width / canvas.width;
                const scaleY = img.height / canvas.height;
                
                rectangles.forEach(rect => {
                    maskCtx.fillRect(
                        rect.x * scaleX,
                        rect.y * scaleY,
                        rect.width * scaleX,
                        rect.height * scaleY
                    );
                });

                // 将原图和掩码转换为blob
                const imgBlob = await fetch(img.src).then(r => r.blob());
                const maskBlob = await new Promise(resolve => maskCanvas.toBlob(resolve, 'image/png'));

                // 创建FormData
                const formData = new FormData();
                formData.append('image', imgBlob, 'image.jpg');
                formData.append('mask', maskBlob, 'mask.png');

                // 发送到服务器
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    const blob = await response.blob();
                    const resultUrl = URL.createObjectURL(blob);
                    
                    document.getElementById('origImage').src = img.src;
                    document.getElementById('resultImage').src = resultUrl;
                    document.getElementById('downloadLink').href = resultUrl;
                    document.getElementById('resultSection').style.display = 'block';
                } else {
                    alert('处理失败: ' + await response.text());
                }
            } catch (err) {
                alert('处理出错: ' + err.message);
            } finally {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('processBtn').disabled = false;
            }
        }
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    load_model()
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "模型未加载，请确认模型文件存在。", 500
        
    if 'image' not in request.files or 'mask' not in request.files:
        return "缺少图片或掩码文件。", 400

    try:
        # 读取图片和掩码
        img_file = request.files['image']
        mask_file = request.files['mask']
        
        img = Image.open(img_file.stream).convert('RGB')
        mask = Image.open(mask_file.stream).convert('L')
        
        # 转换为numpy数组
        orig_img_array = np.array(img)
        orig_mask_array = np.array(mask)
        
        # 找到掩码的边界框
        mask_coords = np.where(orig_mask_array > 127)
        if len(mask_coords[0]) == 0:
            # 没有选中区域，直接返回原图
            img_io = io.BytesIO()
            img.save(img_io, 'JPEG', quality=95)
            img_io.seek(0)
            return send_file(img_io, mimetype='image/jpeg')
        
        y_min, y_max = mask_coords[0].min(), mask_coords[0].max()
        x_min, x_max = mask_coords[1].min(), mask_coords[1].max()
        
        # 扩展边界框一点，增加上下文信息
        padding = 32
        y_min = max(0, y_min - padding)
        y_max = min(orig_img_array.shape[0], y_max + padding)
        x_min = max(0, x_min - padding)
        x_max = min(orig_img_array.shape[1], x_max + padding)
        
        # 裁剪出包含马赛克的区域
        crop_img = orig_img_array[y_min:y_max, x_min:x_max]
        crop_mask = orig_mask_array[y_min:y_max, x_min:x_max]
        
        # 转换为PIL图像
        crop_img_pil = Image.fromarray(crop_img)
        crop_mask_pil = Image.fromarray(crop_mask)
        
        # 预处理 - resize到256x256用于模型推理
        input_tensor = transform_input(crop_img_pil).unsqueeze(0).to(device)
        mask_tensor = transform_input(crop_mask_pil).unsqueeze(0).to(device)
        
        # 确保掩码是二值的
        mask_tensor = (mask_tensor > 0.5).float()
        
        # 推理
        with torch.no_grad():
            output_tensor = model(input_tensor, mask_tensor)
        
        # 后处理 - 将输出resize回裁剪区域的原始尺寸
        output_tensor = output_tensor.squeeze(0).cpu()
        result_crop = transform_output(output_tensor)
        result_crop = result_crop.resize((x_max - x_min, y_max - y_min), Image.Resampling.LANCZOS)
        result_crop_array = np.array(result_crop)
        
        # 创建最终结果图像（从原图复制）
        final_result = orig_img_array.copy()
        
        # 只在掩码区域替换为AI修复的结果
        crop_mask_3ch = np.stack([crop_mask] * 3, axis=-1) / 255.0
        blended_crop = (crop_img * (1 - crop_mask_3ch) + result_crop_array * crop_mask_3ch).astype(np.uint8)
        
        # 将处理后的区域放回原图
        final_result[y_min:y_max, x_min:x_max] = blended_crop
        
        # 转换回PIL图像
        final_img = Image.fromarray(final_result)
        
        # 返回结果
        img_io = io.BytesIO()
        final_img.save(img_io, 'JPEG', quality=95)
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/jpeg')
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"处理图片时发生错误: {str(e)}", 500

if __name__ == '__main__':
    print("=准备启动推理服务器=")
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=False)
