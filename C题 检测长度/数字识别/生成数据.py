import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import numpy as np
 
# 设置文件保存路径
output_dir = r'project\2025\imgs\nums'
os.makedirs(output_dir, exist_ok=True)
 
# 设置字体路径
font_path = "C:/Windows/Fonts/times.ttf"  # Times New Roman字体文件
 
# 设置生成图片的数量
num_images = 20
 
# 图像数据增强：旋转、平移、亮度调整
def augment_image(image):
    # 随机旋转
    angle = random.randint(-15, 15)
    image = image.rotate(angle, resample=Image.BICUBIC, expand=False)
 
    # 随机平移
    max_translation = 5
    translation = (random.randint(-max_translation, max_translation),
                   random.randint(-max_translation, max_translation))
    image = image.transform(
        image.size, Image.AFFINE,
        (1, 0, translation[0], 0, 1, translation[1]),
        resample=Image.BICUBIC
    )
 
    # 随机缩放 + 裁剪或填充回原始尺寸
    scale_factor = random.uniform(0.8, 1.2)
    new_size = (int(image.size[0] * scale_factor), int(image.size[1] * scale_factor))
    image = image.resize(new_size, resample=Image.BICUBIC)
 
    # 保持输出尺寸为 64x64
    if scale_factor < 1.0:
        # 缩小后，放到中央，填充白色背景
        background = Image.new('RGB', (64, 64), 'white')
        offset = ((64 - new_size[0]) // 2, (64 - new_size[1]) // 2)
        background.paste(image, offset)
        image = background
    else:
        # 放大后，居中裁剪
        left = (new_size[0] - 64) // 2
        top = (new_size[1] - 64) // 2
        image = image.crop((left, top, left + 64, top + 64))
 
    # 随机亮度调整
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.uniform(0.8, 1.2))
 
    return image
 
# 生成并保存图像
def generate_images():
    for i in range(num_images):
        # 随机选择数字
        digit = str(random.randint(0, 9))
 
        # 创建白色背景
        img = Image.new('RGB', (64, 64), color='black')
        draw = ImageDraw.Draw(img)
 
        # 随机字体大小
        font_size = random.randint(30, 50)
        font = ImageFont.truetype(font_path, font_size)
 
        # 获取文本边界框（替代 textsize）
        bbox = draw.textbbox((0, 0), digit, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
 
        # 计算文本位置（居中）
        x = (img.width - text_width) // 2
        y = (img.height - text_height) // 2
 
        # 绘制文本
        draw.text((x, y), digit, font=font, fill='black')
 
        # 增强图像
        img = augment_image(img)
 
        # 保存图片
        img.save(os.path.join(output_dir, f"{i}_{digit}.png"))
 
if __name__ == '__main__':
    generate_images()