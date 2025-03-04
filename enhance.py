import cv2
import numpy as np
from pathlib import Path
import argparse

def enhance_license_plate(image):
    """
    增强车牌图像，使车牌区域更加清晰，并输出黑白图像
    
    Args:
        image: 输入的图像
        
    Returns:
        增强后的黑白图像
    """
    # 转换为灰度图
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 自适应直方图均衡化，提高对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 高斯模糊去噪
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # 使用Canny检测边缘
    edges = cv2.Canny(blurred, 50, 150)
    
    # 形态学操作，闭操作连接车牌边缘
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 查找轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 如果找到轮廓，尝试定位车牌区域
    if contours:
        # 按轮廓面积排序
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # 尝试找到最可能是车牌的轮廓（通常是较大且是矩形的轮廓）
        for contour in contours[:5]:  # 检查面积最大的5个轮廓
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            
            # 中国车牌的宽高比大约在2.0到5.0之间
            if 1.5 <= aspect_ratio <= 6.0 and w > image.shape[1] * 0.1:
                # 创建车牌区域掩码
                mask = np.zeros_like(gray)
                cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
                
                # 对车牌区域应用局部自适应阈值二值化
                binary_adaptive = cv2.adaptiveThreshold(
                    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 11, 2
                )
                
                # 应用掩码，凸显车牌区域
                roi = cv2.bitwise_and(binary_adaptive, mask)
                
                # 将车牌区域放回整体二值化图像
                # 先对整个图像进行二值化
                _, binary_global = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # 创建反向掩码
                mask_inv = cv2.bitwise_not(mask)
                
                # 将掩码外区域与全局二值化结果合并
                background = cv2.bitwise_and(binary_global, mask_inv)
                
                # 合并车牌区域和背景
                result = cv2.add(background, roi)
                
                return result
    
    # 如果没有找到合适的车牌轮廓，则返回增强后的二值化图像
    # 对图像进行锐化操作
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(blurred, -1, kernel)
    
    # 尝试几种不同的二值化方法并选择最佳结果
    # 1. Otsu二值化
    _, binary_otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 2. 自适应阈值二值化
    binary_adaptive = cv2.adaptiveThreshold(
        sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # 对两种二值化结果进行融合（取平均）
    binary = cv2.addWeighted(binary_otsu, 0.5, binary_adaptive, 0.5, 0)
    
    return binary


def process_directory(input_dir, output_dir):
    """
    递归处理指定目录及其子目录下的所有图像文件
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 确保输出目录存在
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 支持的图像文件扩展名
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    
    # 处理所有文件和子目录
    for item in input_path.iterdir():
        # 如果是目录，递归处理
        if item.is_dir():
            # 创建对应的输出子目录
            rel_path = item.relative_to(input_path)
            sub_output_dir = output_path / rel_path
            process_directory(item, sub_output_dir)
        
        # 如果是图像文件，进行处理
        elif item.is_file() and item.suffix.lower() in image_extensions:
            try:
                # 读取图像
                image = cv2.imread(str(item))
                
                if image is None:
                    print(f"无法读取图像: {item}")
                    continue
                
                # 增强图像
                enhanced_image = enhance_license_plate(image)
                
                # 创建输出文件路径
                rel_path = item.relative_to(input_path)
                output_file = output_path / rel_path
                
                # 确保输出文件的目录存在
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                # 保存增强后的图像
                cv2.imwrite(str(output_file), enhanced_image)
                print(f"处理成功: {item} -> {output_file}")
                
            except Exception as e:
                print(f"处理图像时出错 {item}: {str(e)}")


def main():
    """
    主函数，解析命令行参数并调用处理函数
    """
    parser = argparse.ArgumentParser(description='增强车牌图像，便于OCR识别')
    parser.add_argument('--input_dir', help='输入图像目录')
    parser.add_argument('--output_dir', help='输出图像目录')
    args = parser.parse_args()
    
    process_directory(args.input_dir, args.output_dir)
    print(f"所有图像处理完成。增强后的图像已保存至: {args.output_dir}")


if __name__ == "__main__":
    main()