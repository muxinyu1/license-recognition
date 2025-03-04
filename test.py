import cv2
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

def recognize_license_plate(image_path):
    """
    使用PaddleOCR识别车牌
    
    Args:
        image_path: 车牌图片路径
        
    Returns:
        识别结果
    """
    # 初始化PaddleOCR，设置为使用中文模型
    ocr = PaddleOCR(use_angle_cls=True, lang='ch')  # 需要启用文字方向分类器
    
    # 使用PIL读取图片
    try:
        pil_img = Image.open(image_path)
        # 转换为numpy数组，并从RGB转为BGR（OpenCV格式）
        img = np.array(pil_img)
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = img[:, :, ::-1].copy()  # RGB转BGR
        print(f"成功读取图片，尺寸为: {img.shape}")
    except Exception as e:
        print(f"PIL读取图片出错: {e}")
        return f"无法读取图片: {e}"
    
    # 进行OCR识别
    result = ocr.ocr(img, cls=True)
    
    # 提取结果
    license_text = ""
    confidence = 0
    
    if result:
        # 取第一页结果
        for line in result[0]:
            text = line[1][0]  # 识别的文本
            conf = line[1][1]  # 置信度
            print(f"检测到文本: {text}, 置信度: {conf}")
            
            # 简单的车牌筛选逻辑（可根据需要调整）
            # 中国车牌通常为7-8个字符，第一个为汉字
            if len(text) >= 6 and conf > confidence:
                license_text = text
                confidence = conf
    
    if license_text:
        return f"车牌号: {license_text}, 置信度: {confidence:.2f}"
    else:
        return "未检测到车牌"

def visualize_result(image_path, output_path=None):
    """
    可视化识别结果
    
    Args:
        image_path: 车牌图片路径
        output_path: 输出图片路径，默认为None（不保存）
    """
    # 初始化PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='ch')
    
    # 使用PIL读取图片
    try:
        pil_img = Image.open(image_path)
        # 转换为numpy数组，并从RGB转为BGR（OpenCV格式）
        img = np.array(pil_img)
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = img[:, :, ::-1].copy()  # RGB转BGR
        print(f"成功读取图片，尺寸为: {img.shape}")
    except Exception as e:
        print(f"PIL读取图片出错: {e}")
        return f"无法读取图片: {e}"
    
    print("成功读取图片")
    
    # 进行OCR识别
    result = ocr.ocr(img, cls=True)
    
    # 可视化结果
    if result:
        # 绘制检测框和识别结果
        for line in result[0]:
            boxes = line[0]  # 检测框坐标
            text = line[1][0]  # 识别的文本
            
            # 绘制矩形框
            points = np.array(boxes, np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [points], True, (0, 255, 0), 2)
            
            # 添加文本
            cv2.putText(img, text, (int(boxes[0][0]), int(boxes[0][1])-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # 显示图像
    cv2.imshow("License Plate Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 保存结果
    if output_path:
        cv2.imwrite(output_path, img)
        print(f"结果已保存至: {output_path}")

# 使用示例
if __name__ == "__main__":
    image_path = "combined_out/96_3_d1_out/京N30525_33356_0_8901241.jpg"  # 替换为你的车牌图片路径
    
    # 路径检查
    import os
    if os.path.exists(image_path):
        print(f"文件存在: {image_path}")
    else:
        print(f"文件不存在: {image_path}")
        exit(1)
    
    # 检查文件是否可访问
    try:
        with open(image_path, 'rb') as f:
            print("文件可以打开")
    except Exception as e:
        print(f"文件无法打开: {e}")
        exit(1)
    
    # 文本识别
    result = recognize_license_plate(image_path)
    print(result)
    
    # 可视化识别结果
    visualize_result(image_path, "result.jpg")