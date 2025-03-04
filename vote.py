import json
import os
import re
import cv2
import shutil
import argparse
import time  # Add timeout functionality
import hyperlpr3 as lpr3
from collections import Counter
from paddleocr import PaddleOCR
# Import statements for Alibaba Cloud remain the same
from alibabacloud_ocr_api20210707.client import Client as ocr_api20210707Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_darabonba_stream.client import Client as StreamClient
from alibabacloud_ocr_api20210707 import models as ocr_api_20210707_models
from alibabacloud_tea_util import models as util_models


class LicensePlateVoting:
    def __init__(self, review_folder=None, timeout=30):
        """初始化车牌识别投票系统"""
        # 初始化HyperLPR3识别器
        self.hyperlpr_catcher = lpr3.LicensePlateCatcher()
        
        # 检查阿里云环境变量
        self.access_key_id = os.environ.get('ALIBABA_CLOUD_ACCESS_KEY_ID')
        self.access_key_secret = os.environ.get('ALIBABA_CLOUD_ACCESS_KEY_SECRET')
        
        # 设置超时时间（秒）
        self.timeout = timeout
        
        # 检查是否设置了阿里云环境变量
        self.use_alibaba = False
        if self.access_key_id and self.access_key_secret:
            # 初始化阿里云OCR客户端
            self.ali_client = self._create_ali_client()
            self.use_alibaba = True
            print("已启用阿里云OCR识别")
        else:
            print("未设置阿里云环境变量，将不使用阿里云OCR")
        
        # 初始化PaddleOCR
        try:
            self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=False)
            print("PaddleOCR初始化成功")
        except Exception as e:
            print(f"PaddleOCR初始化失败: {str(e)}")
            self.paddle_ocr = None
        
        # 需要人工审核的图片存放文件夹
        self.review_folder = review_folder
        
        # 结果统计
        self.statistics = {
            'total': 0,
            'renamed': 0,
            'unchanged': 0,
            'moved': 0,
            'errors': 0
        }
        
    def _create_ali_client(self):
        """创建阿里云OCR客户端"""
        config = open_api_models.Config(
            access_key_id=self.access_key_id, # type: ignore
            access_key_secret=self.access_key_secret # type: ignore
        )
        config.endpoint = 'ocr-api.cn-hangzhou.aliyuncs.com'
        return ocr_api20210707Client(config)
    
    def recognize_with_hyperlpr(self, image_path):
        """使用HyperLPR3识别车牌"""
        try:
            print(f"  开始HyperLPR3识别: {image_path}")
            start_time = time.time()
            
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                print(f"HyperLPR3无法读取图片: {image_path}")
                return None
            
            # 识别车牌
            results = self.hyperlpr_catcher(image)
            
            # 计算耗时
            elapsed = time.time() - start_time
            print(f"  HyperLPR3识别耗时: {elapsed:.2f}秒")
            
            # 如果有结果返回第一个车牌号
            if results and len(results) > 0:
                return results[0][0]  # 车牌号
            return None
        except Exception as e:
            print(f"HyperLPR3识别出错: {str(e)}")
            return None
    
    def recognize_with_ali(self, image_path):
        """使用阿里云OCR识别车牌"""
        # 如果未启用阿里云，直接返回None
        if not self.use_alibaba:
            return None
            
        try:
            print(f"  开始阿里云OCR识别: {image_path}")
            start_time = time.time()
            
            # 检查文件是否存在
            if not os.path.exists(image_path):
                print(f"文件不存在: {image_path}")
                return None
            
            # 从文件读取图片数据流
            body_stream = StreamClient.read_from_file_path(image_path)
            
            # 创建请求对象
            recognize_request = ocr_api_20210707_models.RecognizeCarNumberRequest(
                body=body_stream
            )
            
            # 设置运行时选项 - 添加超时设置
            runtime = util_models.RuntimeOptions(
                connect_timeout=self.timeout * 1000,  # 毫秒
                read_timeout=self.timeout * 1000  # 毫秒
            )
            
            # 发送请求并获取响应
            response = self.ali_client.recognize_car_number_with_options(recognize_request, runtime)
            result = response.body.to_map()
            
            # 计算耗时
            elapsed = time.time() - start_time
            print(f"  阿里云OCR识别耗时: {elapsed:.2f}秒")
            
            # 解析结果
            try:
                result = json.loads(result['Data'])
                return result["data"]["data"]["车牌"]
            except Exception as e:
                print(f"阿里云的响应格式不对，响应为{json.dumps(result, indent=2)}")
                return None
        except Exception as e:
            print(f"阿里云识别出错: {str(e)}")
            return None
    
    def recognize_with_paddle(self, image_path):
        """使用PaddleOCR识别车牌"""
        # 如果PaddleOCR初始化失败，直接返回None
        if self.paddle_ocr is None:
            return None
            
        try:
            print(f"  开始PaddleOCR识别: {image_path}")
            start_time = time.time()
            
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                print(f"PaddleOCR无法读取图片: {image_path}")
                return None
            
            # 设置超时功能
            # PaddleOCR没有内置的超时功能，我们可以在调用后检查是否超时
            result = self.paddle_ocr.ocr(image, cls=True)
            
            # 计算耗时
            elapsed = time.time() - start_time
            print(f"  PaddleOCR识别耗时: {elapsed:.2f}秒")
            
            # 如果耗时超过超时时间，视为超时
            if elapsed > self.timeout:
                print(f"  PaddleOCR识别超时，耗时 {elapsed:.2f}秒 > {self.timeout}秒")
            
            # 如果没有识别结果
            if not result or len(result) == 0 or not result[0]:
                return None
            
            # 提取所有文本
            texts = []
            for line in result[0]:
                if len(line) >= 2 and isinstance(line[1], tuple) and len(line[1]) >= 1:
                    texts.append(line[1][0])
            
            # 使用正则表达式找到车牌号
            # 中国车牌号正则表达式(包括新能源车牌)
            plate_pattern = r'[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领][A-Z][A-Z0-9]{5,6}'
            
            for text in texts:
                match = re.search(plate_pattern, text)
                if match:
                    return match.group(0)
            
            # 如果没有找到匹配的车牌号，选择最长的文本作为可能的车牌
            if texts:
                # 按长度排序并选择最有可能是车牌的文本
                candidates = sorted(texts, key=len)
                for candidate in candidates:
                    # 车牌长度通常为7-8个字符
                    if 6 <= len(candidate) <= 9 and any(c.isdigit() for c in candidate):
                        return candidate
            
            return None
        except Exception as e:
            print(f"PaddleOCR识别出错: {str(e)}")
            return None
    
    def extract_original_plate(self, filename):
        """从文件名中提取原始车牌号"""
        # 假设文件名格式为：鲁N3D523_16687_0_9212828.jpg
        match = re.match(r'^([^_]+)_.*$', filename)
        if match:
            return match.group(1)
        return None
    
    def process_folder(self, folder_path):
        """处理文件夹及其子文件夹内的所有图片"""
        # 重置统计信息
        self.statistics = {
            'total': 0,
            'renamed': 0,
            'unchanged': 0,
            'moved': 0,
            'errors': 0
        }
        
        # 检查文件夹是否存在
        if not os.path.exists(folder_path):
            print(f"文件夹不存在: {folder_path}")
            return
        
        # 创建人工审核文件夹（如果指定了）
        if self.review_folder and not os.path.exists(self.review_folder):
            try:
                os.makedirs(self.review_folder)
                print(f"创建人工审核文件夹: {self.review_folder}")
            except Exception as e:
                print(f"创建人工审核文件夹失败: {str(e)}")
                return
        
        # 存储所有找到的图片文件
        image_files_with_paths = []
        
        # 图片文件扩展名
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        # 遍历所有子文件夹
        for root, dirs, files in os.walk(folder_path):
            # 收集当前文件夹中的所有图片
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    rel_path = os.path.relpath(root, folder_path)
                    rel_path = '' if rel_path == '.' else rel_path
                    image_files_with_paths.append((root, file, rel_path))
        
        total_files = len(image_files_with_paths)
        print(f"找到 {total_files} 个图片文件")
        
        # 处理每个图片
        for i, (root_path, filename, rel_path) in enumerate(image_files_with_paths):
            self.statistics['total'] += 1
            
            # 显示相对路径信息
            if rel_path:
                print(f"处理 [{i+1}/{total_files}]: {os.path.join(rel_path, filename)}")
            else:
                print(f"处理 [{i+1}/{total_files}]: {filename}")
            
            file_path = os.path.join(root_path, filename)
            
            try:
                # 获取原始识别结果（从文件名）
                original_plate = self.extract_original_plate(filename)
                print(f"  原始识别: {original_plate}")
                
                # 使用HyperLPR3识别
                try:
                    hyperlpr_plate = self.recognize_with_hyperlpr(file_path)
                    print(f"  HyperLPR3识别: {hyperlpr_plate}")
                except Exception as e:
                    print(f"  HyperLPR3识别异常: {str(e)}")
                    hyperlpr_plate = None
                
                # 使用PaddleOCR识别(添加超时保护)
                paddle_plate = None
                try:
                    # 使用另一个线程来执行PaddleOCR识别可能是更好的方法
                    # 但为了简单起见，我们在这里只添加异常处理
                    paddle_plate = self.recognize_with_paddle(file_path)
                    print(f"  PaddleOCR识别: {paddle_plate}")
                except Exception as e:
                    print(f"  PaddleOCR识别异常: {str(e)}")
                
                # 阿里云识别结果（仅在启用阿里云时获取）
                ali_plate = None
                if self.use_alibaba:
                    try:
                        ali_plate = self.recognize_with_ali(file_path)
                        print(f"  阿里云识别: {ali_plate}")
                    except Exception as e:
                        print(f"  阿里云识别异常: {str(e)}")
                
                # 检查是否需要移动到人工审核文件夹
                valid_results = [r for r in [original_plate, hyperlpr_plate, ali_plate, paddle_plate] if r]
                
                if self.review_folder and (
                    len(valid_results) < 2 or  # 少于两个有效结果
                    len(set(valid_results)) == len(valid_results)  # 所有结果都不同
                ):
                    self._move_to_review_folder(root_path, filename, rel_path)
                else:
                    # 进行投票
                    self._vote_and_rename(root_path, filename, original_plate, hyperlpr_plate, ali_plate, paddle_plate)
                
            except Exception as e:
                print(f"处理文件时出错: {str(e)}")
                # 移动到人工审核文件夹(如果设置了)
                if self.review_folder:
                    self._move_to_review_folder(root_path, filename, rel_path)
                self.statistics['errors'] += 1
        
        # 打印统计信息
        print("\n处理完成!")
        print(f"总计处理: {self.statistics['total']} 个文件")
        print(f"重命名: {self.statistics['renamed']} 个文件")
        print(f"保持不变: {self.statistics['unchanged']} 个文件")
        if self.review_folder:
            print(f"移动到人工审核: {self.statistics['moved']} 个文件")
        print(f"处理出错: {self.statistics['errors']} 个文件")
    
    def _move_to_review_folder(self, folder_path, filename, rel_path=''):
        """将文件移动到人工审核文件夹，保持相对路径结构"""
        if not self.review_folder:
            return
        
        source_path = os.path.join(folder_path, filename)
        
        # 如果有相对路径，在审核文件夹中创建对应的子文件夹结构
        if rel_path:
            dest_folder = os.path.join(self.review_folder, rel_path)
            if not os.path.exists(dest_folder):
                try:
                    os.makedirs(dest_folder)
                except Exception as e:
                    print(f"  创建审核子文件夹失败: {str(e)}")
                    self.statistics['errors'] += 1
                    return
            dest_path = os.path.join(dest_folder, filename)
        else:
            dest_path = os.path.join(self.review_folder, filename)
        
        try:
            shutil.move(source_path, dest_path)
            if rel_path:
                print(f"  移动文件到人工审核文件夹: {os.path.join(rel_path, filename)}")
            else:
                print(f"  移动文件到人工审核文件夹: {filename}")
            self.statistics['moved'] += 1
        except Exception as e:
            print(f"  移动文件失败: {str(e)}")
            self.statistics['errors'] += 1
    
    def _vote_and_rename(self, folder_path, filename, original_plate, hyperlpr_plate, ali_plate, paddle_plate):
        """进行投票并重命名文件"""
        # 收集有效的识别结果
        valid_results = []
        if original_plate:
            valid_results.append(original_plate)
        if hyperlpr_plate:
            valid_results.append(hyperlpr_plate)
        if ali_plate:
            valid_results.append(ali_plate)
        if paddle_plate:
            valid_results.append(paddle_plate)
        
        # 如果没有有效结果，保持不变
        if not valid_results:
            print("  没有有效的识别结果，保持文件名不变")
            self.statistics['unchanged'] += 1
            return
        
        # 进行投票
        votes = Counter(valid_results)
        most_common = votes.most_common(1)[0]
        plate_number = most_common[0]
        vote_count = most_common[1]
        
        # 获取文件名后缀部分
        suffix_match = re.match(r'^[^_]+(_.*\..*)$', filename)
        suffix = suffix_match.group(1) if suffix_match else f"_{filename.split('_', 1)[1]}" if '_' in filename else ""
        
        # 获取文件扩展名
        _, extension = os.path.splitext(filename)
        
        # 如果没有明显的赢家（最高票数只有1票），保持原文件名
        if vote_count == 1 and len(valid_results) > 1:
            print("  没有明显的赢家，保持文件名不变")
            self.statistics['unchanged'] += 1
            return
        
        # 如果已经是最高票数的车牌，不需要重命名
        if filename.startswith(plate_number):
            print(f"  文件名已经是识别结果: {plate_number}，不需要重命名")
            self.statistics['unchanged'] += 1
            return
        
        # 创建新文件名
        if suffix:
            new_filename = f"{plate_number}{suffix}"
        else:
            new_filename = f"{plate_number}{extension}"
        
        # 重命名文件
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_filename)
        
        try:
            os.rename(old_path, new_path)
            print(f"  重命名文件: {filename} -> {new_filename}")
            self.statistics['renamed'] += 1
        except Exception as e:
            print(f"  重命名文件失败: {str(e)}")
            self.statistics['errors'] += 1


def main():
    """主函数"""
    print("车牌识别投票系统（含PaddleOCR）")
    print("=" * 50)
    print("注意：如需启用阿里云OCR，请设置环境变量 ALIBABA_CLOUD_ACCESS_KEY_ID 和 ALIBABA_CLOUD_ACCESS_KEY_SECRET")
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='车牌识别投票系统（含PaddleOCR）')
    parser.add_argument('folder', help='车牌图片所在文件夹路径')
    parser.add_argument('--review', help='需要人工审核的图片存放文件夹路径')
    parser.add_argument('--timeout', type=int, default=30, help='识别超时时间（秒），默认30秒')
    
    args = parser.parse_args()
    
    # 创建投票系统
    voter = LicensePlateVoting(review_folder=args.review, timeout=args.timeout)
    
    # 处理文件夹
    voter.process_folder(args.folder)


if __name__ == '__main__':
    main()