import os
import cv2
import numpy as np
import re
import shutil
from pathlib import Path
import concurrent.futures
from tqdm import tqdm
from scipy.fftpack import dct


def extract_confidence(filename):
    """
    从文件名中提取置信度值，例如从"京N3D525_383_0_8937652.jpg"提取0.8937652
    """
    # 使用正则表达式从文件名中提取数字部分
    match = re.search(r'_(\d+)\.', filename)
    if match:
        confidence_str = match.group(1)
        return float('0.' + confidence_str)
    return 0.0


def compute_phash(image, hash_size=8):
    """
    计算图片的感知哈希值(pHash)
    
    参数:
    image: 输入图像
    hash_size: 感知哈希的大小
    
    返回:
    感知哈希值(二进制字符串)
    """
    # 先将图像调整为32x32大小，这比要求的哈希大小大，便于进行DCT变换
    resized = cv2.resize(image, (32, 32))
    
    # 转换为灰度图像
    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized
    
    # 对灰度图像进行DCT变换
    dct_coef = dct(dct(gray.T, norm='ortho').T, norm='ortho')
    
    # 只取左上角的系数，这些包含了图像的低频信息
    dct_low = dct_coef[:hash_size, :hash_size]
    
    # 计算DCT系数的中值(不包括直流分量)
    dct_flat = dct_low.flatten()
    # 跳过第一个直流系数，因为它代表整体亮度
    median = np.median(dct_flat[1:])
    
    # 根据DCT系数是否大于中值生成哈希值
    phash_bits = (dct_low > median).flatten()
    
    # 将布尔数组转换为整数
    phash_value = 0
    for bit in phash_bits:
        phash_value = (phash_value << 1) | int(bit)
    
    return phash_value


def hamming_distance(hash1, hash2):
    """
    计算两个哈希值之间的汉明距离
    
    参数:
    hash1, hash2: 两个整数哈希值
    
    返回:
    汉明距离，表示哈希值中不同位的数量
    """
    # 对哈希值进行异或操作，结果中的1表示不同的位
    xor_result = hash1 ^ hash2
    
    # 计算结果中1的个数
    distance = bin(xor_result).count('1')
    
    return distance


def compute_image_similarity(img1, img2, method='mse'):
    """
    计算两张图片的相似度
    method: 'mse'(均方误差) 或 'ssim'(结构相似性指数)
    """
    # 调整图片大小以确保它们具有相同的尺寸，并降低分辨率以加快计算
    target_size = (256, 256)  # 降低分辨率加快比较
    img1_resized = cv2.resize(img1, target_size)
    img2_resized = cv2.resize(img2, target_size)
    
    if method == 'mse':
        # 均方误差，值越小表示图片越相似
        err = np.sum((img1_resized.astype("float") - img2_resized.astype("float")) ** 2)
        err /= float(img1_resized.shape[0] * img1_resized.shape[1])
        return err
    elif method == 'ssim':
        # 结构相似性指数，值越接近1表示图片越相似
        from skimage.metrics import structural_similarity as ssim
        # 确定合适的窗口大小，必须是奇数且不大于图像最小边长
        min_side = min(img1_resized.shape[0], img1_resized.shape[1])
        # 从可能的奇数窗口尺寸中选择一个合适的值
        win_size = min(7, min_side - (min_side % 2 == 0))  # 确保是奇数且不大于最小边长
        if win_size < 3:  # SSIM至少需要3x3的窗口
            win_size = 3
        # 对于彩色图像
        if len(img1_resized.shape) == 3:
            return ssim(img1_resized, img2_resized, win_size=win_size, channel_axis=2)
        # 对于灰度图像
        else:
            return ssim(img1_resized, img2_resized, win_size=win_size)


def load_images_parallel(image_paths, max_workers=8):
    """
    并行加载图片并计算感知哈希值
    """
    results = []
    
    def process_image(img_info):
        try:
            # 读取图片
            img = cv2.imread(img_info['full_path'])
            if img is None:
                return None
            
            # 计算感知哈希
            img_hash = compute_phash(img)
            img_info['image'] = img
            img_info['phash'] = img_hash
            return img_info
        except Exception as e:
            print(f"处理图片出错 {img_info['full_path']}: {str(e)}")
            return None
    
    # 使用线程池并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_image, img_info) for img_info in image_paths]
        
        # 使用tqdm显示进度
        for future in tqdm(concurrent.futures.as_completed(futures), 
                          total=len(futures), 
                          desc="加载图片"):
            result = future.result()
            if result is not None:
                results.append(result)
    
    return results


def group_similar_by_phash(images, max_distance=2):
    """
    使用感知哈希值对图片进行预分组，根据汉明距离聚类
    
    参数:
    images: 带有phash值的图片信息列表
    max_distance: 最大汉明距离，小于等于此距离被认为是相似图片
    
    返回:
    分组后的图片列表
    """
    # 按照phash值排序，便于寻找相似哈希
    images.sort(key=lambda x: x['phash'])
    
    # 结果分组
    grouped_images = []
    processed = set()
    
    # 对每张未处理的图片找出其相似图片
    for i in range(len(images)):
        if i in processed:
            continue
        
        # 当前图片作为一个新组的起点
        current_group = [images[i]]
        processed.add(i)
        
        # 在剩余图片中寻找与当前图片相似的图片
        for j in range(i+1, len(images)):
            if j in processed:
                continue
            
            # 计算汉明距离
            distance = hamming_distance(images[i]['phash'], images[j]['phash'])
            
            # 如果距离小于阈值，将其添加到当前组
            if distance <= max_distance:
                current_group.append(images[j])
                processed.add(j)
        
        # 将当前组添加到结果中
        grouped_images.append(current_group)
    
    return grouped_images


def remove_duplicate_images(input_folder, similarity_threshold=100, method='mse', backup_folder=None, max_workers=8, phash_distance=3):
    """
    处理文件夹及其子文件夹中的图片，删除相似度高的重复图片，只保留置信度最高的一张
    
    参数:
    input_folder: 输入图片文件夹路径
    similarity_threshold: 相似度阈值，mse方法下值越小表示越相似
    method: 相似度计算方法，'mse'或'ssim'
    backup_folder: 可选的备份文件夹路径，如果提供，将在删除前备份原始图片
    max_workers: 并行处理的最大线程数
    phash_distance: 感知哈希的最大汉明距离，用于预分组
    """
    # 如果有备份文件夹，创建它
    if backup_folder:
        os.makedirs(backup_folder, exist_ok=True)
        print(f"将在删除前备份原始图片到: {backup_folder}")
    
    # 存储所有图片文件的路径信息
    all_images = []
    
    # 处理图片文件的格式
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    
    print("收集图片文件信息...")
    
    # 递归遍历所有子文件夹
    for root, dirs, files in os.walk(input_folder):
        # 获取相对于输入文件夹的路径
        rel_path = os.path.relpath(root, input_folder)
        rel_path = '' if rel_path == '.' else rel_path
        
        # 收集当前文件夹中的所有图片
        for file in files:
            if file.lower().endswith(image_extensions):
                # 保存完整路径、文件名和相对路径
                all_images.append({
                    'full_path': os.path.join(root, file),
                    'filename': file,
                    'rel_path': rel_path,
                    'confidence': extract_confidence(file)
                })
    
    total_images = len(all_images)
    print(f"总共找到 {total_images} 张图片")
    
    if total_images == 0:
        print("没有找到图片，退出程序")
        return 0
    
    # 按照置信度排序图片，确保保留置信度最高的图片
    all_images.sort(key=lambda x: x['confidence'], reverse=True)
    
    print("并行加载图片并计算感知哈希值...")
    loaded_images = load_images_parallel(all_images, max_workers)
    print(f"成功加载 {len(loaded_images)} 张图片")
    
    # 使用感知哈希值进行预分组
    print("使用感知哈希对图片进行预分组...")
    grouped_images = group_similar_by_phash(loaded_images, phash_distance)
    print(f"根据感知哈希分为 {len(grouped_images)} 组")
    
    # 要删除的图片列表
    to_delete = []
    
    print("开始检测重复图片...")
    # 对每组哈希值相似的图片进行精确比较
    for group_idx, group in enumerate(grouped_images):
        # 如果只有一张图片，它肯定是唯一的
        if len(group) == 1:
            continue
        
        # 对组内图片按置信度排序（虽然应该已经排序，但以防万一）
        group.sort(key=lambda x: x['confidence'], reverse=True)
        
        # 标记组内最高置信度的图片为"保留"
        keep_img = group[0]
        
        # 对剩余图片进行比较
        for i in range(1, len(group)):
            # 计算精确相似度
            similarity = compute_image_similarity(keep_img['image'], group[i]['image'], method)
            
            # 判断是否相似
            is_similar = False
            if method == 'mse' and similarity < similarity_threshold: # type: ignore
                is_similar = True
            elif method == 'ssim' and similarity > similarity_threshold: # type: ignore
                is_similar = True
                
            if is_similar:
                to_delete.append(group[i])
                
                if group[i]['rel_path']:
                    rel_path_str = os.path.join(group[i]['rel_path'], group[i]['filename'])
                else:
                    rel_path_str = group[i]['filename']
                
                if keep_img['rel_path']:
                    img_path_str = os.path.join(keep_img['rel_path'], keep_img['filename'])
                else:
                    img_path_str = keep_img['filename']
                    
                print(f"发现重复: {rel_path_str} 与 {img_path_str} 相似，将删除 {rel_path_str}")
        
        if (group_idx + 1) % 10 == 0:
            print(f"已处理 {group_idx + 1}/{len(grouped_images)} 组")
    
    # 删除重复图片
    print(f"发现 {len(to_delete)} 张重复图片，开始删除...")
    
    for img_info in tqdm(to_delete, desc="删除重复图片"):
        if backup_folder:
            # 备份图片
            if img_info['rel_path']:
                backup_subdir = os.path.join(backup_folder, img_info['rel_path'])
                os.makedirs(backup_subdir, exist_ok=True)
                backup_path = os.path.join(backup_subdir, img_info['filename'])
            else:
                backup_path = os.path.join(backup_folder, img_info['filename'])
            
            try:
                shutil.copy2(img_info['full_path'], backup_path)
            except Exception as e:
                print(f"备份失败 {img_info['full_path']}: {str(e)}")
        
        # 删除文件
        try:
            os.remove(img_info['full_path'])
        except Exception as e:
            print(f"删除失败 {img_info['full_path']}: {str(e)}")
    
    kept_count = total_images - len(to_delete)
    print(f"处理完成！总共 {total_images} 张图片中删除了 {len(to_delete)} 张重复图片，保留了 {kept_count} 张")
    return kept_count


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='删除相似图片，保留置信度最高的图片')
    parser.add_argument('--input', type=str, required=True, help='输入图片文件夹路径')
    parser.add_argument('--threshold', type=float, default=100, help='相似度阈值 (MSE方法下值越小表示越相似)')
    parser.add_argument('--method', type=str, default='mse', choices=['mse', 'ssim'], help='相似度计算方法')
    parser.add_argument('--backup', type=str, help='删除前备份原始图片的文件夹路径（可选）')
    parser.add_argument('--workers', type=int, default=8, help='并行处理的线程数（默认为8）')
    parser.add_argument('--phash-distance', type=int, default=3, help='感知哈希的最大汉明距离，用于预分组（默认为3）')
    
    args = parser.parse_args()
    
    remove_duplicate_images(
        args.input, 
        args.threshold, 
        args.method, 
        args.backup, 
        args.workers,
        args.phash_distance
    )