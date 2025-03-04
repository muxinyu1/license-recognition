# 车牌图片处理工具

## 安装依赖

```sh
poetry install
```

## `remove.py`
主要功能：
1. 扫描指定文件夹及子文件夹中的所有图片
2. 对于相似图片，保留文件名中置信度值最高的一张，其余删除
3. 可选择在删除前备份原始图片

参数说明：
- input: 需要处理的图片文件夹路径
- threshold: 相似度阈值，ssim模式下值越大表示越相似
- method: 相似度计算方法，可选'mse'或'ssim'
- backup: 可选的备份文件夹路径
- workers: 并行处理的线程数，默认8
- phash-distance: 感知哈希的最大汉明距离，用于初步分组，默认3

使用命令示例：
```
python remove.py --input 图片文件夹 --threshold 0.8 --method ssim --backup 备份文件夹
```

## `vote.py`

主要功能：
1. 结合使用HyperLPR3和阿里云OCR两种识别引擎
2. 从图片文件名中提取原始车牌号
3. 对三种识别结果（原始文件名、HyperLPR3、阿里云）进行投票
4. 如果两种或以上识别结果一致，则以多数结果重命名文件
5. 如果三种结果都不一致或有识别失败，则将图片移至人工审核文件夹

参数说明：
- folder：需要处理的图片文件夹路径
- --review：可选的人工审核文件夹，用于存放无法自动识别的图片

使用示例：
```
ALIBABA_CLOUD_ACCESS_KEY_ID=xxxx ALIBABA_CLOUD_ACCESS_KEY_SECRET=xxxx python vote.py 图片文件夹路径 --review 审核文件夹路径
```

脚本会遍历指定文件夹及子文件夹中的所有图片，对每张图片进行三种识别并投票，然后根据投票结果决定是重命名还是移至审核文件夹。运行完成后会显示统计信息，包括总处理数、重命名数、未变更数等。

> 注意：使用前需要设置环境变量ALIBABA_CLOUD_ACCESS_KEY_ID和ALIBABA_CLOUD_ACCESS_KEY_SECRET来配置阿里云API访问权限。

**经验证，经过三种模型投票后的车票具有较高的准确度，人工审核文件夹内的图片往往是人类也无法识别的车牌。**