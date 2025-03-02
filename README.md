# Face - 人脸对齐与视频生成工具

这是一个强大的人脸处理工具，可以自动对齐照片中的人脸并生成平滑的过渡视频。该工具使用先进的人工智能技术，包括YOLOv8用于人脸检测，MediaPipe用于关键点提取，以及InsightFace用于人脸特征提取和相似度比较。

## 主要功能

- 自动检测和对齐照片中的人脸
- 基于人脸特征的智能匹配
- 生成平滑的过渡视频
- 支持调试模式，可视化处理过程
- 灵活的输出选项（尺寸、边框等）

## 系统要求

- Python >= 3.8
- macOS/Linux/Windows

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/kyleliu/face
cd face
```

2. 安装依赖：
```bash
pip install -e .
```

或者直接通过requirements.txt安装：
```bash
pip install -r requirements.txt
```

## 使用方法

基本用法：

```bash
# 基本用法，只对齐图片
face input_dir

# 生成视频
face input_dir -v

# 调试模式
face input_dir -d

# 指定输出目录
face input_dir -o output_dir

# 自定义视频参数
face input_dir -v --fps 30 -t 1.0 -s 0.5

# 自定义相似度阈值
face input_dir --similarity-threshold 0.6

# 自定义输出尺寸
face input_dir --size 1920x1080

# 不添加白边
face input_dir --no-border
```

### 参数说明

- `input_dir`: 输入图片目录
- `-o, --output-dir`: 输出目录路径（默认为输入目录下的output子目录）
- `-d, --debug`: 是否输出调试信息和标记图片
- `-v, --video`: 是否生成过渡视频
- `--fps`: 视频帧率（默认24fps）
- `-t, --transition`: 图片过渡时间（秒）（默认0.5秒）
- `-s, --stay`: 图片停留时间（秒）（默认0.25秒）
- `--similarity-threshold`: 人脸相似度阈值（默认0.3）
- `-z, --size`: 输出图片尺寸，格式为'宽x高'（如'1920x1080'）
- `--border/--no-border`: 是否添加白边（默认添加）

## 处理流程

1. **图片排序**
   - 按数字或字节序对输入目录中的图片进行排序

2. **基准图片处理**
   - 使用YOLOv8检测第一张图片中的人脸
   - 使用MediaPipe提取人脸关键点
   - 调整人脸至标准位置（眼睛水平，嘴巴在下）
   - 使用InsightFace获取人脸特征向量作为基准

3. **后续图片处理**
   - 对每张图片重复人脸检测和特征提取过程
   - 选择与基准人脸最相似的人脸
   - 根据眼睛位置调整图片角度和大小
   - 保持所有图片尺寸一致

4. **视频生成（可选）**
   - 生成平滑的过渡效果
   - 支持自定义帧率和过渡时间

## 依赖库

- opencv-python >= 4.8.0
- numpy >= 1.24.0
- mediapipe >= 0.10.0
- click >= 8.0.0
- tqdm >= 4.65.0
- ultralytics >= 8.0.0
- insightface >= 0.7.0
- onnxruntime >= 1.15.0

## 项目结构

```
face/
├── src/
│   ├── main.py          # 主程序入口
│   ├── models/          # 模型相关代码
│   ├── processors/      # 图片处理相关代码
│   ├── utils/          # 工具函数
│   └── video/          # 视频生成相关代码
├── test_images/        # 测试图片目录
├── requirements.txt    # 项目依赖
└── setup.py           # 安装配置
```

## 注意事项

1. 确保输入图片质量良好，人脸清晰可见
2. 建议先使用少量图片在调试模式下测试
3. 处理大量图片时，请确保有足够的磁盘空间
4. 视频生成可能需要较长时间，请耐心等待

## License

MIT License

Copyright (c) 2024 Kyle Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. 