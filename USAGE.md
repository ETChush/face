# 使用说明

## 安装

```bash
pip install -e .
```

或使用 requirements.txt：

```bash
pip install -r requirements.txt
```

## 基本使用

### 对齐图片

```bash
face input_dir
```

### 生成视频

```bash
face input_dir -v
```

### 调试模式

```bash
face input_dir -d
```

## 常用参数

- `-o, --output-dir`: 输出目录
- `-d, --debug`: 调试模式，显示处理过程
- `-v, --video`: 生成过渡视频
- `--fps`: 视频帧率（默认24）
- `-t, --transition`: 过渡时间秒数（默认0.5）
- `-s, --stay`: 图片停留时间秒数（默认0.25）
- `--similarity-threshold`: 人脸相似度阈值（默认0.3）
- `-z, --size`: 输出尺寸，格式'宽x高'（如'1920x1080'）
- `--no-border`: 不添加白边

## 示例

```bash
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

## 注意事项

1. 确保输入图片中人脸清晰可见
2. 建议先用少量图片在调试模式下测试
3. 处理大量图片时确保有足够磁盘空间
4. 视频生成可能需要较长时间
