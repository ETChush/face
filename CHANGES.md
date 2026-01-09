# 修改记录

## 2025-01-09 - MediaPipe 版本兼容性修复

### 问题
项目使用的 MediaPipe 旧 API (mp.solutions.face_mesh) 在 MediaPipe 0.10+ 版本中已被移除，导致程序无法运行。

### 解决方案
将 MediaPipe 实现从旧的 Solutions API 迁移到新的 Tasks API (mediapipe.tasks.vision.FaceLandmarker)。

### 修改的文件

#### 1. src/models/mediapipe.py
**状态**: 完全重写

**主要变更**:
- 从 `mp.solutions.face_mesh` 迁移到 `mediapipe.tasks.vision.FaceLandmarker`
- 添加模型文件路径管理
- 重写初始化方法，使用新的 API 创建检测器
- 更新人脸检测逻辑，使用新的 API 调用方式
- 修改关键点坐标转换逻辑
- 更新绘制函数以适配新的数据结构

**关键代码变更**:
```python
# 旧 API (已移除)
self.face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 新 API
base_options = python.BaseOptions(model_asset_path=self.model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
```

#### 2. download_model.py (新建)
**状态**: 创建新文件

**功能**: 下载 MediaPipe FaceLandmarker 所需的模型文件

**说明**: 
- 从 Google Cloud Storage 下载 face_landmarker.task 模型
- 模型文件保存到 models/face_landmarker.task
- 文件大小约 10MB

#### 3. test_mediapipe_new.py (新建)
**状态**: 创建新文件

**功能**: 测试新的 MediaPipe Tasks API

**说明**:
- 验证 FaceLandmarker API 是否正常工作
- 测试人脸关键点检测功能
- 验证模型文件加载和初始化

#### 4. USAGE.md (新建)
**状态**: 创建新文件

**功能**: 项目使用说明文档

**内容**:
- 安装方法
- 基本使用命令
- 常用参数说明
- 实用示例
- 注意事项

### 技术细节

#### API 迁移要点
1. **模型文件**: 新 API 需要单独下载 .task 模型文件
2. **初始化方式**: 从直接实例化改为使用 create_from_options()
3. **检测方法**: 从 process() 改为 detect()
4. **结果结构**: 返回的数据结构有所变化，需要适配
5. **坐标系统**: 保持归一化坐标，但转换逻辑需要调整

#### 修复的错误
1. **模块导入错误**: `ModuleNotFoundError: No module named 'mediapipe.python.solutions'`
   - 原因: MediaPipe 0.10+ 移除了 solutions 模块
   - 解决: 使用新的 tasks.vision 模块

2. **关键点列表初始化错误**: `NormalizedLandmarkList.__init__() missing 1 required positional argument: 'landmarks'`
   - 原因: 新 API 的数据结构初始化方式不同
   - 解决: 修改 _convert_landmarks 方法，正确传递 landmarks 列表

### 测试结果
- 成功处理 18 张测试图片
- 平均处理速度: 2.15 张/秒
- 输出文件保存在 test_images/output/ 目录
- 所有图片的人脸检测、关键点提取、对齐和年龄估计功能正常

### 兼容性
- MediaPipe 版本: 0.10.31
- Python 版本: 3.8-3.11
- 操作系统: Windows/macOS/Linux

### 后续建议
1. 考虑将模型文件下载集成到安装脚本中
2. 添加模型文件存在性检查
3. 考虑支持本地模型文件路径配置
4. 添加单元测试覆盖关键功能
