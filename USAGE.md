# OnionParsing 使用指南

## 1. CLI命令行方式

### 基本用法

```bash
# 基本处理
python -m onion_parsing.cli.main input.pdf output.md

# 使用自定义配置文件
python -m onion_parsing.cli.main input.pdf output.md --config custom.yaml

# 启用详细日志
python -m onion_parsing.cli.main input.pdf output.md --verbose

# 指定运行的processors
python -m onion_parsing.cli.main input.pdf output.md --processors coarse_detector,sorter,cropper,ocr

# 跳过某些processors
python -m onion_parsing.cli.main input.pdf output.md --skip-processors reorder,aggregator
```

### CLI参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `input` | 输入PDF文件路径 | `input.pdf` |
| `output` | 输出Markdown文件路径 | `output.md` |
| `--config` | 配置文件路径 | `--config custom.yaml` |
| `--processors` | 指定运行的processors（逗号分隔） | `--processors coarse_detector,ocr` |
| `--skip-processors` | 跳过的processors（逗号分隔） | `--skip-processors reorder` |
| `--verbose` | 启用详细日志（DEBUG级别） | `--verbose` |

---

## 2. Python API方式

### 基本用法

```python
from onion_parsing.core.pipeline import Pipeline

# 使用默认配置
pipeline = Pipeline()
result = pipeline.process("input.pdf", "output.md")

# 使用自定义配置文件
pipeline = Pipeline(config_path="custom.yaml")
result = pipeline.process("input.pdf", "output.md")

# 跳过某些阶段
pipeline = Pipeline(skip_stages=["reorder"])
result = pipeline.process("input.pdf", "output.md")

# 运行时配置覆盖
runtime_config = {
    "coarse_detector": {"threshold": 0.3},
    "ocr": {"vl_rec_server_url": "http://localhost:8118/v1"}
}
pipeline = Pipeline(runtime_config=runtime_config)
result = pipeline.process("input.pdf", "output.md")
```

### 获取处理结果

```python
result = pipeline.process("input.pdf", "output.md")

# 结果包含以下字段
print(result["final_clean_markdown"])  # 最终清理后的Markdown
print(result["predict_md"])  # OCR识别结果字典
print(result["final_img_arrays"])  # 裁剪的图片数组列表
print(result["final_img_names"])  # 裁剪的图片名称列表
print(result["native_results"])  # PaddleOCRVL原生结果列表
```

---

## 3. 单独使用Reorder模块

### 基本用法

```python
from onion_parsing.processors.reorder import reorder_markdown

# 带 bbox 的 Markdown 字符串
md_with_bbox = """
## 标题1
bbox:[100,200,300,400]

这是一段文本内容，
bbox:[100,400,300,600]

## 标题2
bbox:[350,200,550,400]
"""

# 执行精排序
final_md = reorder_markdown(md_with_bbox)
print(final_md)
```

### 自定义参数

```python
from onion_parsing.processors.reorder.reorder_models import reorder_markdown

final_md = reorder_markdown(
    md_with_bbox,
    model_path="/data/MGSO/liuyadong/mlmv1/checkpoint-1250",
    model_type="nsp",
    device_str="npu:0",
    max_length=256,
    left_threshold=200,  # 左侧邻近阈值（像素）
    below_threshold=100,  # 下方邻近阈值（像素）
    below_left_threshold=1000,  # 左侧邻近底边阈值（像素）
    nsp_threshold=0.6,  # 连接概率阈值
    distance_scale=500  # 距离权重缩放因子
)
```

### 使用ReorderEngine

```python
from onion_parsing.processors.reorder.engine import ReorderEngine
from onion_parsing.processors.reorder.predictors import NSPPredictor
from onion_parsing.processors.reorder.utils import parse_markdown_from_str, blocks_to_markdown
import torch

# 加载NSP模型
predictor = NSPPredictor(
    model_path="/data/MGSO/liuyadong/mlmv1/checkpoint-1250",
    device=torch.device("npu:0"),
    max_length=256
)
predictor.load_model()

# 创建Reorder引擎
engine = ReorderEngine(
    predictor=predictor,
    left_threshold=200,
    below_threshold=100,
    below_left_threshold=1000,
    nsp_threshold=0.6,
    distance_scale=500
)

# 解析Markdown
blocks = parse_markdown_from_str(md_with_bbox)

# 执行重排序
reordered_blocks = engine.reorder(blocks)

# 转换为Markdown
final_md = blocks_to_markdown(reordered_blocks)
```

---

## 4. 配置文件说明

### default.yaml配置示例

```yaml
# 粗粒度版面检测模型配置
coarse_detector:
  model_name: "PP-DocLayout_plus-L"
  model_dir: "/path/to/coarse_detection_model"
  threshold: 0.25
  target_size: [1240, 1755]
  layout_merge_bboxes_mode: "large"
  device: "npu:0"

# 细粒度版面检测模型配置
fine_detector:
  model_name: "PP-DocLayout_plus-L"
  model_dir: "/path/to/fine_detection_model"
  threshold: 0.1
  layout_merge_bboxes_mode: "large"
  area_ratio_threshold: 0.125
  aspect_ratio_ranges: [[0.15, 0.71], [1.45, 6.8]]
  smallcrop_scale: [0.2, 0.0667]
  device: "npu:0"

# OCR配置
ocr:
  pipeline_version: "v1.6"
  vl_rec_backend: "vllm-server"
  vl_rec_server_url: "http://localhost:8118/v1"
  vl_rec_max_concurrency: 32
  vl_rec_fallback_concurrency: 8
  timeout: 120
  vl_rec_max_concurrency: 32

# 精排序配置
reorder:
  model_path: "/path/to/nsp_model"
  model_type: "nsp"
  left_threshold: 200
  below_threshold: 100
  below_left_threshold: 1000
  nsp_threshold: 0.6
  max_length: 256
  device: "npu:0"

# Pipeline配置
pipeline:
  processors:
    - coarse_detector
    - sorter
    - cropper
    - crop_filter
    - fine_detector
    - column_expander
    - preprocessor
    - ocr
    - postprocessor
    - reorder
    - aggregator

# 日志配置
logging:
  level: "INFO"
  file: "onion_parsing.log"
  format: "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
```

### 配置优先级

1. **运行时参数**（最高优先级）
   ```python
   runtime_config = {"coarse_detector": {"threshold": 0.3}}
   pipeline = Pipeline(runtime_config=runtime_config)
   ```

2. **环境变量**
   ```bash
   export OP_COARSE_DETECTOR_THRESHOLD=0.25
   export OP_REORDER_NSP_THRESHOLD=0.6
   ```

3. **YAML配置文件**
   ```yaml
   coarse_detector:
     threshold: 0.25
   ```

4. **默认配置**（最低优先级）
   - 代码中的默认值

---

## 5. 环境变量映射规则

环境变量格式：`OP_<PROCESSOR>_<PARAM>`

示例：

| 环境变量 | 配置路径 | 说明 |
|----------|----------|------|
| `OP_COARSE_DETECTOR_THRESHOLD` | `coarse_detector.threshold` | 版面检测阈值 |
| `OP_COARSE_DETECTOR_DEVICE` | `coarse_detector.device` | 设备配置 |
| `OP_OCR_VL_REC_SERVER_URL` | `ocr.vl_rec_server_url` | OCR服务地址 |
| `OP_REORDER_NSP_THRESHOLD` | `reorder.nsp_threshold` | NSP阈值 |
| `OP_REORDER_MODEL_PATH` | `reorder.model_path` | 模型路径 |

---

## 6. Processor列表

| Processor | 说明 | 配置项 |
|-----------|------|--------|
| `coarse_detector` | 粗粒度版面检测 | model_name, model_dir, threshold, device |
| `sorter` | XY-Cut排序 | scale |
| `cropper` | 图片裁剪 | 无 |
| `crop_filter` | 过滤大crop | ignore_labels |
| `fine_detector` | 细粒度版面检测 | model_name, model_dir, threshold |
| `column_expander` | 列间距扩展 | threshold, depth_threshold, extra_spacing |
| `preprocessor` | 图片预处理（白边填充） | pad_crop_range |
| `ocr` | OCR识别 | vl_rec_backend, vl_rec_server_url, timeout |
| `postprocessor` | Markdown后处理 | 无 |
| `reorder` | 精排序（NSP模型） | model_path, nsp_threshold, device |
| `aggregator` | 结果聚合 | 无 |

---

## 7. 完整示例

### 示例1：基本处理

```python
from onion_parsing.core.pipeline import Pipeline

pipeline = Pipeline()
result = pipeline.process("test.pdf", "output.md")

# 输出结果
print(result["final_clean_markdown"])
```

### 示例2：自定义配置

```python
from onion_parsing.core.pipeline import Pipeline

runtime_config = {
    "coarse_detector": {
        "threshold": 0.3,
        "device": "cuda:0"
    },
    "ocr": {
        "vl_rec_server_url": "http://localhost:8118/v1",
        "timeout": 60
    },
    "reorder": {
        "nsp_threshold": 0.7,
        "left_threshold": 150
    }
}

pipeline = Pipeline(runtime_config=runtime_config)
result = pipeline.process("test.pdf", "output.md")
```

### 示例3：跳过Reorder

```python
from onion_parsing.core.pipeline import Pipeline

# 如果不需要精排序，可以跳过reorder阶段
pipeline = Pipeline(skip_stages=["reorder"])
result = pipeline.process("test.pdf", "output.md")
```

### 示例4：仅运行OCR

```python
from onion_parsing.core.pipeline import Pipeline

# 仅运行OCR和后处理
pipeline = Pipeline(
    stages=["ocr", "postprocessor", "aggregator"]
)
result = pipeline.process("test.pdf", "output.md")
```

---

## 8. 常见问题

### Q1: 如何修改版面检测阈值？

```python
runtime_config = {"coarse_detector": {"threshold": 0.3}}
pipeline = Pipeline(runtime_config=runtime_config)
```

或环境变量：
```bash
export OP_COARSE_DETECTOR_THRESHOLD=0.3
```

### Q2: 如何更换OCR服务地址？

```python
runtime_config = {"ocr": {"vl_rec_server_url": "http://127.0.0.1:8105/v1"}}
pipeline = Pipeline(runtime_config=runtime_config)
```

### Q3: 如何调整NSP阈值？

```python
runtime_config = {"reorder": {"nsp_threshold": 0.7}}
pipeline = Pipeline(runtime_config=runtime_config)
```

### Q4: 如何禁用Reorder？

```python
pipeline = Pipeline(skip_stages=["reorder"])
```

### Q5: 如何查看详细日志？

CLI方式：
```bash
python -m onion_parsing.cli.main input.pdf output.md --verbose
```

Python方式：
```python
from onion_parsing.core.logging import setup_logging
setup_logging(level="DEBUG")
```

---

## 9. 输出结果结构

```python
result = {
    "final_clean_markdown": str,  # 最终清理后的Markdown（不带bbox）
    "final_clean_markdown_with_bbox": str,  # 带 bbox 的 Markdown（用于reorder）
    "predict_md": dict,  # OCR识别结果字典 {crop_name: markdown}
    "final_img_arrays": list,  # 裁剪的图片数组列表
    "final_img_names": list,  # 裁剪的图片名称列表
    "native_results": list,  # PaddleOCRVL原生结果列表
    "all_secondary_labels": list,  # 二次切割label列表
    "bigcrop_bbox_map": dict,  # 大crop序号 → bbox坐标映射
    "img_path": str,  # 输入文件路径
    "layout_visualization": ndarray,  # 版面检测可视化（debug模式）
    "secondary_visualization": ndarray,  # 二次切割可视化（debug模式）
}
```

---

## 10. 性能优化建议

1. **调整版面检测阈值**：根据文档类型调整 `threshold`（默认0.25）
2. **调整NSP阈值**：根据文档复杂度调整 `nsp_threshold`（默认0.6）
3. **调整OCR并发数**：根据服务器性能调整 `vl_rec_max_concurrency`（默认32）
4. **跳过不必要的阶段**：使用 `skip_stages` 跳过不需要的processor
5. **使用合适的设备**：根据硬件选择 `device`（npu:0、cuda:0、cpu）

---

## 11. 注意事项

1. **Reorder阶段**：仅在 `add_bbox=True` 时执行，需要带bbox的Markdown输入
2. **模型路径**：确保模型路径正确且可访问
3. **设备配置**：确保设备可用（npu需要torch_npu，cuda需要CUDA环境）
4. **OCR服务**：确保vLLM服务正常运行且可访问
5. **配置优先级**：运行时参数 > 环境变量 > YAML配置 > 默认配置

---

## 12. 许可证

Apache License 2.0

---

**更多详细信息请参考：**
- README.md
- onion_parsing/config/default.yaml
- onion_parsing/core/pipeline.py
- onion_parsing/processors/reorder/reorder.py