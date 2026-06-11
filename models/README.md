# 模型部署说明

## 目录结构

```
models/
├── README.md              # 本文件
└── minimind-fc/           # 训练完成后将权重放于此
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── ...
```

## 部署步骤

### 1. 训练模型

按照 MiniMind 训练流程完成 Function Calling 专项训练。

### 2. 放置权重

将训练产出的完整模型目录复制到 `models/minimind-fc/`：

```bash
# 假设训练输出在 output/fc_dpo/final_model/
cp -r output/fc_dpo/final_model/* models/minimind-fc/
```

确保目录包含：
- `config.json` — 模型配置
- `model.safetensors` — 模型权重
- `tokenizer.json` — 分词器
- `tokenizer_config.json` — 分词器配置

### 3. 修改配置

编辑 `config.yaml`：

```yaml
model:
  mock: false                           # 关闭 Mock
  model_path: "./models/minimind-fc"    # 指向权重
  backend: "directml"                   # directml | onnx | cpu
  max_tokens: 2048
  temperature: 0.1
```

### 4. 重启服务

```bash
start_server.bat
```

### 5. 验证

```bash
python test_api.py
```

期望输出 `7/7 通过`。

## Mock 模式 vs 真实模型

| | Mock 模式 | 真实模型 |
|------|---------|---------|
| 配置 | `mock: true` | `mock: false` |
| 推理 | 预设固定回复 | 真实 AI 推理 |
| 工具选择 | 关键词匹配 | 语义理解 |
| GPU | 不需要 | 建议使用 |
| 用途 | 开发验证 | 生产上线 |

## 常见问题

**Q：启动报错 "模型路径不存在"**

检查 `config.yaml` 中 `model_path` 是否正确，以及 `models/minimind-fc/` 目录是否存在。

**Q：DirectML 报错**

尝试切换后端：
```yaml
backend: "cpu"  # 先用 CPU 验证模型加载正常
```

**Q：推理速度慢**

- 检查是否成功启用 DirectML
- 减小 `max_tokens`
- 考虑 ONNX Runtime 优化
