# 自主AI系统

具备自主分化、迭代优化能力的多AI系统,支持视觉推演和语言交互。

## 系统特性

### 核心功能
- **自主分化**: 主AI可自动生成具有不同能力的子AI
- **迭代优化**: 基于性能评估自动进化子AI
- **智能淘汰**: 自动删除低性能子AI
- **多模态支持**: 视觉理解、语言生成、多模态融合

### 子AI能力
- **视觉推演**: 图像识别、场景预测、视觉推理
- **语言交互**: 多轮对话、文本生成、情感分析
- **多模态**: 图文理解、跨模态推理

## 系统架构

```
autonomous_ai_system/
├── main_ai.py              # 主AI核心控制器
├── visual_ai.py            # 视觉推演模块
├── language_ai.py          # 语言交互模块
├── integrated_system.py    # 集成系统与API
├── start_server.py         # 启动脚本
├── config.json             # 系统配置
└── requirements.txt        # 依赖包
```

## 安装与运行

### 1. 环境要求
- Python 3.8+
- PyTorch 2.0+
- CUDA (可选,用于GPU加速)

### 2. 安装依赖

```bash
# 自动安装依赖
python start_server.py --install

# 或手动安装
pip install -r requirements.txt
```

### 3. 启动系统

```bash
# CPU/GPU自动检测启动
python start_server.py

# 检查系统环境
python start_server.py --check-only
```

### 4. 服务地址
启动后访问: `http://localhost:5000`

## API接口

### 获取系统状态
```bash
GET /api/status
```

### 对话接口
```bash
POST /api/chat
{
  "message": "你好,介绍一下自己",
  "ai_id": "optional_agent_id"
}
```

### 文本生成
```bash
POST /api/generate
{
  "prompt": "未来AI的发展方向是",
  "max_length": 200,
  "ai_id": "optional_agent_id"
}
```

### 图像处理
```bash
POST /api/image
Content-Type: multipart/form-data
image: <file>
ai_id: optional_agent_id
```

### 情感分析
```bash
POST /api/sentiment
{
  "text": "今天天气真好!",
  "ai_id": "optional_agent_id"
}
```

### 系统进化
```bash
POST /api/evolve
```

### 删除子AI
```bash
DELETE /api/agent/<ai_id>
```

### 评估AI性能
```bash
POST /api/agent/<ai_id>/evaluate
{
  "metrics": {
    "accuracy": 0.85,
    "efficiency": 0.75,
    "innovation": 0.80,
    "stability": 0.90
  }
}
```

### 保存所有模型
```bash
POST /api/save
```

## 使用示例

### Python示例
```python
import requests

# 对话
response = requests.post('http://localhost:5000/api/chat', json={
    'message': '你好,请介绍一下你自己'
})
print(response.json())

# 文本生成
response = requests.post('http://localhost:5000/api/generate', json={
    'prompt': '人工智能的未来',
    'max_length': 150
})
print(response.json())

# 系统进化
response = requests.post('http://localhost:5000/api/evolve')
print(response.json())
```

### cURL示例
```bash
# 获取状态
curl http://localhost:5000/api/status

# 对话
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "你好"}'

# 情感分析
curl -X POST http://localhost:5000/api/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "今天很开心"}'

# 系统进化
curl -X POST http://localhost:5000/api/evolve
```

## 技术实现

### 主AI (MainAI)
- 神经网络控制器
- 子AI生命周期管理
- 性能评估与进化算法

### 视觉AI (VisualSubAI)
- CNN特征提取
- 注意力机制
- ConvLSTM时空推理
- 未来帧预测

### 语言AI (LanguageSubAI)
- Transformer架构
- 多头自注意力
- 上下文对话管理
- 温度采样生成

### 硬件支持
- **GPU**: CUDA加速 (自动检测)
- **CPU**: 多线程优化
- **混合精度**: FP16加速(可选)

## 配置说明

编辑 `config.json` 自定义系统:

```json
{
  "system": {
    "max_sub_agents": 10,        // 最大子AI数量
    "min_performance_threshold": 0.3,  // 最低性能阈值
    "mutation_rate": 0.1          // 变异率
  }
}
```

## 性能优化

### GPU优化
- 自动启用CUDA加速
- cuDNN基准优化
- 批量处理支持

### CPU优化
- 多线程并行
- 模型量化(可选)
- 内存高效加载

## 注意事项

1. 首次运行会生成初始子AI
2. 模型文件存储在 `sub_ai_agents/` 目录
3. 定期保存模型以避免丢失
4. GPU运行建议显存 > 8GB

## 扩展开发

### 添加新的AI类型
1. 在对应模块中创建新AI类
2. 在 `IntegratedAISystem` 中注册
3. 实现必要的API接口

### 自定义进化策略
修改 `MainAI.evolve_system()` 方法实现自定义优化算法。

## 许可证

MIT License
