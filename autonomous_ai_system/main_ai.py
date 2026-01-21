"""
主AI核心系统 - 负责管理子AI的生成、优化和删除
具备自主迭代优化能力
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict
import uuid


@dataclass
class SubAISpec:
    """子AI规格配置"""
    id: str
    name: str
    parent_id: Optional[str]
    generation: int
    creation_time: str
    performance_score: float
    architecture_config: Dict
    task_specialization: str


class MainAI(nn.Module):
    """
    主AI类 - 核心控制器
    功能:
    1. 生成子AI
    2. 评估子AI性能
    3. 迭代优化子AI
    4. 删除低性能子AI
    """

    def __init__(self, config_path: str = "config.json"):
        super(MainAI, self).__init__()

        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[MainAI] Running on: {self.device}")

        # 核心参数
        self.sub_ai_registry: Dict[str, SubAISpec] = {}
        self.max_sub_agents = 10
        self.min_performance_threshold = 0.3
        self.mutation_rate = 0.1
        self.generation_count = 0

        # 主控神经网络 - 用于评估和决策
        self.controller = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128)  # 决策向量
        ).to(self.device)

        # 优化器
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        # 创建存储目录
        self.storage_dir = Path("sub_ai_agents")
        self.storage_dir.mkdir(exist_ok=True)

        print(f"[MainAI] 初始化完成 - 存储目录: {self.storage_dir}")

    def generate_sub_ai(self, task_type: str = "general", parent_id: Optional[str] = None) -> SubAISpec:
        """
        生成新的子AI

        Args:
            task_type: 任务类型 (visual, language, multimodal, general)
            parent_id: 父AI ID (用于继承特性)
        """
        if len(self.sub_ai_registry) >= self.max_sub_agents:
            print("[MainAI] 达到最大子AI数量限制")
            return None

        # 生成唯一ID
        ai_id = str(uuid.uuid4())[:8]

        # 确定代数
        generation = 1
        if parent_id and parent_id in self.sub_ai_registry:
            generation = self.sub_ai_registry[parent_id].generation + 1

        # 继承父代特性或随机初始化
        if parent_id and parent_id in self.sub_ai_registry:
            parent_config = self.sub_ai_registry[parent_id].architecture_config
            arch_config = self._mutate_config(parent_config)
        else:
            arch_config = self._random_architecture_config(task_type)

        # 创建子AI规格
        sub_ai_spec = SubAISpec(
            id=ai_id,
            name=f"SubAI-{ai_id}",
            parent_id=parent_id,
            generation=generation,
            creation_time=datetime.now().isoformat(),
            performance_score=0.5,  # 初始评分
            architecture_config=arch_config,
            task_specialization=task_type
        )

        # 注册到系统
        self.sub_ai_registry[ai_id] = sub_ai_spec

        # 保存配置
        self._save_sub_ai_config(sub_ai_spec)

        print(f"[MainAI] 生成新子AI: {sub_ai_spec.name} (第{generation}代)")
        return sub_ai_spec

    def _random_architecture_config(self, task_type: str) -> Dict:
        """生成随机架构配置"""
        base_configs = {
            "visual": {
                "input_channels": 3,
                "hidden_layers": [64, 128, 256, 512],
                "kernel_size": 3,
                "use_attention": True
            },
            "language": {
                "vocab_size": 50000,
                "hidden_size": 768,
                "num_layers": 12,
                "num_heads": 12,
                "use_transformer": True
            },
            "multimodal": {
                "visual_encoder": "cnn",
                "language_encoder": "transformer",
                "fusion_layer": 512,
                "cross_attention": True
            },
            "general": {
                "input_size": 512,
                "hidden_layers": [1024, 512, 256],
                "output_size": 128,
                "activation": "gelu"
            }
        }
        return base_configs.get(task_type, base_configs["general"])

    def _mutate_config(self, parent_config: Dict) -> Dict:
        """从父代配置突变"""
        new_config = parent_config.copy()

        # 随机突变某些参数
        if "hidden_layers" in new_config:
            # 以mutation_rate概率添加或移除层
            if np.random.random() < self.mutation_rate:
                layer_idx = np.random.randint(0, len(new_config["hidden_layers"]))
                change = np.random.choice([-1, 1])
                new_val = max(32, new_config["hidden_layers"][layer_idx] + change * 32)
                new_config["hidden_layers"][layer_idx] = new_val

        if "hidden_size" in new_config:
            if np.random.random() < self.mutation_rate:
                new_config["hidden_size"] = max(128, new_config["hidden_size"] + np.random.choice([-64, 64]))

        return new_config

    def evaluate_sub_ai(self, ai_id: str, performance_metrics: Dict) -> float:
        """
        评估子AI性能

        Args:
            ai_id: 子AI ID
            performance_metrics: 性能指标字典

        Returns:
            综合性能评分 (0-1)
        """
        if ai_id not in self.sub_ai_registry:
            return 0.0

        # 计算加权得分
        weights = {
            "accuracy": 0.4,
            "efficiency": 0.2,
            "innovation": 0.2,
            "stability": 0.2
        }

        score = sum(
            performance_metrics.get(k, 0) * v
            for k, v in weights.items()
        )

        # 更新注册表
        self.sub_ai_registry[ai_id].performance_score = score
        self._save_sub_ai_config(self.sub_ai_registry[ai_id])

        return score

    def evolve_system(self) -> List[SubAISpec]:
        """
        系统自主迭代优化 - 淘汰低性能AI，生成新AI
        """
        print("[MainAI] 开始系统迭代优化...")

        # 按性能排序
        sorted_agents = sorted(
            self.sub_ai_registry.values(),
            key=lambda x: x.performance_score,
            reverse=True
        )

        new_agents = []

        # 淘汰低性能AI
        for agent in sorted_agents:
            if agent.performance_score < self.min_performance_threshold:
                self.delete_sub_ai(agent.id)
                print(f"[MainAI] 淘汰低性能AI: {agent.name} (得分: {agent.performance_score:.3f})")

        # 从高性能AI生成新子代
        top_performers = [a for a in sorted_agents if a.performance_score > 0.7]

        for parent in top_performers[:3]:  # 只保留前3名
            if len(self.sub_ai_registry) < self.max_sub_agents:
                new_agent = self.generate_sub_ai(
                    task_type=parent.task_specialization,
                    parent_id=parent.id
                )
                if new_agent:
                    new_agents.append(new_agent)

        self.generation_count += 1
        print(f"[MainAI] 迭代完成 - 第{self.generation_count}轮")

        return new_agents

    def delete_sub_ai(self, ai_id: str) -> bool:
        """删除子AI"""
        if ai_id not in self.sub_ai_registry:
            return False

        # 删除配置文件
        config_path = self.storage_dir / f"{ai_id}_config.json"
        if config_path.exists():
            config_path.unlink()

        # 删除模型文件
        model_path = self.storage_dir / f"{ai_id}_model.pt"
        if model_path.exists():
            model_path.unlink()

        # 从注册表移除
        del self.sub_ai_registry[ai_id]

        print(f"[MainAI] 已删除子AI: {ai_id}")
        return True

    def _save_sub_ai_config(self, spec: SubAISpec):
        """保存子AI配置"""
        config_path = self.storage_dir / f"{spec.id}_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(spec), f, indent=2, ensure_ascii=False)

    def get_system_status(self) -> Dict:
        """获取系统状态"""
        return {
            "total_sub_agents": len(self.sub_ai_registry),
            "generation": self.generation_count,
            "device": str(self.device),
            "agents": [
                {
                    "id": spec.id,
                    "name": spec.name,
                    "performance": spec.performance_score,
                    "generation": spec.generation
                }
                for spec in self.sub_ai_registry.values()
            ]
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.controller(x)


if __name__ == "__main__":
    # 测试主AI
    main_ai = MainAI()

    # 生成初始子AI
    for i in range(5):
        task_types = ["visual", "language", "multimodal", "general"]
        main_ai.generate_sub_ai(task_type=np.random.choice(task_types))

    # 查看状态
    status = main_ai.get_system_status()
    print(json.dumps(status, indent=2, ensure_ascii=False))
