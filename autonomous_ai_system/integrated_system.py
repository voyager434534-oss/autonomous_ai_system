"""
集成系统 - 将主AI、视觉AI、语言AI整合
提供统一的API接口和Web服务
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
from PIL import Image
import io
import base64
from typing import Dict, List
import json
from pathlib import Path

from main_ai import MainAI, SubAISpec
from visual_ai import VisualSubAI, MultimodalSubAI
from language_ai import LanguageSubAI


class IntegratedAISystem:
    """
    集成AI系统 - 统一管理所有AI模块

    功能:
    1. 主AI控制子AI生命周期
    2. 视觉AI处理图像/视频
    3. 语言AI处理文本/对话
    4. 提供统一API接口
    """

    def __init__(self):
        print("[System] 初始化集成AI系统...")

        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[System] 运行设备: {self.device}")

        # 初始化主AI
        self.main_ai = MainAI()

        # 子AI实例存储
        self.visual_agents: Dict[str, VisualSubAI] = {}
        self.language_agents: Dict[str, LanguageSubAI] = {}
        self.multimodal_agents: Dict[str, MultimodalSubAI] = {}

        # 生成初始子AI
        self._initialize_agents()

        print("[System] 系统初始化完成")

    def _initialize_agents(self):
        """初始化子AI"""
        print("[System] 生成初始子AI...")

        # 创建2个视觉AI
        for i in range(2):
            spec = self.main_ai.generate_sub_ai(task_type="visual")
            if spec:
                visual_ai = VisualSubAI(spec.architecture_config, spec.id)
                self.visual_agents[spec.id] = visual_ai

        # 创建2个语言AI
        for i in range(2):
            spec = self.main_ai.generate_sub_ai(task_type="language")
            if spec:
                lang_ai = LanguageSubAI(spec.architecture_config, spec.id)
                self.language_agents[spec.id] = lang_ai

        # 创建1个多模态AI
        spec = self.main_ai.generate_sub_ai(task_type="multimodal")
        if spec:
            mm_ai = MultimodalSubAI(spec.architecture_config, spec.id)
            self.multimodal_agents[spec.id] = mm_ai

        print(f"[System] 已创建 {len(self.visual_agents)} 个视觉AI, "
              f"{len(self.language_agents)} 个语言AI, "
              f"{len(self.multimodal_agents)} 个多模态AI")

    # ========== 视觉相关API ==========

    def process_image(self, image_data: bytes, ai_id: str = None) -> Dict:
        """
        处理图像

        Args:
            image_data: 图像二进制数据
            ai_id: 指定AI ID, None则自动选择

        Returns:
            处理结果
        """
        # 选择AI
        if ai_id is None:
            ai_id = list(self.visual_agents.keys())[0]

        if ai_id not in self.visual_agents:
            return {"error": "指定的视觉AI不存在"}

        # 保存临时图像
        temp_path = f"temp_image_{ai_id}.jpg"
        with open(temp_path, 'wb') as f:
            f.write(image_data)

        # 处理
        try:
            result = self.visual_agents[ai_id].understand_image(temp_path)
            Path(temp_path).unlink()  # 删除临时文件
            return result
        except Exception as e:
            return {"error": str(e)}

    def predict_video(self, frames_data: List[bytes], ai_id: str = None) -> Dict:
        """预测视频未来帧"""
        if ai_id is None:
            ai_id = list(self.visual_agents.keys())[0]

        if ai_id not in self.visual_agents:
            return {"error": "指定的视觉AI不存在"}

        # 转换帧数据
        frames = []
        for frame_bytes in frames_data:
            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            frames.append(frame)

        # 预测
        predicted_frames = self.visual_agents[ai_id].predict_future_frames(
            frames,
            num_future_frames=5
        )

        return {
            "predicted_frames": len(predicted_frames),
            "ai_id": ai_id
        }

    # ========== 语言相关API ==========

    def chat(self, message: str, ai_id: str = None) -> Dict:
        """
        对话接口

        Args:
            message: 用户消息
            ai_id: 指定AI ID

        Returns:
            {"response": "...", "confidence": 0.95}
        """
        if ai_id is None:
            ai_id = list(self.language_agents.keys())[0]

        if ai_id not in self.language_agents:
            return {"error": "指定的语言AI不存在"}

        response = self.language_agents[ai_id].chat(message)
        return response

    def generate_text(self, prompt: str, max_length: int = 100, ai_id: str = None) -> Dict:
        """文本生成"""
        if ai_id is None:
            ai_id = list(self.language_agents.keys())[0]

        if ai_id not in self.language_agents:
            return {"error": "指定的语言AI不存在"}

        generated = self.language_agents[ai_id].generate_text(
            prompt,
            max_length=max_length
        )

        return {
            "generated_text": generated,
            "ai_id": ai_id
        }

    def analyze_sentiment(self, text: str, ai_id: str = None) -> Dict:
        """情感分析"""
        if ai_id is None:
            ai_id = list(self.language_agents.keys())[0]

        if ai_id not in self.language_agents:
            return {"error": "指定的语言AI不存在"}

        result = self.language_agents[ai_id].understand_sentiment(text)
        return result

    # ========== 系统管理API ==========

    def get_system_status(self) -> Dict:
        """获取系统状态"""
        return self.main_ai.get_system_status()

    def evolve_system(self) -> Dict:
        """系统进化"""
        new_agents = self.main_ai.evolve_system()

        # 为新AI创建实例
        for agent_spec in new_agents:
            if agent_spec.task_specialization == "visual":
                self.visual_agents[agent_spec.id] = VisualSubAI(
                    agent_spec.architecture_config,
                    agent_spec.id
                )
            elif agent_spec.task_specialization == "language":
                self.language_agents[agent_spec.id] = LanguageSubAI(
                    agent_spec.architecture_config,
                    agent_spec.id
                )
            elif agent_spec.task_specialization == "multimodal":
                self.multimodal_agents[agent_spec.id] = MultimodalSubAI(
                    agent_spec.architecture_config,
                    agent_spec.id
                )

        return {
            "message": f"系统进化完成,生成了 {len(new_agents)} 个新子AI",
            "new_agents": [a.name for a in new_agents]
        }

    def delete_agent(self, ai_id: str) -> Dict:
        """删除指定AI"""
        success = self.main_ai.delete_sub_ai(ai_id)

        if success:
            # 从内存中移除
            if ai_id in self.visual_agents:
                del self.visual_agents[ai_id]
            if ai_id in self.language_agents:
                del self.language_agents[ai_id]
            if ai_id in self.multimodal_agents:
                del self.multimodal_agents[ai_id]

            return {"message": f"已删除AI: {ai_id}"}
        else:
            return {"error": "删除失败, AI不存在"}

    def evaluate_agent(self, ai_id: str, metrics: Dict) -> Dict:
        """评估AI性能"""
        score = self.main_ai.evaluate_sub_ai(ai_id, metrics)

        return {
            "ai_id": ai_id,
            "performance_score": score,
            "status": "good" if score > 0.7 else "needs_improvement"
        }

    def save_all_models(self):
        """保存所有模型"""
        save_dir = Path("saved_models")
        save_dir.mkdir(exist_ok=True)

        # 保存视觉AI
        for ai_id, agent in self.visual_agents.items():
            agent.save_model(str(save_dir / f"visual_{ai_id}.pt"))

        # 保存语言AI
        for ai_id, agent in self.language_agents.items():
            agent.save_model(str(save_dir / f"language_{ai_id}.pt"))

        # 保存多模态AI
        for ai_id, agent in self.multimodal_agents.items():
            torch.save(
                {"state_dict": agent.state_dict(), "ai_id": ai_id},
                str(save_dir / f"multimodal_{ai_id}.pt")
            )

        print(f"[System] 所有模型已保存到 {save_dir}")


# ========== Flask Web服务 ==========

app = Flask(__name__)
CORS(app)

# 全局系统实例
ai_system = None


@app.route('/api/status', methods=['GET'])
def get_status():
    """获取系统状态"""
    return jsonify(ai_system.get_system_status())


@app.route('/api/chat', methods=['POST'])
def chat():
    """对话接口"""
    data = request.json
    message = data.get('message', '')
    ai_id = data.get('ai_id', None)

    result = ai_system.chat(message, ai_id)
    return jsonify(result)


@app.route('/api/generate', methods=['POST'])
def generate():
    """文本生成"""
    data = request.json
    prompt = data.get('prompt', '')
    max_length = data.get('max_length', 100)
    ai_id = data.get('ai_id', None)

    result = ai_system.generate_text(prompt, max_length, ai_id)
    return jsonify(result)


@app.route('/api/image', methods=['POST'])
def process_image():
    """图像处理"""
    if 'image' not in request.files:
        return jsonify({"error": "未提供图像"}), 400

    image_file = request.files['image']
    image_data = image_file.read()
    ai_id = request.form.get('ai_id', None)

    result = ai_system.process_image(image_data, ai_id)
    return jsonify(result)


@app.route('/api/sentiment', methods=['POST'])
def sentiment():
    """情感分析"""
    data = request.json
    text = data.get('text', '')
    ai_id = data.get('ai_id', None)

    result = ai_system.analyze_sentiment(text, ai_id)
    return jsonify(result)


@app.route('/api/evolve', methods=['POST'])
def evolve():
    """系统进化"""
    result = ai_system.evolve_system()
    return jsonify(result)


@app.route('/api/agent/<ai_id>', methods=['DELETE'])
def delete_agent(ai_id):
    """删除AI"""
    result = ai_system.delete_agent(ai_id)
    return jsonify(result)


@app.route('/api/agent/<ai_id>/evaluate', methods=['POST'])
def evaluate_agent(ai_id):
    """评估AI"""
    data = request.json
    metrics = data.get('metrics', {})

    result = ai_system.evaluate_agent(ai_id, metrics)
    return jsonify(result)


@app.route('/api/save', methods=['POST'])
def save_models():
    """保存所有模型"""
    ai_system.save_all_models()
    return jsonify({"message": "所有模型已保存"})


if __name__ == "__main__":
    # 初始化系统
    ai_system = IntegratedAISystem()

    # 启动Flask服务
    print("\n[Server] 启动Web服务...")
    print("[Server] API文档:")
    print("  GET  /api/status          - 获取系统状态")
    print("  POST /api/chat             - 对话接口")
    print("  POST /api/generate         - 文本生成")
    print("  POST /api/image            - 图像处理")
    print("  POST /api/sentiment        - 情感分析")
    print("  POST /api/evolve           - 系统进化")
    print("  DELETE /api/agent/<id>     - 删除AI")
    print("  POST /api/agent/<id>/evaluate - 评估AI")
    print("  POST /api/save             - 保存模型")

    app.run(host='0.0.0.0', port=5000, debug=True)
