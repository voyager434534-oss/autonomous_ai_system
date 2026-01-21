"""
测试客户端 - 用于测试AI系统的各项功能
"""
import requests
import json
import time


class AIClient:
    """AI系统客户端"""

    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url

    def get_status(self):
        """获取系统状态"""
        response = requests.get(f"{self.base_url}/api/status")
        return response.json()

    def chat(self, message, ai_id=None):
        """对话"""
        response = requests.post(f"{self.base_url}/api/chat", json={
            "message": message,
            "ai_id": ai_id
        })
        return response.json()

    def generate_text(self, prompt, max_length=100, ai_id=None):
        """生成文本"""
        response = requests.post(f"{self.base_url}/api/generate", json={
            "prompt": prompt,
            "max_length": max_length,
            "ai_id": ai_id
        })
        return response.json()

    def analyze_sentiment(self, text, ai_id=None):
        """情感分析"""
        response = requests.post(f"{self.base_url}/api/sentiment", json={
            "text": text,
            "ai_id": ai_id
        })
        return response.json()

    def evolve_system(self):
        """系统进化"""
        response = requests.post(f"{self.base_url}/api/evolve")
        return response.json()

    def delete_agent(self, ai_id):
        """删除AI"""
        response = requests.delete(f"{self.base_url}/api/agent/{ai_id}")
        return response.json()

    def evaluate_agent(self, ai_id, metrics):
        """评估AI"""
        response = requests.post(f"{self.base_url}/api/agent/{ai_id}/evaluate", json={
            "metrics": metrics
        })
        return response.json()


def main():
    """测试主程序"""
    print("="*60)
    print("自主AI系统测试客户端")
    print("="*60)

    # 初始化客户端
    client = AIClient()

    # 1. 获取系统状态
    print("\n[测试1] 获取系统状态")
    status = client.get_status()
    print(json.dumps(status, indent=2, ensure_ascii=False))

    # 2. 对话测试
    print("\n[测试2] 对话测试")
    test_messages = [
        "你好,请介绍一下你自己",
        "你能做什么?",
        "告诉我一些关于AI的知识"
    ]

    for msg in test_messages:
        print(f"\n用户: {msg}")
        response = client.chat(msg)
        print(f"AI: {response.get('response', 'Error')}")
        print(f"置信度: {response.get('confidence', 0):.2f}")
        time.sleep(0.5)

    # 3. 文本生成测试
    print("\n[测试3] 文本生成测试")
    prompts = [
        "人工智能的未来是",
        "深度学习技术"
    ]

    for prompt in prompts:
        print(f"\n提示: {prompt}")
        response = client.generate_text(prompt, max_length=100)
        print(f"生成: {response.get('generated_text', 'Error')}")

    # 4. 情感分析测试
    print("\n[测试4] 情感分析测试")
    texts = [
        "今天天气真好,我很开心!",
        "这个产品太差了,我很失望。",
        "一般般吧,没什么特别的。"
    ]

    for text in texts:
        print(f"\n文本: {text}")
        response = client.analyze_sentiment(text)
        print(f"情感: {response.get('sentiment', 'Error')}")
        print(f"得分: {response.get('score', 0):.2f}")

    # 5. 系统进化测试
    print("\n[测试5] 系统进化测试")
    print("进化前状态:")
    status_before = client.get_status()
    print(f"子AI数量: {status_before['total_sub_agents']}")

    print("\n执行进化...")
    evolve_response = client.evolve_system()
    print(evolve_response['message'])

    print("\n进化后状态:")
    status_after = client.get_status()
    print(f"子AI数量: {status_after['total_sub_agents']}")

    # 6. AI评估测试
    print("\n[测试6] AI性能评估测试")
    agents = status_after.get('agents', [])

    if agents:
        # 评估第一个AI
        ai_id = agents[0]['id']
        print(f"评估AI: {ai_id}")

        metrics = {
            "accuracy": 0.85,
            "efficiency": 0.78,
            "innovation": 0.82,
            "stability": 0.90
        }

        response = client.evaluate_agent(ai_id, metrics)
        print(f"性能得分: {response['performance_score']:.2f}")
        print(f"状态: {response['status']}")

    # 7. 删除AI测试 (仅演示,不实际删除)
    print("\n[测试7] AI删除功能演示")
    print("说明: 删除功能已实现,为保护测试环境暂不执行")

    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("\n错误: 无法连接到服务器")
        print("请先启动服务器: python start_server.py")
    except Exception as e:
        print(f"\n错误: {e}")
