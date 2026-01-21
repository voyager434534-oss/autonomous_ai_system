"""
语言输出与交互模块 - 基于Transformer的语言生成系统
支持对话、文本生成、多轮交互
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import json
import re
from collections import deque


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.shape

        # 生成Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        return self.proj(output)


class TransformerBlock(nn.Module):
    """Transformer块"""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # 自注意力 + 残差连接
        attn_output = self.attention(x, mask)
        x = self.norm1(x + attn_output)

        # 前馈网络 + 残差连接
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)

        return x


class LanguageSubAI(nn.Module):
    """
    语言子AI - 具备自然语言理解与生成能力

    功能:
    1. 多轮对话
    2. 文本生成
    3. 上下文理解
    4. 情感分析
    """

    def __init__(self, config: Dict, ai_id: str):
        super(LanguageSubAI, self).__init__()

        self.ai_id = ai_id
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 模型参数
        self.vocab_size = config.get("vocab_size", 50000)
        self.hidden_size = config.get("hidden_size", 768)
        self.num_layers = config.get("num_layers", 12)
        self.num_heads = config.get("num_heads", 12)
        self.max_seq_length = 512

        # 词嵌入
        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.dropout = nn.Dropout(0.1)

        # Transformer层
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                self.hidden_size,
                self.num_heads,
                self.hidden_size * 4,
                0.1
            )
            for _ in range(self.num_layers)
        ]).to(self.device)

        # 输出头
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size)

        # 对话历史
        self.conversation_history = deque(maxlen=10)

        print(f"[LanguageSubAI {ai_id}] 初始化完成")

    def generate_text(self,
                     prompt: str,
                     max_length: int = 100,
                     temperature: float = 0.8,
                     top_k: int = 50) -> str:
        """
        生成文本

        Args:
            prompt: 输入提示
            max_length: 最大生成长度
            temperature: 采样温度 (越高越随机)
            top_k: top-k采样

        Returns:
            生成的文本
        """
        # 这里简化处理，实际需要tokenizer
        # 模拟tokenization
        tokens = self._mock_tokenize(prompt)

        # 生成
        generated_tokens = []

        for _ in range(max_length):
            # 转换为tensor
            input_ids = torch.tensor([tokens + generated_tokens]).to(self.device)

            # 创建因果mask
            seq_len = input_ids.size(1)
            mask = torch.tril(torch.ones(seq_len, seq_len)).to(self.device)

            # 前向传播
            embeddings = self._forward_with_embeddings(input_ids, mask)

            # 获取下一个token的logits
            logits = self.lm_head(embeddings[:, -1, :])

            # 温度采样
            logits = logits / temperature

            # Top-k过滤
            if top_k > 0:
                values, indices = torch.topk(logits, top_k)
                logits = torch.full_like(logits, -float('inf'))
                logits.scatter_(1, indices, values)

            # 采样
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            generated_tokens.append(next_token)

            # 停止条件
            if next_token == 2:  # EOS token
                break

        # 解码
        generated_text = self._mock_decode(generated_tokens)

        return generated_text

    def _forward_with_embeddings(self, input_ids, mask=None):
        """前向传播获取嵌入"""
        batch_size, seq_len = input_ids.shape

        # Token嵌入
        token_embeds = self.token_embedding(input_ids)

        # 位置嵌入
        positions = torch.arange(seq_len, device=self.device).unsqueeze(0)
        position_embeds = self.position_embedding(positions)

        # 组合
        x = self.dropout(token_embeds + position_embeds)

        # Transformer层
        for block in self.transformer_blocks:
            x = block(x, mask)

        return x

    def chat(self, user_input: str, context: Optional[List[Dict]] = None) -> Dict:
        """
        多轮对话

        Args:
            user_input: 用户输入
            context: 对话上下文 [{"role": "user", "content": "..."}]

        Returns:
            {"response": "...", "confidence": 0.95}
        """
        # 构建对话历史
        if context is None:
            context = []

        # 添加当前输入到历史
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })

        # 构建完整prompt
        conversation_prompt = self._build_conversation_prompt()

        # 生成回复
        response = self.generate_text(
            conversation_prompt,
            max_length=150,
            temperature=0.8
        )

        # 计算置信度 (简化)
        confidence = self._calculate_confidence(response)

        # 添加到历史
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })

        return {
            "response": response,
            "confidence": confidence,
            "ai_id": self.ai_id
        }

    def _build_conversation_prompt(self) -> str:
        """构建对话prompt"""
        prompt = "以下是对话历史:\n"

        for turn in self.conversation_history:
            role = "用户" if turn["role"] == "user" else "助手"
            prompt += f"{role}: {turn['content']}\n"

        prompt += "助手: "
        return prompt

    def _calculate_confidence(self, response: str) -> float:
        """计算回复置信度"""
        # 简化: 基于响应长度和结构
        base_confidence = 0.7

        if len(response) > 10:
            base_confidence += 0.1

        # 检查是否有标点符号
        if any(punct in response for punct in ['。', '.', '!', '？', '?']):
            base_confidence += 0.1

        return min(base_confidence, 1.0)

    def understand_sentiment(self, text: str) -> Dict:
        """情感分析"""
        # 简化实现
        positive_words = ['好', '棒', '优秀', '喜欢', '高兴']
        negative_words = ['差', '坏', '讨厌', '难过', '糟糕']

        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)

        if pos_count > neg_count:
            sentiment = "positive"
            score = (pos_count - neg_count) / max(len(text), 1)
        elif neg_count > pos_count:
            sentiment = "negative"
            score = (neg_count - pos_count) / max(len(text), 1)
        else:
            sentiment = "neutral"
            score = 0.5

        return {
            "sentiment": sentiment,
            "score": score,
            "confidence": max(0.5, 1.0 - abs(0.5 - score))
        }

    def summarize_text(self, text: str, max_summary_length: int = 50) -> str:
        """文本摘要"""
        sentences = re.split(r'[。.!?！？]', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= 2:
            return text

        # 简单提取前两句作为摘要
        summary = '。'.join(sentences[:2]) + '。'

        if len(summary) > max_summary_length:
            summary = summary[:max_summary_length] + '...'

        return summary

    def _mock_tokenize(self, text: str) -> List[int]:
        """模拟分词 (实际应使用tokenizer)"""
        # 简单字符级编码
        return [ord(c) % self.vocab_size for c in text[:self.max_seq_length]]

    def _mock_decode(self, tokens: List[int]) -> str:
        """模拟解码"""
        return ''.join(chr(t) for t in tokens if t < 128)

    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'ai_id': self.ai_id,
            'conversation_history': list(self.conversation_history)
        }, path)
        print(f"[LanguageSubAI {self.ai_id}] 模型已保存到 {path}")

    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.conversation_history = deque(
            checkpoint.get('conversation_history', []),
            maxlen=10
        )
        print(f"[LanguageSubAI {self.ai_id}] 模型已从 {path} 加载")

    def clear_conversation(self):
        """清空对话历史"""
        self.conversation_history.clear()
        print(f"[LanguageSubAI {self.ai_id}] 对话历史已清空")


if __name__ == "__main__":
    # 测试语言AI
    config = {
        "vocab_size": 50000,
        "hidden_size": 768,
        "num_layers": 6,
        "num_heads": 8
    }

    lang_ai = LanguageSubAI(config, "test_lang_ai")
    print(f"\n语言AI模型参数数量: {sum(p.numel() for p in lang_ai.parameters()):,}")

    # 测试对话
    response = lang_ai.chat("你好,请介绍一下你自己")
    print(f"\nAI回复: {response['response']}")
    print(f"置信度: {response['confidence']:.2f}")
