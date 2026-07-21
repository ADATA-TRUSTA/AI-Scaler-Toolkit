"""
GPT Model Harmony Parser - Handles OpenAI GPT-OSS model response parsing
Uses openai-harmony library to parse structured responses with channels (analysis, commentary, final)
"""
import logging
import time
import os
from typing import List, Dict, Any, Optional, Generator

# Import TIKTOKEN_CACHE_DIR from settings (must be imported early)
import sys
from pathlib import Path
# Add parent directory to path to import settings
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from service.settings import TIKTOKEN_CACHE_DIR

logger = logging.getLogger(__name__)

# Try to import openai-harmony
try:
    from openai_harmony import (
        load_harmony_encoding,
        HarmonyEncodingName,
        Role,
        Message,
        Conversation,
    )
    HARMONY_AVAILABLE = True
    logger.info("✅ openai-harmony library available")
except ImportError:
    HARMONY_AVAILABLE = False
    logger.warning("⚠️  openai-harmony not available. Install with: pip install openai-harmony")

def is_gpt_model(model_name: str) -> bool:
    """
    檢測模型是否為 OpenAI GPT-OSS 系列模型
    
    Args:
        model_name: 模型名稱或路徑
    
    Returns:
        True 如果是 GPT-OSS 模型
    """
    if not model_name:
        return False
    
    model_name_lower = model_name.lower()
    
    # 檢查 GPT-OSS 模型名稱模式
    gpt_patterns = [
        "gpt-oss",
        "openai/gpt",
        "gpt_oss",
    ]
    
    return any(pattern in model_name_lower for pattern in gpt_patterns)


class TokenIDStreamer:
    """Streamer that returns token IDs instead of decoded text for GPT StreamableParser
    
    Note: This streamer is designed to work with generation threads that may fail.
    It will timeout and raise StopIteration if no tokens are received for too long,
    allowing the main loop to check for generation exceptions.
    """
    def __init__(self, skip_prompt: bool = True, timeout: float = 30.0):
        self.token_queue = []
        self.stop = False
        self.skip_prompt = skip_prompt
        self.prompt_skipped = False
        self.timeout = timeout  # Timeout in seconds for waiting for next token
        self.last_token_time = None
        self.exception = None  # Store exception from generation thread
    
    def put(self, value):
        if value is None:
            self.stop = True
            return
        
        # Skip the first batch (prompt) if skip_prompt is True
        if self.skip_prompt and not self.prompt_skipped:
            self.prompt_skipped = True
            return
        
        # value is token_ids tensor, could be [batch, seq] or [seq]
        if hasattr(value, 'tolist'):
            ids = value.tolist()
            # Handle batch dimension
            if isinstance(ids, list):
                if isinstance(ids[0], list):
                    # [[token_ids...]]
                    for token_id in ids[0]:
                        self.token_queue.append(token_id)
                else:
                    # [token_ids...]
                    for token_id in ids:
                        self.token_queue.append(token_id)
        else:
            # Direct integer
            self.token_queue.append(value)
        
        # Update last token time whenever we receive tokens
        self.last_token_time = time.time()
    
    def end(self):
        self.stop = True
    
    def set_exception(self, exception: Exception):
        """Allow generation thread to signal an exception"""
        self.exception = exception
        self.stop = True
    
    def __iter__(self):
        return self
    
    def __next__(self):
        # Initialize last_token_time on first call
        if self.last_token_time is None:
            self.last_token_time = time.time()
        
        # Wait for token or stop signal with timeout check
        wait_start = time.time()
        while not self.token_queue:
            # Check if exception was set by generation thread
            if self.exception is not None:
                logger.error(f"[TokenIDStreamer] Generation exception detected: {self.exception}")
                raise self.exception
            
            # Check for stop signal
            if self.stop:
                raise StopIteration
            
            # Check for timeout (no tokens received for too long)
            elapsed = time.time() - self.last_token_time
            if elapsed > self.timeout:
                logger.error(f"[TokenIDStreamer] Timeout after {elapsed:.1f}s waiting for next token")
                raise TimeoutError(f"TokenIDStreamer timeout after {elapsed:.1f}s - generation thread may have failed")
            
            time.sleep(0.001)
        
        # Update last token time when we successfully return a token
        self.last_token_time = time.time()
        return self.token_queue.pop(0)


def is_harmony_available() -> bool:
    """檢查 openai-harmony 庫是否可用"""
    return HARMONY_AVAILABLE


class GPTResponseParser:
    """GPT 響應解析器 - 處理 Harmony 格式的結構化響應"""
    
    def __init__(self, model_name: str):
        """
        初始化 GPT 響應解析器
        
        Args:
            model_name: 模型名稱（用於檢測）
        """
        self.model_name = model_name
        self.is_gpt = is_gpt_model(model_name)
        
        if self.is_gpt and HARMONY_AVAILABLE:
            self.enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
            logger.info(f"🚀 GPT Harmony Parser initialized for model: {model_name}")
        else:
            self.enc = None
            if self.is_gpt:
                logger.warning(f"⚠️  Model {model_name} is GPT but harmony library not available")
    
    def should_parse(self) -> bool:
        """是否應該使用 Harmony 解析"""
        return self.is_gpt and self.enc is not None
    
    def render_conversation_for_completion(self, messages: List[Dict[str, str]]) -> List[int]:
        """
        使用 Harmony 格式渲染對話為 tokens
        
        Args:
            messages: 對話訊息列表，格式 [{"role": "system", "content": "..."}, ...]
        
        Returns:
            token IDs 列表
        """
        if not self.should_parse():
            raise ValueError("Harmony parser not available or not a GPT model")
        
        try:
            # 轉換為 Harmony Message 格式
            harmony_messages = []
            for msg in messages:
                role_str = msg.get("role", "user")
                content = msg.get("content", "")
                
                # 映射角色
                if role_str == "system":
                    role = Role.SYSTEM
                elif role_str == "assistant":
                    role = Role.ASSISTANT
                else:
                    role = Role.USER
                
                harmony_messages.append(
                    Message.from_role_and_content(role, content)
                )
            
            # 建立對話
            convo = Conversation.from_messages(harmony_messages)
            
            # 渲染為 tokens
            input_tokens = self.enc.render_conversation_for_completion(convo, Role.ASSISTANT)
            
            logger.debug(f"Rendered {len(messages)} messages to {len(input_tokens)} tokens")
            return input_tokens
            
        except Exception as e:
            logger.error(f"Failed to render conversation: {e}")
            raise
    
    def parse_generated_tokens(self, token_ids: List[int], strict: bool = False) -> Dict[str, Any]:
        """
        解析生成的 tokens 為結構化響應，將 analysis/commentary 包裹在 <think></think> 標籤中
        
        Args:
            token_ids: 生成的 token IDs
            strict: 是否嚴格解析
        
        Returns:
            結構化響應字典:
            {
                "formatted_text": str,  # 格式化後的完整文本（thinking 用 <think></think> 包裹）
                "parsed": bool          # 是否成功解析
            }
        """
        if not self.should_parse():
            return {
                "formatted_text": "",
                "parsed": False
            }
        
        try:
            # 使用官方 API 解析生成的 tokens
            parsed_messages = self.enc.parse_messages_from_completion_tokens(
                token_ids,
                role=Role.ASSISTANT,
                strict=strict
            )
            
            formatted_parts = []
            
            # 遍歷解析後的訊息
            for i, msg in enumerate(parsed_messages):
                try:
                    if hasattr(msg, 'content'):
                        contents = msg.content if isinstance(msg.content, list) else [msg.content]
                        
                        for content in contents:
                            # 獲取文本和頻道信息
                            text = None
                            channel = None
                            
                            if hasattr(content, 'text'):
                                text = content.text
                            if hasattr(content, 'channel'):
                                channel = content.channel
                            
                            # 根據頻道格式化
                            if text:
                                # 思考頻道（analysis 或 commentary）用 <think></think> 包裹
                                if channel in ['analysis', 'commentary']:
                                    formatted_parts.append(f"<think>\n{text}\n</think>")
                                # 正式回覆（final 頻道）直接輸出
                                elif channel == 'final':
                                    formatted_parts.append(text)
                                # 沒有明確頻道的第一條訊息視為思考
                                elif not channel and i == 0:
                                    formatted_parts.append(f"<think>\n{text}\n</think>")
                                # 其他情況直接輸出
                                else:
                                    formatted_parts.append(text)
                            
                            # 處理其他可能的屬性
                            if hasattr(content, 'analysis') and content.analysis:
                                formatted_parts.append(f"<think>\n{content.analysis}\n</think>")
                            
                            if hasattr(content, 'final') and content.final:
                                formatted_parts.append(content.final)
                                
                except Exception as e:
                    logger.debug(f"Error parsing message {i+1}: {e}")
                    continue
            
            # 組合結果
            formatted_text = "\n\n".join(formatted_parts)
            
            return {
                "formatted_text": formatted_text,
                "parsed": True
            }
            
        except Exception as e:
            logger.warning(f"Failed to parse GPT response: {e}")
            return {
                "formatted_text": "",
                "parsed": False
            }


def create_gpt_parser(model_name: str) -> Optional[GPTResponseParser]:
    """
    創建 GPT 解析器實例（工廠函數）
    
    Args:
        model_name: 模型名稱
    
    Returns:
        GPTResponseParser 實例，如果不是 GPT 模型或庫不可用則返回 None
    """
    if not is_gpt_model(model_name):
        return None
    
    if not HARMONY_AVAILABLE:
        logger.warning(f"GPT model detected ({model_name}) but openai-harmony not available")
        return None
    
    return GPTResponseParser(model_name)


def create_stream_parser(model_name: str):
    """
    創建 GPT 串流解析器（使用官方 StreamableParser）
    
    Args:
        model_name: 模型名稱
    
    Returns:
        StreamableParser 實例，如果不是 GPT 模型或庫不可用則返回 None
    """
    if not is_gpt_model(model_name):
        return None
    
    if not HARMONY_AVAILABLE:
        logger.warning(f"GPT model detected ({model_name}) but openai-harmony not available")
        return None
    
    try:
        from openai_harmony import StreamableParser
        enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        stream_parser = StreamableParser(enc, role=Role.ASSISTANT)
        logger.info(f"🚀 GPT StreamableParser initialized for model: {model_name}")
        return stream_parser
    except Exception as e:
        logger.error(f"Failed to create StreamableParser: {e}")
        return None