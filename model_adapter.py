"""真实模型适配器 — 封装 MiniMind 模型，暴露与 MockModel 一致的 generate()/stream_generate() 接口"""

import json
import re
import time
from typing import List, Optional, Union


class RealModel:
    """
    真实 MiniMind 模型适配器

    外部接口与 MockModel 完全一致：
    - generate(messages, tools) → str | dict
    - stream_generate(messages, tools) → generator
    - is_loaded 属性

    内部通过 tokenizer + MiniMindForCausalLM 实现真实推理。
    """

    def __init__(self, config: dict):
        self.config = config
        self.model_path = config.get("model_path", "")
        self.max_tokens = config.get("max_tokens", 2048)
        self.temperature = config.get("temperature", 0.7)
        self.top_p = config.get("top_p", 0.85)
        self.top_k = config.get("top_k", 50)
        self._model = None
        self._tokenizer = None
        self._device = "cpu"
        self._loaded = False

        self._load_model()

    def _detect_device(self):
        """检测可用的 GPU 设备类型"""
        import torch
        if torch.cuda.is_available():
            # ROCm 在 Windows 上伪装成 CUDA，用 torch.version.hip 区分
            if torch.version.hip is not None:
                return "rocm", torch.float16
            return "cuda", torch.float16
        try:
            import torch_directml
            torch_directml.device()
            return "directml", torch.float32
        except ImportError:
            pass
        return "cpu", torch.float32

    def _load_model(self):
        """加载 tokenizer 和模型，支持两种格式：
        - 目录模式：model_path 指向目录，自动找 .pth 文件
        - 文件模式：model_path 直接指向 .pth 文件
        """
        try:
            import os
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            device_type, dtype = self._detect_device()
            print(f"[RealModel] 检测到设备: {device_type}")

            # 判断加载格式
            if os.path.isdir(self.model_path):
                # 目录模式：找 .pth 文件
                pth_files = [f for f in os.listdir(self.model_path) if f.endswith(".pth")]
                if pth_files:
                    # 优先选 full_sft_*.pth
                    pth_name = next((f for f in pth_files if f.startswith("full_sft")), pth_files[0])
                    self.model_path = os.path.join(self.model_path, pth_name)
                self._load_pth_model(device_type, dtype)
            elif self.model_path.endswith(".pth"):
                self._load_pth_model(device_type, dtype)
            else:
                self._load_hf_model(device_type, dtype)

            self._model.eval()
            self._device = device_type
            self._loaded = True
            print(f"[RealModel] 模型加载完成 (设备: {device_type})")
        except Exception as e:
            print(f"[RealModel] 模型加载失败: {e}")
            self._loaded = False

    def _load_hf_model(self, device_type, dtype):
        """加载 HuggingFace 格式模型"""
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print(f"[RealModel] 加载 tokenizer: {self.model_path}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )

        print(f"[RealModel] 加载模型权重: {self.model_path}")
        load_kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
        if device_type in ("cuda", "rocm"):
            load_kwargs["device_map"] = "auto"

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path, **load_kwargs
        )

        if device_type == "directml":
            import torch_directml
            self._model = self._model.to(torch_directml.device())

    def _load_pth_model(self, device_type, dtype):
        """加载 MiniMind .pth 格式模型（底层 Qwen3 架构）

        流程：
        1. AutoConfig 读 config.json
        2. AutoModelForCausalLM.from_config() 创建 Qwen3 架构
        3. 加载 .pth 权重
        """
        import os
        import torch
        from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

        model_dir = os.path.dirname(os.path.abspath(self.model_path))

        # 1. 加载 tokenizer
        tokenizer_path = model_dir
        if not os.path.exists(os.path.join(tokenizer_path, "tokenizer_config.json")):
            tokenizer_path = os.path.join(model_dir, "tokenizer.json")
        print(f"[RealModel] 加载 tokenizer: {tokenizer_path}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )

        # 2. 读 config.json，创建 Qwen3 模型架构
        print(f"[RealModel] 读取 config: {model_dir}")
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

        # 3. 加载 .pth 权重
        print(f"[RealModel] 加载权重: {self.model_path}")
        state_dict = torch.load(self.model_path, map_location="cpu", weights_only=True)
        self._model.load_state_dict(state_dict, strict=True)

        # 4. 移到设备
        if device_type in ("cuda", "rocm"):
            self._model = self._model.half().cuda()
        elif device_type == "directml":
            import torch_directml
            self._model = self._model.float().to(torch_directml.device())

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def generate(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
    ) -> Union[str, dict]:
        """
        核心推理接口

        返回值：
        - 无 tools：返回 str（纯文本）
        - 有 tools：返回 dict，如 {"name": "get_time", "arguments": {}}
        """
        if not self._loaded:
            return "模型未加载，请检查 model_path 配置"

        # Step 1: 消息 → 带特殊标记的文本字符串
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Step 2: 文本 → token ID 张量
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)

        # 移动到模型所在设备
        import torch
        if self._device == "directml":
            import torch_directml
            device = torch_directml.device()
        elif self._device in ("cuda", "rocm"):
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Step 3: 模型推理（自回归生成）
        import torch
        with torch.inference_mode():
            generated = self._model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                eos_token_id=self._tokenizer.eos_token_id,
                do_sample=True,
                repetition_penalty=1.0,
            )

        # Step 4: token IDs → 文本
        new_tokens = generated[0][input_ids.shape[1]:]
        response_text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Step 5: 如果有工具，解析工具调用
        if tools:
            tool_call = self._parse_tool_call(response_text, tools)
            if tool_call:
                return tool_call

        return response_text

    def stream_generate(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
    ):
        """流式生成，yield 每个 token"""
        if not self._loaded:
            yield {"type": "chunk", "data": "模型未加载，请检查 model_path 配置"}
            yield {"type": "done", "data": ""}
            return

        # 有工具时不支持流式，直接返回完整结果
        if tools:
            result = self.generate(messages, tools=tools)
            yield {"type": "tool_calls", "data": result}
            return

        # Step 1: 消息 → 文本
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Step 2: 文本 → token IDs
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)

        # 移动到模型所在设备
        import torch
        if self._device == "directml":
            import torch_directml
            device = torch_directml.device()
        elif self._device in ("cuda", "rocm"):
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Step 3: 使用 TextIteratorStreamer 逐 token 输出
        from transformers import TextIteratorStreamer
        from threading import Thread

        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        generate_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=1.0,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
            streamer=streamer,
        )

        thread = Thread(target=self._model.generate, kwargs=generate_kwargs)
        thread.start()

        # 逐 token yield
        for token_text in streamer:
            if token_text:
                yield {"type": "chunk", "data": token_text}

        thread.join()
        yield {"type": "done", "data": ""}

    def _parse_tool_call(self, text: str, tools: List[dict]) -> Optional[dict]:
        """
        解析模型输出中的工具调用

        MiniMind SFT 训练数据中的工具调用格式：
        <tool_call>
        {"name": "get_time", "arguments": {"timezone": "UTC"}}
        </tool_call>
        """
        pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            return None

        try:
            tool_call = json.loads(match.group(1))
            name = tool_call.get("name", "")
            arguments = tool_call.get("arguments", {})

            # 验证工具名是否在可用工具列表中
            available_names = [
                t.get("function", {}).get("name", "") for t in tools
            ]
            if name in available_names:
                return {"name": name, "arguments": arguments}
        except json.JSONDecodeError:
            pass

        return None
