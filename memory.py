"""记忆管理 — ChromaDB 优先，Milvus 预留接口"""

import json
import os
import uuid
from typing import List, Optional

import yaml


class MemoryManager:
    """对话记忆管理器"""

    def __init__(self, config: dict):
        self.config = config
        self.backend = config["memory"]["backend"]
        self.persist_dir = config["memory"]["persist_dir"]
        self.collection_name = config["memory"]["collection_name"]
        self._client = None
        self._collection = None
        self._init_backend()

    def _init_backend(self):
        """初始化存储后端"""
        os.makedirs(self.persist_dir, exist_ok=True)

        try:
            if self.backend == "chromadb":
                import chromadb
                self._client = chromadb.PersistentClient(path=self.persist_dir)
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            elif self.backend == "milvus":
                # Milvus 预留接口
                raise NotImplementedError("Milvus backend not implemented yet")
            elif self.backend == "none":
                # 内存后端，不持久化
                self._memory = []
        except Exception as e:
            print(f"[警告] 记忆后端初始化失败: {e}，使用内存后端")
            self.backend = "none"
            self._memory = []

    def store(self, conversation: dict):
        """存储对话片段"""
        content = conversation.get("content", "")
        role = conversation.get("role", "unknown")
        metadata = {
            "role": role,
            "timestamp": conversation.get("timestamp", ""),
            "session_id": conversation.get("session_id", "default"),
        }

        if self.backend == "chromadb":
            doc_id = f"{role}_{uuid.uuid4().hex[:8]}"
            self._collection.add(
                documents=[content],
                metadatas=[metadata],
                ids=[doc_id],
            )
        elif self.backend == "none":
            self._memory.append({"content": content, "metadata": metadata})

    def search(self, query: str, top_k: int = 5, session_id: str = "default") -> List[dict]:
        """语义检索相关历史"""
        if self.backend == "chromadb":
            results = self._collection.query(
                query_texts=[query],
                n_results=top_k,
                where={"session_id": session_id} if session_id != "default" else None,
            )
            conversations = []
            if results and results["documents"]:
                for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                    conversations.append({
                        "role": meta.get("role", "unknown"),
                        "content": doc,
                        "metadata": meta,
                    })
            return conversations
        elif self.backend == "none":
            # 返回最近的消息，不进行语义搜索
            filtered = [m for m in self._memory if m["metadata"].get("session_id") == session_id]
            return [{"role": m["metadata"].get("role", "unknown"), "content": m["content"]} for m in filtered[-top_k:]]
        return []

    def get_history(self, session_id: str = "default") -> List[dict]:
        """获取会话历史"""
        if self.backend == "chromadb":
            results = self._collection.get(
                where={"session_id": session_id} if session_id != "default" else None,
            )
            conversations = []
            if results and results["documents"]:
                for doc, meta in zip(results["documents"], results["metadatas"]):
                    conversations.append({
                        "role": meta.get("role", "unknown"),
                        "content": doc,
                        "metadata": meta,
                    })
            return conversations
        elif self.backend == "none":
            filtered = [m for m in self._memory if m["metadata"].get("session_id") == session_id]
            return [{"role": m["metadata"].get("role", "unknown"), "content": m["content"]} for m in filtered]
        return []


def create_memory(config: dict) -> MemoryManager:
    """工厂函数创建 MemoryManager"""
    return MemoryManager(config)
