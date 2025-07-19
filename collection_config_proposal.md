# Collection-Embedding 綁定架構提案

## 問題描述
目前系統使用全域 embedding 設定，導致：
1. 無法支援多種 embedding 模型
2. Collection 與 embedding 模型無法對應
3. 搜尋時可能用錯模型

## 解決方案

### 1. Collection 配置文件
```json
{
  "collections": {
    "default": {
      "embedding_provider": "fastembed",
      "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
      "vector_name": "fast-all-minilm-l6-v2",
      "vector_size": 384
    },
    "modern_docs": {
      "embedding_provider": "ollama", 
      "embedding_model": "nomic-embed-text",
      "vector_name": "nomic-embed-text",
      "vector_size": 768,
      "ollama_base_url": "http://localhost:11434"
    }
  }
}
```

### 2. 動態 Embedding Provider
```python
class CollectionAwareQdrantConnector:
    def __init__(self):
        self.collection_configs = load_collection_configs()
        self.embedding_providers = {}
    
    def get_embedding_provider(self, collection_name: str):
        if collection_name in self.embedding_providers:
            return self.embedding_providers[collection_name]
            
        config = self.collection_configs[collection_name]
        provider = create_embedding_provider(config)
        self.embedding_providers[collection_name] = provider
        return provider
    
    async def search(self, query: str, collection_name: str):
        provider = self.get_embedding_provider(collection_name)
        query_vector = await provider.embed_query(query)
        vector_name = provider.get_vector_name()
        # 使用正確的 embedding 進行搜尋
```

### 3. MCP 工具改進
- `qdrant-find` 自動偵測 collection 對應的 embedding 模型
- `qdrant-store` 根據目標 collection 使用正確的 embedding
- 新增 `qdrant-collection-info` 顯示每個 collection 的 embedding 設定

### 4. 向下相容性
- 保持現有 .env 設定作為預設值
- 未配置的 collection 使用全域設定
- 逐步遷移現有 collections

## 實作步驟

1. **階段一**：新增 collection 配置系統
2. **階段二**：修改 QdrantConnector 支援動態 embedding
3. **階段三**：更新 MCP 工具支援 collection-aware 操作
4. **階段四**：遷移現有資料到新架構

## 效益
- 支援多種 embedding 模型並存
- Collection 與 embedding 精確對應  
- 避免向量維度不匹配錯誤
- 保持系統靈活性和擴展性