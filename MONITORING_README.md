# MCP Server Qdrant 系統監控功能

## 🎯 功能介紹

基於實際使用經驗，為 `mcp-server-qdrant` 添加了綜合的系統監控功能，可以監控：

- 🏥 **Qdrant 健康狀態** - 服務狀態、Collections 數量
- 📊 **Collections 統計** - 點數、向量數、記憶體使用估算
- 💻 **系統資源** - CPU、記憶體、磁碟使用率
- 🐳 **Docker 容器** - 容器資源使用、狀態監控
- 🔍 **智能分析** - 自動檢測問題並提供最佳化建議

## 🛠️ 安裝步驟

### 1. 更新依賴
```bash
cd /path/to/mcp-server-qdrant
./install_monitoring.sh
```

或手動安裝：
```bash
# 使用 uv
uv add psutil

# 或使用 pip
pip install psutil
```

### 2. 重啟 MCP Server
```bash
# 停止現有的 MCP Server
pkill -f mcp-server-qdrant

# 重新啟動 Claude Desktop 以載入新功能
```

## 🔧 新增的 MCP 工具

### 1. `qdrant-system-status`
**整體系統狀態報告**
```bash
# 使用方式
請透過 Claude 執行：「獲取 Qdrant 系統狀態」
```

**功能：**
- Qdrant 健康檢查
- Collections 統計總覽
- 系統資源使用情況
- Docker 容器狀態
- 智能分析和建議

### 2. `qdrant-performance-analysis`
**詳細效能分析**
```bash
# 使用方式
請透過 Claude 執行：「分析 Qdrant 效能」
```

**功能：**
- 每個 Collection 的詳細統計
- 記憶體使用分析
- Docker 容器效能指標
- 記憶體使用比例分析

### 3. `qdrant-docker-containers`
**Docker 容器管理**
```bash
# 使用方式
請透過 Claude 執行：「查看 Qdrant Docker 容器」
```

**功能：**
- 列出所有 Qdrant 相關容器
- 容器狀態和配置資訊
- 端口映射資訊

### 4. `qdrant-container-logs`
**容器日誌查看**
```bash
# 使用方式
請透過 Claude 執行：「查看 Qdrant 容器日誌」
```

**功能：**
- 獲取容器運行日誌
- 可配置日誌行數
- 故障診斷

## 💡 使用範例

### 日常監控
```
👤 用戶: "檢查 Qdrant 系統狀態"
🤖 Claude: 使用 qdrant-system-status 工具
```

### 效能分析
```
👤 用戶: "分析 Qdrant 的記憶體使用情況"
🤖 Claude: 使用 qdrant-performance-analysis 工具
```

### 故障排除
```
👤 用戶: "Qdrant 好像有問題，幫我檢查一下"
🤖 Claude: 
1. 使用 qdrant-system-status 檢查整體狀態
2. 使用 qdrant-container-logs 查看錯誤日誌
3. 使用 qdrant-performance-analysis 分析效能問題
```

## 🎯 監控指標

### Qdrant 指標
- Collections 數量和狀態
- 點數和向量數統計
- 索引狀態和優化器狀態
- 記憶體使用估算

### 系統指標
- CPU 使用率和核心數
- 記憶體使用量和可用量
- 磁碟使用率
- 系統負載

### Docker 指標
- 容器記憶體使用量
- CPU 使用率
- 網路和磁碟 I/O
- 進程數量

## 🚨 智能預警

系統會自動檢測以下問題並提供建議：

### 記憶體相關
- ⚠️ 系統記憶體使用超過 80%
- 🐳 Docker 容器記憶體使用過高
- 📊 記憶體使用與資料量不成比例

### 效能相關
- 🔍 Collections 優化器狀態異常
- 📈 索引效能問題
- 💾 磁碟空間不足

### 建議範例
```
💡 建議:
• 資料量較少，可考慮使用記憶體優化配置
• 記憶體開銷較高，考慮重啟容器或調整配置
• 考慮增加系統記憶體或關閉不必要的程序
```

## 🔧 進階配置

### 自定義容器名稱
```bash
# 如果您的 Qdrant 容器名稱不是 "qdrant"
# 可以在使用日誌工具時指定：
qdrant-container-logs --container_name my-qdrant-container
```

### 調整日誌行數
```bash
# 獲取更多日誌行數
qdrant-container-logs --lines 100
```

## 🐛 故障排除

### 常見問題

1. **Docker 命令找不到**
   ```
   錯誤: Docker command not found
   解決: 確保 Docker 已安裝並在 PATH 中
   ```

2. **容器不存在**
   ```
   錯誤: Container not found or not running
   解決: 檢查 Qdrant 容器是否正在運行
   ```

3. **權限問題**
   ```
   錯誤: Permission denied
   解決: 確保用戶有執行 Docker 命令的權限
   ```

### 測試安裝
```bash
# 測試系統監控功能
cd /path/to/mcp-server-qdrant
python test_system_monitor.py
```

## 📊 效能影響

這些監控功能設計為：
- **低開銷** - 僅在需要時才執行監控
- **非阻塞** - 使用異步操作，不影響正常使用
- **智能緩存** - 避免重複查詢
- **錯誤容忍** - 監控失敗不影響主要功能

## 🎉 未來計劃

- 📈 歷史資料趨勢分析
- 🔔 主動通知和警告
- 🎛️ 自定義監控閾值
- 🔧 自動化最佳化建議
- 📊 監控資料匯出

---

**問題回報**: 如有任何問題，請提供系統監控報告以便診斷
