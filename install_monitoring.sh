#!/bin/bash

# MCP Server Qdrant 系統監控功能安裝腳本

echo "🔧 安裝 MCP Server Qdrant 系統監控功能"
echo "============================================"

# 檢查是否在正確目錄
if [ ! -f "pyproject.toml" ]; then
    echo "❌ 請在 mcp-server-qdrant 專案根目錄執行此腳本"
    exit 1
fi

echo "📦 安裝新依賴..."
if command -v uv &> /dev/null; then
    echo "使用 uv 安裝依賴..."
    uv add psutil
else
    echo "使用 pip 安裝依賴..."
    pip install psutil
fi

echo "🧪 測試系統監控功能..."
python test_system_monitor.py

echo ""
echo "🎉 安裝完成！"
echo ""
echo "新增的 MCP 工具："
echo "  📊 qdrant-system-status         - 整體系統狀態"
echo "  📈 qdrant-performance-analysis  - 效能詳細分析"
echo "  🐳 qdrant-docker-containers     - Docker 容器資訊"
echo "  📋 qdrant-container-logs        - 容器日誌查看"
echo ""
echo "⚠️  請重啟 MCP Server 以啟用新功能："
echo "     pkill -f mcp-server-qdrant"
echo "     # 然後重新啟動 Claude Desktop"
