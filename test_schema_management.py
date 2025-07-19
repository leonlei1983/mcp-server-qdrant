#!/usr/bin/env python3
"""
測試動態 Schema 管理功能
"""
import asyncio
import os
from mcp_server_qdrant.ragbridge.schema_api import schema_api

async def test_schema_management():
    """測試 Schema 管理功能"""
    
    print("🧪 **開始測試動態 Schema 管理功能**\n")
    
    try:
        # 1. 測試獲取當前 Schema
        print("1️⃣ **測試獲取當前 Schema**")
        current_schema = await schema_api.get_current_schema()
        print(f"✅ 當前 Schema 版本: {current_schema['schema_version']}")
        print(f"📊 總欄位數: {current_schema['total_fields']}")
        print(f"🔒 核心欄位數: {current_schema['core_fields_count']}")
        print()
        
        # 2. 測試新增欄位
        print("2️⃣ **測試新增 Schema 欄位**")
        add_result = await schema_api.add_schema_field(
            field_name="test_priority",
            field_type="string",
            description="測試用的優先級欄位",
            required=False,
            validation_rules={
                "allowed_values": ["low", "medium", "high", "urgent"],
                "default_value": "medium"
            }
        )
        
        if add_result["success"]:
            print(f"✅ {add_result['message']}")
            print(f"📋 新 Schema 版本: {add_result['new_schema_version']}")
        else:
            print(f"❌ {add_result['message']}")
        print()
        
        # 3. 測試數據驗證
        print("3️⃣ **測試數據驗證**")
        test_data = {
            "content_id": "test-001",
            "title": "測試內容",
            "content_type": "experience",
            "created_at": "2025-07-19T08:00:00",
            "updated_at": "2025-07-19T08:00:00",
            "test_priority": "high"
        }
        
        validation_result = await schema_api.validate_data(test_data)
        print(f"✅ 驗證結果: {'通過' if validation_result['is_valid'] else '失敗'}")
        if validation_result["validation_errors"]:
            print("❌ 驗證錯誤:")
            for error in validation_result["validation_errors"]:
                print(f"   - {error}")
        print()
        
        # 4. 測試 Schema 使用分析
        print("4️⃣ **測試 Schema 使用分析**")
        sample_data = [
            {
                "content_id": "sample-001",
                "title": "樣本內容 1",
                "content_type": "experience",
                "created_at": "2025-07-19T08:00:00",
                "updated_at": "2025-07-19T08:00:00",
                "test_priority": "high"
            },
            {
                "content_id": "sample-002",
                "title": "樣本內容 2",
                "content_type": "knowledge_base",
                "created_at": "2025-07-19T08:00:00",
                "updated_at": "2025-07-19T08:00:00",
                "test_priority": "medium",
                "unknown_field": "這是未定義的欄位"
            }
        ]
        
        usage_analysis = await schema_api.analyze_schema_usage(sample_data)
        if "error" not in usage_analysis:
            print(f"✅ Schema 合規率: {usage_analysis['schema_compliance_rate']:.1%}")
            print(f"📦 分析樣本數: {usage_analysis['total_samples']}")
            if usage_analysis.get("unknown_fields"):
                print(f"🔍 未知欄位: {', '.join(usage_analysis['unknown_fields'])}")
        else:
            print(f"❌ 分析失敗: {usage_analysis['error']}")
        print()
        
        # 5. 測試獲取改進建議
        print("5️⃣ **測試獲取改進建議**")
        suggestions = await schema_api.get_schema_suggestions(sample_data)
        if "error" not in suggestions:
            print(f"💡 建議數量: {suggestions['suggestion_count']}")
            if suggestions["suggestions"]:
                print("🎯 具體建議:")
                for i, suggestion in enumerate(suggestions["suggestions"], 1):
                    print(f"   {i}. {suggestion['type']}: {suggestion['field_name']}")
                    print(f"      原因: {suggestion['reason']}")
            else:
                print("🎉 Schema 設計良好，無需調整！")
        else:
            print(f"❌ 獲取建議失敗: {suggestions['error']}")
        print()
        
        # 6. 測試演進歷史
        print("6️⃣ **測試 Schema 演進歷史**")
        history = await schema_api.get_schema_evolution_history()
        if "error" not in history:
            print(f"📚 總版本數: {history['total_versions']}")
            print(f"✅ 活躍版本數: {history['active_versions']}")
            print(f"🔄 總遷移數: {history['total_migrations']}")
            
            if history["evolution_history"]:
                print("📋 版本列表:")
                for version_info in history["evolution_history"][-3:]:  # 顯示最新3個版本
                    status = "✅ 活躍" if version_info["is_active"] else "⏸️ 非活躍"
                    print(f"   - 版本 {version_info['version']} {status}")
                    print(f"     描述: {version_info['description']}")
        else:
            print(f"❌ 獲取歷史失敗: {history['error']}")
        print()
        
        # 7. 測試移除欄位（標記為棄用）
        print("7️⃣ **測試移除 Schema 欄位**")
        remove_result = await schema_api.remove_schema_field("test_priority")
        if remove_result["success"]:
            print(f"✅ {remove_result['message']}")
            print(f"📋 新 Schema 版本: {remove_result['new_schema_version']}")
            print(f"💡 {remove_result['note']}")
        else:
            print(f"❌ {remove_result['message']}")
        print()
        
        # 8. 最終驗證
        print("8️⃣ **最終驗證 - 獲取更新後的 Schema**")
        final_schema = await schema_api.get_current_schema()
        print(f"✅ 最終 Schema 版本: {final_schema['schema_version']}")
        print(f"📊 總欄位數: {final_schema['total_fields']}")
        print(f"🔴 已棄用欄位數: {final_schema['deprecated_fields_count']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 測試過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_schema_management())
    print(f"\n🎯 **測試結果: {'成功' if success else '失敗'}**")