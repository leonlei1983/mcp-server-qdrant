#!/usr/bin/env python3
"""
測試 Schema 審查工作流程
"""
import asyncio
from mcp_server_qdrant.ragbridge.schema_approval import get_approval_manager

async def test_schema_approval_workflow():
    """測試完整的 Schema 審查工作流程"""
    
    print("🧪 **開始測試 Schema 審查工作流程**\n")
    
    try:
        approval_manager = get_approval_manager()
        
        # 1. 測試創建低風險變更請求（應該自動核准）
        print("1️⃣ **測試低風險變更請求（自動核准）**")
        request_id_1 = approval_manager.create_change_request(
            change_type="add_field",
            field_name="priority_level",
            change_details={
                "field_type": "string",
                "description": "優先級別",
                "required": False,  # 可選欄位，低風險
                "validation": {
                    "allowed_values": ["low", "medium", "high", "urgent"]
                }
            },
            proposed_by="test_user",
            justification="新增內容優先級管理功能"
        )
        print(f"✅ 低風險請求 ID: {request_id_1}")
        print(f"📋 待審查請求數量: {len(approval_manager.pending_requests)}")
        print(f"📚 歷史記錄數量: {len(approval_manager.approval_history)}")
        print()
        
        # 2. 測試創建中風險變更請求（需要審查）
        print("2️⃣ **測試中風險變更請求（需要審查）**")
        request_id_2 = approval_manager.create_change_request(
            change_type="add_field",
            field_name="mandatory_tags",
            change_details={
                "field_type": "list",
                "description": "必填標籤列表",
                "required": True,  # 必填欄位，中風險
                "validation": {
                    "min_length": 1
                }
            },
            proposed_by="test_user",
            justification="強化內容分類機制"
        )
        print(f"✅ 中風險請求 ID: {request_id_2}")
        print(f"📋 待審查請求數量: {len(approval_manager.pending_requests)}")
        print()
        
        # 3. 測試創建高風險變更請求（需要管理員審查）
        print("3️⃣ **測試高風險變更請求（管理員審查）**")
        request_id_3 = approval_manager.create_change_request(
            change_type="remove_field",
            field_name="old_field",
            change_details={"deprecated": True},
            proposed_by="test_user",
            justification="移除過時的欄位定義"
        )
        print(f"✅ 高風險請求 ID: {request_id_3}")
        print(f"📋 待審查請求數量: {len(approval_manager.pending_requests)}")
        print()
        
        # 4. 測試獲取待審查請求
        print("4️⃣ **測試獲取待審查請求**")
        pending_requests = approval_manager.get_pending_requests("admin")
        print(f"📋 管理員可見的待審查請求: {len(pending_requests)}")
        for request in pending_requests:
            print(f"   - 請求 {request['request_id'][:8]}: {request['change_type']} → {request['field_name']}")
            print(f"     風險級別: {request['risk_level']}, 需要權限: {request['required_approval_level']}")
        print()
        
        # 5. 測試審查者權限（一般審查員）
        print("5️⃣ **測試一般審查員權限**")
        reviewer_requests = approval_manager.get_pending_requests("schema_reviewer")
        print(f"📋 一般審查員可見的請求: {len(reviewer_requests)}")
        
        # 審查中風險請求
        if request_id_2 in approval_manager.pending_requests:
            success = approval_manager.review_request(
                request_id=request_id_2,
                reviewer="schema_reviewer",
                action="approve",
                comments="審查通過，必填標籤有助於內容分類"
            )
            print(f"✅ 中風險請求審查結果: {'成功' if success else '失敗'}")
        print()
        
        # 6. 測試管理員權限審查高風險請求
        print("6️⃣ **測試管理員審查高風險請求**")
        if request_id_3 in approval_manager.pending_requests:
            success = approval_manager.review_request(
                request_id=request_id_3,
                reviewer="admin",
                action="reject",
                comments="目前該欄位仍在使用中，暫不移除"
            )
            print(f"✅ 高風險請求審查結果: {'成功' if success else '失敗'}")
        print()
        
        # 7. 測試權限控制（非授權用戶）
        print("7️⃣ **測試權限控制**")
        # 創建一個新的請求來測試
        request_id_4 = approval_manager.create_change_request(
            change_type="add_field",
            field_name="test_field",
            change_details={
                "field_type": "string",
                "required": True
            },
            proposed_by="test_user",
            justification="測試權限控制"
        )
        
        # 嘗試用非授權用戶審查
        success = approval_manager.review_request(
            request_id=request_id_4,
            reviewer="unauthorized_user",
            action="approve",
            comments="嘗試非授權審查"
        )
        print(f"✅ 非授權審查嘗試結果: {'失敗' if not success else '意外成功'}")
        print()
        
        # 8. 測試審查歷史
        print("8️⃣ **測試審查歷史**")
        history = approval_manager.get_approval_history(10)
        print(f"📚 審查歷史記錄數量: {len(history)}")
        for record in history[-3:]:  # 顯示最近3個
            print(f"   - 請求 {record['request_id'][:8]}: {record['status']} by {record['reviewed_by']}")
            print(f"     變更: {record['change_type']} → {record['field_name']}")
        print()
        
        # 9. 測試最終狀態
        print("9️⃣ **測試最終狀態**")
        print(f"📋 剩餘待審查請求: {len(approval_manager.pending_requests)}")
        print(f"📚 總審查歷史: {len(approval_manager.approval_history)}")
        
        # 顯示剩餘的待審查請求
        if approval_manager.pending_requests:
            print("🔍 剩餘待審查請求:")
            for req_id, request in approval_manager.pending_requests.items():
                print(f"   - {req_id[:8]}: {request.change_type} → {request.field_name}")
                print(f"     風險: {request.risk_level.value}, 需要: {request.required_approval_level.value}")
        
        return True
        
    except Exception as e:
        print(f"❌ 測試過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_schema_approval_workflow())
    print(f"\n🎯 **審查工作流程測試結果: {'成功' if success else '失敗'}**")