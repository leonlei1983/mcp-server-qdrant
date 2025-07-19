#!/usr/bin/env python3
"""
Schema 審查管理 CLI 工具

用法:
  python schema_admin_cli.py list                           # 列出待審查請求
  python schema_admin_cli.py review <request_id> approve <reviewer> [comments]  # 核准請求
  python schema_admin_cli.py review <request_id> reject <reviewer> [comments]   # 拒絕請求
  python schema_admin_cli.py history [limit]                # 查看審查歷史
  python schema_admin_cli.py request add <field_name> <field_type> [options]    # 創建新增請求
  python schema_admin_cli.py request remove <field_name> [justification]        # 創建移除請求
"""
import argparse
import asyncio
import sys
from typing import List, Optional

from mcp_server_qdrant.ragbridge.schema_approval import get_approval_manager
from mcp_server_qdrant.ragbridge.schema_manager import schema_manager


class SchemaAdminCLI:
    """Schema 管理 CLI"""
    
    def __init__(self):
        self.approval_manager = get_approval_manager()
    
    def list_pending_requests(self, reviewer: str = "admin") -> None:
        """列出待審查請求"""
        requests = self.approval_manager.get_pending_requests(reviewer)
        
        if not requests:
            print("✅ 目前沒有待審查的 Schema 變更請求")
            return
        
        print(f"📋 待審查的 Schema 變更請求 ({len(requests)} 個):")
        print("=" * 80)
        
        for i, request in enumerate(requests, 1):
            risk_emoji = {
                "low": "🟢",
                "medium": "🟡",
                "high": "🔴",
                "critical": "🚨"
            }.get(request["risk_level"], "⚪")
            
            print(f"{i}. 請求 {request['request_id']} {risk_emoji}")
            print(f"   🔧 變更類型: {request['change_type']}")
            print(f"   🏗️ 欄位名稱: {request['field_name']}")
            print(f"   ⚠️ 風險級別: {request['risk_level']}")
            print(f"   👥 需要權限: {request['required_approval_level']}")
            print(f"   👤 提案者: {request['proposed_by']}")
            print(f"   📅 提案時間: {request['proposed_at']}")
            print(f"   📝 理由: {request['justification']}")
            print(f"   💊 影響分析: {request['impact_analysis']}")
            print()
        
        print("💡 審查指令:")
        print("   核准: python schema_admin_cli.py review <request_id> approve <reviewer> [comments]")
        print("   拒絕: python schema_admin_cli.py review <request_id> reject <reviewer> [comments]")
    
    def review_request(self, request_id: str, action: str, reviewer: str, comments: str = "") -> None:
        """審查請求"""
        if action not in ["approve", "reject"]:
            print(f"❌ 無效的審查動作: {action}")
            print("💡 請使用 'approve' 或 'reject'")
            return
        
        if request_id not in self.approval_manager.pending_requests:
            print(f"❌ 請求不存在: {request_id}")
            print("🔍 使用 'list' 命令查看所有待審查請求")
            return
        
        success = self.approval_manager.review_request(
            request_id=request_id,
            reviewer=reviewer,
            action=action,
            comments=comments
        )
        
        if not success:
            print(f"❌ 審查失敗: 權限不足或請求不存在")
            print(f"👤 審查者: {reviewer}")
            print(f"📋 請求ID: {request_id}")
            return
        
        # 從歷史記錄中找到審查結果
        reviewed_request = next(
            req for req in self.approval_manager.approval_history
            if req.request_id == request_id
        )
        
        if action == "approve":
            if reviewed_request.status == "approved":
                print(f"✅ Schema 變更請求已核准並執行")
                print(f"📋 請求ID: {request_id}")
                print(f"🏗️ 變更類型: {reviewed_request.change_type}")
                print(f"🔧 欄位: {reviewed_request.field_name}")
                print(f"👤 審查者: {reviewer}")
                print(f"📝 審查意見: {comments}")
                print(f"⏰ 審查時間: {reviewed_request.reviewed_at}")
                print(f"🎉 Schema 變更已成功應用！")
            else:
                print(f"❌ Schema 變更執行失敗")
                print(f"📋 請求ID: {request_id}")
                print(f"💬 錯誤: {reviewed_request.review_comments}")
        else:  # reject
            print(f"❌ Schema 變更請求已拒絕")
            print(f"📋 請求ID: {request_id}")
            print(f"🏗️ 變更類型: {reviewed_request.change_type}")
            print(f"🔧 欄位: {reviewed_request.field_name}")
            print(f"👤 審查者: {reviewer}")
            print(f"📝 拒絕理由: {comments}")
            print(f"⏰ 審查時間: {reviewed_request.reviewed_at}")
    
    def show_history(self, limit: int = 10) -> None:
        """顯示審查歷史"""
        history = self.approval_manager.get_approval_history(limit)
        
        if not history:
            print("📋 暫無 Schema 審查歷史記錄")
            return
        
        print(f"📚 Schema 審查歷史記錄 (最近 {len(history)} 個):")
        print("=" * 80)
        
        for i, record in enumerate(history, 1):
            status_emoji = {"approved": "✅", "rejected": "❌"}.get(record["status"], "⏳")
            risk_emoji = {
                "low": "🟢",
                "medium": "🟡",
                "high": "🔴",
                "critical": "🚨"
            }.get(record["risk_level"], "⚪")
            
            print(f"{i}. 請求 {record['request_id']} {status_emoji} {risk_emoji}")
            print(f"   🔧 變更: {record['change_type']} → {record['field_name']}")
            print(f"   👤 提案者: {record['proposed_by']}")
            print(f"   👥 審查者: {record['reviewed_by'] or 'N/A'}")
            print(f"   📅 提案時間: {record['proposed_at']}")
            print(f"   ⏰ 審查時間: {record['reviewed_at'] or 'N/A'}")
            print(f"   📝 審查意見: {record['review_comments']}")
            print()
    
    def create_add_request(
        self, 
        field_name: str, 
        field_type: str, 
        description: str = "",
        required: bool = False,
        justification: str = "",
        proposed_by: str = "cli_user"
    ) -> None:
        """創建新增欄位請求"""
        change_details = {
            "field_type": field_type,
            "description": description,
            "required": required
        }
        
        request_id = self.approval_manager.create_change_request(
            change_type="add_field",
            field_name=field_name,
            change_details=change_details,
            proposed_by=proposed_by,
            justification=justification
        )
        
        if request_id in self.approval_manager.approval_history:
            # 已自動核准
            executed_request = next(
                req for req in self.approval_manager.approval_history
                if req.request_id == request_id
            )
            
            if executed_request.status == "approved":
                print(f"✅ 低風險變更已自動核准並執行")
                print(f"📋 請求ID: {request_id}")
                print(f"🏗️ 新增欄位: {field_name} ({field_type})")
                print(f"📝 說明: {executed_request.review_comments}")
            else:
                print(f"❌ 自動執行失敗")
                print(f"📋 請求ID: {request_id}")
                print(f"💬 錯誤: {executed_request.review_comments}")
        else:
            # 等待審查
            pending_request = self.approval_manager.pending_requests[request_id]
            print(f"📋 Schema 變更請求已創建")
            print(f"🆔 請求ID: {request_id}")
            print(f"🏗️ 變更類型: 新增欄位 '{field_name}'")
            print(f"⚠️ 風險級別: {pending_request.risk_level.value}")
            print(f"👥 需要審查級別: {pending_request.required_approval_level.value}")
            print(f"📝 提案理由: {justification}")
            print(f"⏳ 狀態: 等待審查")
    
    def create_remove_request(
        self,
        field_name: str,
        justification: str = "",
        proposed_by: str = "cli_user"
    ) -> None:
        """創建移除欄位請求"""
        request_id = self.approval_manager.create_change_request(
            change_type="remove_field",
            field_name=field_name,
            change_details={"deprecated": True},
            proposed_by=proposed_by,
            justification=justification
        )
        
        pending_request = self.approval_manager.pending_requests[request_id]
        print(f"⚠️ 高風險 Schema 變更請求已創建")
        print(f"🆔 請求ID: {request_id}")
        print(f"🗑️ 變更類型: 移除欄位 '{field_name}'")
        print(f"🔴 風險級別: {pending_request.risk_level.value}")
        print(f"👥 需要審查級別: {pending_request.required_approval_level.value}")
        print(f"📝 移除理由: {justification}")
        print(f"⏳ 狀態: 等待高級審查")


def main():
    parser = argparse.ArgumentParser(description='Schema 審查管理 CLI 工具')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # list 命令
    list_parser = subparsers.add_parser('list', help='列出待審查請求')
    list_parser.add_argument('--reviewer', default='admin', help='審查者身份')
    
    # review 命令
    review_parser = subparsers.add_parser('review', help='審查請求')
    review_parser.add_argument('request_id', help='請求ID')
    review_parser.add_argument('action', choices=['approve', 'reject'], help='審查動作')
    review_parser.add_argument('reviewer', help='審查者身份')
    review_parser.add_argument('comments', nargs='?', default='', help='審查意見')
    
    # history 命令
    history_parser = subparsers.add_parser('history', help='查看審查歷史')
    history_parser.add_argument('--limit', type=int, default=10, help='顯示記錄數量')
    
    # request 命令
    request_parser = subparsers.add_parser('request', help='創建變更請求')
    request_subparsers = request_parser.add_subparsers(dest='request_type', help='請求類型')
    
    # request add
    add_parser = request_subparsers.add_parser('add', help='創建新增欄位請求')
    add_parser.add_argument('field_name', help='欄位名稱')
    add_parser.add_argument('field_type', help='欄位類型')
    add_parser.add_argument('--description', default='', help='欄位描述')
    add_parser.add_argument('--required', action='store_true', help='是否必填')
    add_parser.add_argument('--justification', default='', help='變更理由')
    add_parser.add_argument('--proposed-by', default='cli_user', help='提案者')
    
    # request remove
    remove_parser = request_subparsers.add_parser('remove', help='創建移除欄位請求')
    remove_parser.add_argument('field_name', help='欄位名稱')
    remove_parser.add_argument('--justification', default='', help='移除理由')
    remove_parser.add_argument('--proposed-by', default='cli_user', help='提案者')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = SchemaAdminCLI()
    
    try:
        if args.command == 'list':
            cli.list_pending_requests(args.reviewer)
        
        elif args.command == 'review':
            cli.review_request(args.request_id, args.action, args.reviewer, args.comments)
        
        elif args.command == 'history':
            cli.show_history(args.limit)
        
        elif args.command == 'request':
            if args.request_type == 'add':
                cli.create_add_request(
                    args.field_name,
                    args.field_type,
                    args.description,
                    args.required,
                    args.justification,
                    args.proposed_by
                )
            elif args.request_type == 'remove':
                cli.create_remove_request(
                    args.field_name,
                    args.justification,
                    args.proposed_by
                )
            else:
                request_parser.print_help()
        
    except Exception as e:
        print(f"❌ 執行命令時發生錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()