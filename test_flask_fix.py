#!/usr/bin/env python3
"""
测试Flask API是否能正常启动
"""

import sys
import os


def test_flask_import():
    """测试Flask API是否能正常导入"""
    try:
        print("🔍 测试Flask API导入...")

        # 测试导入
        from flask_api import app, predictor, AddressNERPredictor
        print("✅ Flask API导入成功")

        # 测试应用创建
        if app:
            print("✅ Flask应用创建成功")
        else:
            print("❌ Flask应用创建失败")
            return False

        # 测试预测器类
        if AddressNERPredictor:
            print("✅ AddressNERPredictor类定义成功")
        else:
            print("❌ AddressNERPredictor类定义失败")
            return False

        print("✅ 所有测试通过！")
        return True

    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        return False
    except AttributeError as e:
        print(f"❌ 属性错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        return False


def test_routes():
    """测试路由是否正确定义"""
    try:
        from flask_api import app

        print("\n🔍 测试路由定义...")

        # 获取所有路由
        routes = []
        for rule in app.url_map.iter_rules():
            routes.append({
                'endpoint': rule.endpoint,
                'methods': list(rule.methods),
                'rule': rule.rule
            })

        expected_routes = [
            '/health',
            '/predict',
            '/batch_predict',
            '/model_info'
        ]

        print("📝 已定义的路由:")
        for route in routes:
            if route['rule'] in expected_routes:
                print(f"✅ {route['rule']} - {route['methods']}")

        return True

    except Exception as e:
        print(f"❌ 路由测试失败: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Flask API 修复验证测试")
    print("=" * 50)

    success = True

    # 测试导入
    if not test_flask_import():
        success = False

    # 测试路由
    if not test_routes():
        success = False

    print("\n" + "=" * 50)
    if success:
        print("🎉 所有测试通过！Flask API修复成功！")
        print("💡 现在可以运行: python start_server.py")
    else:
        print("❌ 测试失败，请检查错误信息")
        sys.exit(1)
