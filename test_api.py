#!/usr/bin/env python3
"""
Flask API测试脚本
用于测试地址NER预测接口
"""

import requests
import json
import sys


def test_api(base_url="http://localhost:5000"):
    """测试API接口"""

    print("开始测试地址NER API...")

    # 测试健康检查
    print("\n1. 测试健康检查接口...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.json()}")
    except Exception as e:
        print(f"健康检查失败: {e}")
        return

    # 测试单个地址预测
    print("\n2. 测试单个地址预测...")
    test_address = "北京市东城区朝阳街"

    try:
        response = requests.post(
            f"{base_url}/predict",
            json={"address": test_address},
            headers={'Content-Type': 'application/json'}
        )
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"响应: {json.dumps(result, ensure_ascii=False, indent=2)}")

        if result.get("code") == 200:
            data = result.get("data", {})
            origin = data.get("origin", [])
            labels = data.get("label", [])

            print("\n预测结果:")
            print("字符 | 标签")
            print("-" * 20)
            for char, label in zip(origin, labels):
                print(f"{char:4} | {label}")

    except Exception as e:
        print(f"单个预测失败: {e}")

    # 测试批量预测
    print("\n3. 测试批量预测...")
    test_addresses = [
        "北京市东城区朝阳街",
        "上海市浦东新区张江路123号",
        "广州市天河区体育东路456号"
    ]

    try:
        response = requests.post(
            f"{base_url}/batch_predict",
            json={"addresses": test_addresses},
            headers={'Content-Type': 'application/json'}
        )
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"批量预测结果: {json.dumps(result, ensure_ascii=False, indent=2)}")

    except Exception as e:
        print(f"批量预测失败: {e}")

    # 测试模型信息
    print("\n4. 测试模型信息接口...")
    try:
        response = requests.get(f"{base_url}/model_info")
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"模型信息: {json.dumps(result, ensure_ascii=False, indent=2)}")

    except Exception as e:
        print(f"获取模型信息失败: {e}")


if __name__ == "__main__":
    base_url = "http://localhost:5000"
    if len(sys.argv) > 1:
        base_url = sys.argv[1]

    test_api(base_url)
