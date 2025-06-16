#!/usr/bin/env python3
"""
地址NER Flask服务器启动脚本
"""

import os
import sys
import argparse
import logging
from flask_api import app, predictor, AddressNERPredictor


def setup_logging(log_level='INFO'):
    """设置日志配置"""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('server.log', mode='a', encoding='utf-8')
        ]
    )


def check_model_availability():
    """检查模型文件是否存在"""
    possible_paths = [
        'result/pretrained/sijunhe_nezha-base-wwm_adapted_ep2_seed2025/best_model.pt',
        'result/pretrained/hfl_chinese-roberta-wwm-ext_adapted_ep2_seed2025/best_model.pt',
        'result/pretrained/hfl_chinese-macbert-base_adapted_ep2_seed2025/best_model.pt'
    ]

    available_models = []
    for path in possible_paths:
        if os.path.exists(path):
            available_models.append(path)

    if not available_models:
        print("❌ 错误: 未找到任何训练好的模型文件!")
        print("请确保以下任一模型文件存在:")
        for path in possible_paths:
            print(f"  - {path}")
        return False
    else:
        print("✅ 找到可用的模型文件:")
        for path in available_models:
            print(f"  - {path}")
        return True


def initialize_predictor():
    """初始化预测器"""
    global predictor
    try:
        print("🔄 正在初始化预测器...")
        predictor = AddressNERPredictor()
        if predictor.model is not None:
            print("✅ 预测器初始化成功!")
            return True
        else:
            print("❌ 预测器初始化失败: 模型未加载")
            return False
    except Exception as e:
        print(f"❌ 预测器初始化失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='启动地址NER Flask服务器')
    parser.add_argument('--host', default='0.0.0.0',
                        help='服务器主机地址 (默认: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000,
                        help='服务器端口 (默认: 5000)')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='日志级别 (默认: INFO)')
    parser.add_argument('--check-only', action='store_true',
                        help='仅检查模型文件是否存在，不启动服务器')

    args = parser.parse_args()

    # 设置日志
    setup_logging(args.log_level)

    print("🚀 地址NER Flask服务器启动器")
    print("=" * 50)

    # 检查模型文件
    if not check_model_availability():
        sys.exit(1)

    if args.check_only:
        print("✅ 模型检查完成")
        return

    # 初始化预测器
    if not initialize_predictor():
        sys.exit(1)

    print(f"🌐 启动服务器...")
    print(f"   主机: {args.host}")
    print(f"   端口: {args.port}")
    print(f"   调试模式: {'开启' if args.debug else '关闭'}")
    print(f"   访问地址: http://{args.host}:{args.port}")
    print("=" * 50)
    print("📝 可用的API接口:")
    print("   GET  /health       - 健康检查")
    print("   POST /predict      - 单个地址预测")
    print("   POST /batch_predict - 批量地址预测")
    print("   GET  /model_info   - 模型信息")
    print("=" * 50)
    print("📖 API使用示例:")
    print("   curl -X POST http://localhost:5000/predict \\")
    print('        -H "Content-Type: application/json" \\')
    print('        -d \'{"address": "北京市东城区朝阳街"}\'')
    print("=" * 50)
    print("⏹️  按 Ctrl+C 停止服务器")
    print()

    try:
        # 启动Flask应用
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
    except Exception as e:
        print(f"❌ 服务器启动失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
