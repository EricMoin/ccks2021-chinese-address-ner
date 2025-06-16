import os
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from typing import List, Dict, Any
import traceback

# 导入项目模块
from config import Config
from model import AddressNER
from label import LabelMap
from conll_reader import ConllEntity
from dataset import NERDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AddressNERPredictor:
    """地址NER预测器类"""

    def __init__(self, config_path: str = 'config.yaml', model_path: str = None):
        """
        初始化预测器
        Args:
            config_path: 配置文件路径
            model_path: 模型文件路径
        """
        self.config = Config(config_path)
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")

        # 加载模型
        self.model = None
        self.tokenizer = None

        if model_path is None:
            # 查找可用的模型文件
            model_path = self._find_best_model()

        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            logger.error("未找到可用的模型文件")

    def _find_best_model(self) -> str:
        """查找最佳模型文件"""
        possible_paths = [
            'result/pretrained/sijunhe_nezha-base-wwm_adapted_ep2_seed2025/best_model.pt',
            'result/pretrained/hfl_chinese-roberta-wwm-ext_adapted_ep2_seed2025/best_model.pt',
            'result/pretrained/hfl_chinese-macbert-base_adapted_ep2_seed2025/best_model.pt'
        ]

        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"找到模型文件: {path}")
                return path

        logger.warning("未找到预训练模型文件")
        return None

    def _load_model(self, model_path: str):
        """加载模型"""
        try:
            # 从模型路径推断模型名称
            if 'nezha' in model_path:
                model_name = 'sijunhe/nezha-cn-base'
            elif 'macbert' in model_path:
                model_name = 'hfl/chinese-macbert-base'
            else:
                model_name = 'hfl/chinese-roberta-wwm-ext'

            # 更新配置中的模型名称
            self.config.model_name = model_name

            # 初始化模型
            self.model = AddressNER(
                num_labels=len(self.config.label_map.labels),
                config=self.config
            )

            # 加载模型权重
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            # 获取tokenizer
            self.tokenizer = self.model.tokenizer

            logger.info(f"成功加载模型: {model_path}")

        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            self.model = None
            self.tokenizer = None

    def predict_single(self, address: str) -> Dict[str, Any]:
        """
        对单个地址进行预测
        Args:
            address: 输入地址字符串
        Returns:
            包含origin和label的字典
        """
        if not self.model or not self.tokenizer:
            return {"error": "模型未加载"}

        try:
            # 将地址转换为字符列表
            chars = list(address)

            # 创建ConllEntity对象
            conll_entity = ConllEntity(chars, ['O'] * len(chars))

            # 创建数据集
            dataset = NERDataset(
                data=[conll_entity],
                tokenizer=self.tokenizer,
                label_map=self.config.label_map.label2id
            )

            # 创建数据加载器
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

            # 进行预测
            with torch.no_grad():
                for batch in dataloader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)

                    # 模型推理
                    predictions = self.model(input_ids, attention_mask)

                    # 获取第一个样本的预测结果
                    pred_indices = predictions[0]

                    # 将索引转换为标签
                    pred_labels = [
                        self.config.label_map.id2label.get(idx, 'O')
                        for idx in pred_indices
                    ]

                    # 截断到原始长度
                    pred_labels = pred_labels[:len(chars)]

                    return {
                        "origin": chars,
                        "label": pred_labels
                    }

        except Exception as e:
            logger.error(f"预测失败: {e}")
            return {"error": f"预测失败: {str(e)}"}

        return {"error": "预测失败"}


# 创建Flask应用
app = Flask(__name__)
# 启用跨域支持，允许所有来源和方法
CORS(app, origins='*',
     methods=['GET', 'POST', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization'])

# 全局预测器实例
predictor = None


def initialize_predictor():
    """初始化预测器"""
    global predictor
    try:
        if predictor is None:
            predictor = AddressNERPredictor()
            logger.info("预测器初始化成功")
    except Exception as e:
        logger.error(f"预测器初始化失败: {e}")
        predictor = None


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    initialize_predictor()  # 确保预测器已初始化
    return jsonify({
        "code": 200,
        "message": "健康检查成功",
        "data": {
            "status": "healthy",
            "model_loaded": predictor is not None and predictor.model is not None
        }
    })


@app.route('/predict', methods=['POST'])
def predict_address():
    """地址NER预测接口"""
    try:
        initialize_predictor()  # 确保预测器已初始化
        # 检查预测器是否可用
        if not predictor or not predictor.model:
            return jsonify({
                "code": 500,
                "message": "预测器未初始化或模型未加载",
                "data": None
            }), 500

        # 获取请求数据
        data = request.get_json()
        if not data or 'address' not in data:
            return jsonify({
                "code": 400,
                "message": "请求格式错误，需要包含address字段",
                "data": None
            }), 400

        address = data['address'].strip()
        if not address:
            return jsonify({
                "code": 400,
                "message": "地址不能为空",
                "data": None
            }), 400

        # 进行预测
        result = predictor.predict_single(address)

        if "error" in result:
            return jsonify({
                "code": 500,
                "message": result["error"],
                "data": None
            }), 500

        # 返回成功结果
        return jsonify({
            "code": 200,
            "message": "预测成功",
            "data": result
        })

    except Exception as e:
        logger.error(f"处理请求时发生错误: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "code": 500,
            "message": f"服务器内部错误: {str(e)}",
            "data": None
        }), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict_addresses():
    """批量地址NER预测接口"""
    try:
        initialize_predictor()  # 确保预测器已初始化
        # 检查预测器是否可用
        if not predictor or not predictor.model:
            return jsonify({
                "code": 500,
                "message": "预测器未初始化或模型未加载",
                "data": None
            }), 500

        # 获取请求数据
        data = request.get_json()
        if not data or 'addresses' not in data:
            return jsonify({
                "code": 400,
                "message": "请求格式错误，需要包含addresses字段",
                "data": None
            }), 400

        addresses = data['addresses']
        if not isinstance(addresses, list) or len(addresses) == 0:
            return jsonify({
                "code": 400,
                "message": "addresses必须是非空数组",
                "data": None
            }), 400

        # 批量预测
        results = []
        for addr in addresses:
            if isinstance(addr, str) and addr.strip():
                result = predictor.predict_single(addr.strip())
                results.append(result)
            else:
                results.append({"error": "无效地址"})

        return jsonify({
            "code": 200,
            "message": "批量预测成功",
            "data": results
        })

    except Exception as e:
        logger.error(f"处理批量请求时发生错误: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "code": 500,
            "message": f"服务器内部错误: {str(e)}",
            "data": None
        }), 500


@app.route('/model_info', methods=['GET'])
def get_model_info():
    """获取模型信息接口"""
    try:
        initialize_predictor()  # 确保预测器已初始化
        if not predictor or not predictor.model:
            return jsonify({
                "code": 500,
                "message": "模型未加载",
                "data": None
            }), 500

        info = {
            "model_name": predictor.config.model_name,
            "num_labels": len(predictor.config.label_map.labels),
            "labels": predictor.config.label_map.labels,
            "device": str(predictor.device)
        }

        return jsonify({
            "code": 200,
            "message": "获取模型信息成功",
            "data": info
        })

    except Exception as e:
        logger.error(f"获取模型信息时发生错误: {e}")
        return jsonify({
            "code": 500,
            "message": f"服务器内部错误: {str(e)}",
            "data": None
        }), 500


if __name__ == '__main__':
    # 在主线程中初始化预测器
    try:
        predictor = AddressNERPredictor()
        logger.info("预测器初始化成功")
    except Exception as e:
        logger.error(f"预测器初始化失败: {e}")
        predictor = None

    # 启动Flask应用
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,  # 生产环境设置为False
        threaded=True
    )
