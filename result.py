import os
from result_writer import ResultWriter
from sentence_reader import SentenceReader
from config import Config
from logger import logger


def main():
    config = Config('config.yaml')

    # 使用新的模型保存目录结构：result/pretrained/
    pretrained_base_dir = os.path.join('result', 'pretrained')

    result_writer = ResultWriter(config=config)

    logger.info(f"配置的标签数量: {len(config.label_map.labels)}")
    logger.info(f"标签映射: {config.label_map.labels}")

    # 自动发现所有训练好的模型目录
    fold_dirs_to_predict = result_writer.discover_work_dirs(
        main_work_dir=pretrained_base_dir,
        model_name_pattern="*"  # 发现所有模型目录
    )

    if not fold_dirs_to_predict:
        logger.warning(f"未在 {pretrained_base_dir} 中找到任何有效的模型目录")
        logger.info("尝试使用手动指定的路径...")

        # 备用：手动指定模型目录（基于实际可能存在的模型）
        potential_model_dirs = [
            os.path.join(pretrained_base_dir,
                         'hfl_chinese-roberta-wwm-ext_adapted_ep2_seed2025'),
            os.path.join(pretrained_base_dir,
                         'hfl_chinese-macbert-base_adapted_ep2_seed2025'),
            os.path.join(pretrained_base_dir,
                         'sijunhe_nezha-cn-base_adapted_ep2_seed2025'),
            os.path.join(pretrained_base_dir, 'chinese-roberta-wwm-ext'),
            os.path.join(pretrained_base_dir, 'chinese-macbert-base'),
        ]

        fold_dirs_to_predict = []
        for model_dir in potential_model_dirs:
            if os.path.exists(model_dir) and (
                os.path.exists(os.path.join(model_dir, "best_model.pt")) or
                os.path.exists(os.path.join(model_dir, "swa_model.pt"))
            ):
                fold_dirs_to_predict.append(model_dir)
                logger.info(f"找到手动指定的模型目录: {model_dir}")

    if fold_dirs_to_predict:
        logger.info(f"将使用 {len(fold_dirs_to_predict)} 个模型进行集成预测:")
        for i, model_dir in enumerate(fold_dirs_to_predict, 1):
            logger.info(f"  {i}. {model_dir}")

        # 运行预测和集成流程
        result_writer.run_prediction_and_ensembling_pipeline(
            fold_work_dirs=fold_dirs_to_predict
        )
    else:
        logger.error("未找到任何有效的模型目录，无法进行预测")
        logger.info("请确保：")
        logger.info("1. 已完成模型训练")
        logger.info("2. 模型文件位于 result/pretrained/[model_name]/ 目录下")
        logger.info("3. 目录中包含 best_model.pt 或 swa_model.pt 文件")


if __name__ == "__main__":
    main()
