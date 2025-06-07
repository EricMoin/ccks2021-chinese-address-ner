import os
from result_writer import ResultWriter
from config import Config


def main():
    config = Config('config.yaml')
    config.work_dir = os.path.join(config.work_dir, 'pretrained')

    # 定义模型权重字典用于软投票
    # 格式: {"model_name": weight}
    model_weights = {
        "hfl_chinese-macbert-base_adapted_ep2_seed2025": 1.0,
        "hfl_chinese-roberta-wwm-ext_adapted_ep3_seed2024": 0.5,
        "sijunhe_nezha-cn-base_adapted_ep2_seed2025": 1.5,
    }

    # 创建ResultWriter实例
    # 传入model_weights启用软投票，传入None使用硬投票
    result_writer = ResultWriter(
        config=config,
        model_weights=model_weights  # 软投票
        # model_weights=None         # 硬投票
    )

    print(len(config.label_map.labels))
    print(config.label_map.labels)

    fold_dirs_to_predict = [
        os.path.join(config.work_dir,
                     'hfl_chinese-macbert-base_adapted_ep2_seed2025'),
        os.path.join(config.work_dir,
                     'sijunhe_nezha-base-wwm_adapted_ep2_seed2025'),
        os.path.join(config.work_dir,
                     'hfl_chinese-roberta-wwm-ext_adapted_ep2_seed2025'),
    ]

    result_writer.run_prediction_and_ensembling_pipeline(
        fold_work_dirs=fold_dirs_to_predict)


if __name__ == "__main__":
    main()
