import os
from result_writer import ResultWriter
from sentence_reader import SentenceReader
from config import Config
from logger import logger


def main():
    config = Config('config.yaml')
    config.work_dir = os.path.join(config.work_dir, 'pretrained')
    
    result_writer = ResultWriter(
        config=config
    )
    print(len(config.label_map.labels))
    print(config.label_map.labels)

    # fold_dirs_to_predict = result_writer.discover_fold_work_dirs(
    #     main_work_dir=config.work_dir)
    
    fold_dirs_to_predict = [
        # os.path.join(config.work_dir,'hfl_chinese-macbert-base_adapted_ep2_seed2025'),
        # os.path.join(config.work_dir,'sijunhe_nezha-cn-base_adapted_ep2_seed2025'),
        os.path.join(config.work_dir,'hfl_chinese-roberta-wwm-ext_adapted_ep2_seed2025'),        
    ]
    
    # dirs_to_predict = result_writer.discover_work_dirs(
    #     main_work_dir=config.work_dir)

    result_writer.run_prediction_and_ensembling_pipeline(
        fold_work_dirs=fold_dirs_to_predict)


if __name__ == "__main__":
    main()
