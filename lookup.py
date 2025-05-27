from transformers import AutoTokenizer, AutoModel, AutoConfig
import re
# 替换为你实际使用的 checkpoint 路径

# 查看 config
path = "result/pretrained/hfl_chinese-macbert-base_adapted_ep2_seed2025/swa_model.pt"
model_dir = path.replace("result/pretrained/", "").replace("/swa_model.pt", "")
model_name = re.sub(r'_adapted_ep\d+_seed\d+', '', model_dir)
hf_model_name = model_name.replace('_', '/')
print(hf_model_name)
# print(AutoConfig.from_pretrained("result/pretrained/hfl_chinese-macbert-base_adapted_ep2_seed2025"))
# print(AutoConfig.from_pretrained("result/pretrained/hfl_chinese-macbert-large_adapted_ep2_seed2025"))
# print(AutoConfig.from_pretrained("result/pretrained/hfl_chinese-bert-wwm-ext_adapted_ep2_seed2025"))
# print(AutoConfig.from_pretrained("result/pretrained/hfl_chinese-roberta-wwm-ext_adapted_ep2_seed2025"))
