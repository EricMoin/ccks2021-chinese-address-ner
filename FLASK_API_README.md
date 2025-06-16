# 地址NER Flask API 使用说明

## 概述

本项目提供了一个基于Flask的REST API服务，用于中文地址的命名实体识别（NER）。该API能够将输入的地址字符串分解为字符级别的标记，并为每个字符分配相应的实体标签。

## 功能特性

- **单个地址预测**: 对单个地址进行NER分析
- **批量地址预测**: 同时处理多个地址
- **健康检查**: 检查服务状态和模型加载情况
- **模型信息**: 获取当前加载的模型详细信息
- **跨域支持**: 支持CORS，便于前端调用
- **错误处理**: 完善的错误处理和日志记录

## 安装依赖

```bash
# 安装Flask相关依赖
pip install -r requirements_flask.txt

# 或者手动安装主要依赖
pip install flask flask-cors torch transformers TorchCRF
```

## 启动服务

### 方法1: 直接运行Flask应用

```bash
python flask_api.py
```

### 方法2: 使用启动脚本（推荐）

```bash
# 检查模型文件是否存在
python start_server.py --check-only

# 启动服务器
python start_server.py

# 自定义端口和主机
python start_server.py --host 127.0.0.1 --port 8080

# 启用调试模式
python start_server.py --debug

# 设置日志级别
python start_server.py --log-level DEBUG
```

### 方法3: 使用Gunicorn（生产环境）

```bash
# 安装gunicorn
pip install gunicorn

# 启动生产服务器
gunicorn -w 4 -b 0.0.0.0:5000 flask_api:app
```

## API接口说明

### 1. 健康检查

**接口地址**: `GET /health`

**功能**: 检查服务状态和模型加载情况

**请求示例**:

```bash
curl -X GET http://localhost:5000/health
```

**响应示例**:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 2. 单个地址预测

**接口地址**: `POST /predict`

**功能**: 对单个地址进行NER预测

**请求格式**:

```json
{
  "address": "北京市东城区朝阳街"
}
```

**请求示例**:

```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"address": "北京市东城区朝阳街"}'
```

**响应格式**:

```json
{
  "code": 200,
  "message": "预测成功",
  "data": {
    "origin": ["北", "京", "市", "东", "城", "区", "朝", "阳", "街"],
    "label": ["B-prov", "I-prov", "E-prov", "B-district", "I-district", "E-district", "B-road", "I-road", "E-road"]
  }
}
```

### 3. 批量地址预测

**接口地址**: `POST /batch_predict`

**功能**: 同时处理多个地址的NER预测

**请求格式**:

```json
{
  "addresses": [
    "北京市东城区朝阳街",
    "上海市浦东新区张江路123号"
  ]
}
```

**请求示例**:

```bash
curl -X POST http://localhost:5000/batch_predict \
     -H "Content-Type: application/json" \
     -d '{"addresses": ["北京市东城区朝阳街", "上海市浦东新区张江路123号"]}'
```

**响应格式**:

```json
{
  "code": 200,
  "message": "批量预测成功",
  "data": [
    {
      "origin": ["北", "京", "市", "东", "城", "区", "朝", "阳", "街"],
      "label": ["B-prov", "I-prov", "E-prov", "B-district", "I-district", "E-district", "B-road", "I-road", "E-road"]
    },
    {
      "origin": ["上", "海", "市", "浦", "东", "新", "区", "张", "江", "路", "1", "2", "3", "号"],
      "label": ["B-prov", "I-prov", "E-prov", "B-district", "I-district", "I-district", "E-district", "B-road", "I-road", "E-road", "B-houseno", "I-houseno", "I-houseno", "E-houseno"]
    }
  ]
}
```

### 4. 模型信息

**接口地址**: `GET /model_info`

**功能**: 获取当前加载的模型详细信息

**请求示例**:

```bash
curl -X GET http://localhost:5000/model_info
```

**响应示例**:

```json
{
  "code": 200,
  "message": "获取模型信息成功",
  "data": {
    "model_name": "sijunhe/nezha-cn-base",
    "num_labels": 85,
    "labels": ["B-prov", "I-prov", "E-prov", "S-prov", ...],
    "device": "cuda:0"
  }
}
```

## 标签说明

本项目使用BIOES标注体系，支持以下地址要素：

| 标签类型 | 说明 | 示例 |
|---------|------|------|
| prov | 省份 | 北京市、上海市 |
| city | 城市 | 海淀区、朝阳区 |
| district | 区县 | 东城区、西城区 |
| devzone | 开发区 | 经济技术开发区 |
| town | 乡镇 | 中关村街道 |
| community | 社区 | 清华园社区 |
| village_group | 村组 | 第一村民小组 |
| road | 道路 | 中关村大街、学院路 |
| roadno | 道路编号 | 甲1号、乙2号 |
| poi | 兴趣点 | 清华大学、中关村 |
| subpoi | 子兴趣点 | 主楼、图书馆 |
| houseno | 门牌号 | 123号、456号 |
| cellno | 小区编号 | A座、B栋 |
| floorno | 楼层号 | 3层、5楼 |
| roomno | 房间号 | 301室、205房 |
| detail | 详细信息 | 靠近、旁边 |
| assist | 辅助信息 | 大约、左右 |
| distance | 距离 | 100米、2公里 |
| intersection | 路口 | 十字路口 |
| redundant | 冗余信息 | 的、地 |
| others | 其他 | 未分类内容 |

**标注体系说明**:

- **B-**: 实体开始（Begin）
- **I-**: 实体内部（Inside）
- **E-**: 实体结束（End）
- **S-**: 单字符实体（Single）
- **O**: 非实体（Outside）

## 错误处理

所有API都遵循统一的错误响应格式：

```json
{
  "code": 400,
  "message": "错误描述",
  "data": null
}
```

常见错误码：

- `400`: 请求格式错误
- `500`: 服务器内部错误
- `503`: 服务不可用（模型未加载）

## 性能优化建议

1. **批量处理**: 对于多个地址，建议使用批量预测接口以提高效率
2. **模型缓存**: 模型在首次加载后会保持在内存中，后续请求响应更快
3. **GPU加速**: 如果有GPU可用，模型会自动使用GPU加速推理
4. **连接池**: 建议使用HTTP连接池来减少连接开销

## 测试

项目提供了测试脚本：

```bash
# 测试所有API接口
python test_api.py

# 测试指定服务器
python test_api.py http://your-server:5000
```

## 部署建议

### 开发环境

```bash
python start_server.py --debug
```

### 生产环境

```bash
# 使用Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 flask_api:app

# 使用Docker
docker build -t address-ner-api .
docker run -p 5000:5000 address-ner-api
```

## 故障排除

### 常见问题

1. **模型文件未找到**
   - 确保模型文件在正确的路径下
   - 运行 `python start_server.py --check-only` 检查模型状态

2. **内存不足**
   - 减少批量预测的数量
   - 考虑使用更小的模型

3. **预测结果不准确**
   - 检查输入地址格式是否正确
   - 确认使用的是正确的预训练模型

4. **服务启动失败**
   - 检查依赖是否完整安装
   - 查看日志文件 `server.log`

### 日志查看

```bash
# 查看实时日志
tail -f server.log

# 查看错误日志
grep ERROR server.log
```

## 许可证

本项目遵循项目原有的许可证协议。
