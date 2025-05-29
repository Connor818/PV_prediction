# 概述
该项目旨在对中国未来光伏的装机量进行预测，采用零样本预测模型TimeGPT以解决由于样本量小造成的训练集不足的问题。

# TimeGPT
TimeGPT 是由 ​Nixtla 公司开发的全球首个专注于时间序列预测的生成式预训练 Transformer 模型。它在超过 ​1000 亿个数据点上进行预训练，覆盖金融、气象、能源、医疗等多元领域，提升跨场景泛化能力，支持零样本推理。
由于使用TimeGPT模型需要调用Nixtla公司的API，作者已在程序中提供了两个API。如果API失效，请自行前往Nixtla公司的官网免费申请。
```
PV_prediction
├── data                    # Source Data
│   ├── DataSet.csv         # Raw data set
│   └── TestSet.csv         # Test set
├── results                 # Results
├── Capacity_Prod.py        # PV waste calculation
├── main.py                 # Main programe
├── requirements.txt        # Project dependencies
└── README.md               # This file
```
