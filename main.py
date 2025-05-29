import numpy as np
import pandas as pd
from nixtla import NixtlaClient
from utilsforecast.losses import mae, rmse, mape
from utilsforecast.evaluation import evaluate
from tabulate import tabulate
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
import random
from functools import reduce
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM, NHITS, RNN
import logging
import os

def TimeGPT_no_finetune(df, test_df):
    # 数据预处理
    PV_capacity_test = test_df
    PV_capacity = df.iloc[:, [0, 1, 2]]

    # 进行四折交叉验证
    timegpt_cv_df = nixtla_client.cross_validation(
        df=PV_capacity,
        h=24,
        n_windows=4,
        model='timegpt-1-long-horizon',
        time_col='Date',
        target_col='PV_capacity_China_GW',
        id_col='unique_id'
    )

    # 计算模型在验证集上的MAE, RMSE, MAPE
    cv_evaluation = evaluate(
        timegpt_cv_df,
        metrics=[mae, mape, rmse],
        models=['TimeGPT'],
        target_col='PV_capacity_China_GW',
        id_col='unique_id'
    )
    average_metrics = cv_evaluation.groupby('metric')['TimeGPT'].mean()
    print("TimeGPT初始模型（未经微调）交叉验证结果：")
    print(average_metrics)

    # 进行预测
    fcst_df = nixtla_client.forecast(
        df=PV_capacity,
        h=336,
        level=[95],
        model='timegpt-1-long-horizon',
        time_col='Date',
        target_col='PV_capacity_China_GW',
        id_col='unique_id'
    )

    # 评估测试集
    fcst_test = fcst_df.iloc[:24, :]
    test = pd.merge(PV_capacity_test, fcst_test, 'left', ['unique_id', 'Date'])
    test_evaluation = evaluate(
        test,
        metrics=[mae, rmse, mape],
        models=['TimeGPT'],
        target_col='PV_capacity_China_GW',
        id_col='unique_id'
    )
    
    # 输出模型在测试集上的表现
    average_metrics = test_evaluation.groupby('metric')['TimeGPT'].mean()
    print("TimeGPT初始模型（未经微调）测试集结果：")
    print(average_metrics)

    # 输出结果
    fcst_df = fcst_df.rename(
        columns={'TimeGPT': 'TimeGPT_no_finetune','TimeGPT-hi-95': 'TimeGPT_no_finetune-hi-95','TimeGPT-lo-95': 'TimeGPT_no_finetune-lo-95'},
        inplace=False)
    fcst_df = fcst_df[['Date', 'TimeGPT_no_finetune', 'TimeGPT_no_finetune-hi-95', 'TimeGPT_no_finetune-lo-95']]
    return fcst_df


def TimeGPT(df, test_df):
    # 数据预处理
    PV_capacity_test = test_df
    PV_capacity = df.iloc[:, [0, 1, 2]]
    solar_prod = df.iloc[:, [0, 1, 3]]
    renewable_capacity = df.iloc[:, [0, 1, 4]]

    # 第一次微调
    first_model_id = nixtla_client.finetune(
        df=renewable_capacity,
        freq='MS',
        finetune_steps=7,
        finetune_depth=1,
        time_col='Date',
        target_col='Renewable_energy_capacity_China_%',
        model='timegpt-1-long-horizon',
        id_col='unique_id',
        output_model_id='first-finetuned-model'
    )

    # 第二次微调
    second_model_id = nixtla_client.finetune(
        df=solar_prod,
        freq='MS',
        finetune_steps=6,
        finetune_depth=1,
        time_col='Date',
        target_col='solar_prod_GW',
        finetuned_model_id=first_model_id,
        model='timegpt-1-long-horizon',
        id_col='unique_id',
        output_model_id='second-finetuned-model'
    )

    # 微调模型输出
    finetuned_models = nixtla_client.finetuned_models(as_df=True)
    print("微调模型：")
    print(tabulate(finetuned_models, headers='keys', tablefmt='psql', showindex=False))

    # 进行四折交叉验证
    timegpt_cv_df = nixtla_client.cross_validation(
        df=PV_capacity,
        h=24,
        n_windows=4,
        finetuned_model_id=second_model_id,
        model='timegpt-1-long-horizon',
        time_col='Date',
        target_col='PV_capacity_China_GW',
        id_col='unique_id'
    )

    # 计算模型在验证集上的MAE, RMSE, MAPE
    cv_evaluation = evaluate(
        timegpt_cv_df,
        metrics=[mae, mape, rmse],
        models=['TimeGPT'],
        target_col='PV_capacity_China_GW',
        id_col='unique_id'
    )
    average_metrics = cv_evaluation.groupby('metric')['TimeGPT'].mean()
    print("TimeGPT微调后模型交叉验证结果：")
    print(average_metrics)

    # 进行预测
    fcst_df = nixtla_client.forecast(
        df=PV_capacity,
        h=336,
        level=[95],
        finetuned_model_id=second_model_id,
        model='timegpt-1-long-horizon',
        time_col='Date',
        target_col='PV_capacity_China_GW',
        id_col='unique_id'
    )

    # 评估测试集
    fcst_test = fcst_df.iloc[:24, :]
    test = pd.merge(PV_capacity_test, fcst_test, 'left', ['unique_id', 'Date'])
    test_evaluation = evaluate(
        test,
        metrics=[mae, rmse, mape],
        models=['TimeGPT'],
        target_col='PV_capacity_China_GW',
        id_col='unique_id'
    )

    # 输出模型在测试集上的表现
    average_metrics = test_evaluation.groupby('metric')['TimeGPT'].mean()
    print("TimeGPT微调后模型测试集结果：")
    print(average_metrics)

    return fcst_df

def Prophet_fcst(df, test_df):
    # 导入数据
    PV_capacity = df.iloc[:, [1, 2]]
    PV_capacity = PV_capacity.rename(columns={'Date': 'ds', 'PV_capacity_China_GW': 'y'}, inplace=False)

    # 初始化模型（关键参数说明）
    model = Prophet(
        weekly_seasonality=False,  # 关闭周季节性
        daily_seasonality=False,  # 关闭日季节性
        yearly_seasonality=True,  # 保留年季节性
        seasonality_mode='additive',  # 根据数据选择加法/乘法模式
        interval_width=0.95
    )

    # 模型拟合
    model.fit(PV_capacity)

    # 创建未来12个月的日期（格式为'YYYY-MM'）
    future_months = pd.date_range(
        start='2023-01-01',
        periods=336,
        freq='ME'
    ).strftime('%Y-%m').tolist()

    # 进行四折交叉验证
    cutoffs = pd.date_range(start='2014-12-01', end='2020-12-01', freq='2YS-DEC')  # 2年间隔，12月为起始月
    df_cv = cross_validation(model, cutoffs=cutoffs, horizon='730 days')
    df_p = performance_metrics(df_cv)
    print('\nProphet模型交叉验证结果：')
    print('mae:', df_p['mae'].mean())
    print('mape:', df_p['mape'].mean())
    print('rmse:', df_p['rmse'].mean())

    # 构建未来数据框
    future = pd.DataFrame({'ds': future_months})
    # 生成预测结果
    fcst_df = model.predict(future)

    # 测试集结果
    test = test_df.iloc[:, 2]
    mae = mean_absolute_error(test, fcst_df.loc[:23, 'yhat'])
    mape = mean_absolute_percentage_error(test, fcst_df.loc[:23, 'yhat'])
    rmse = root_mean_squared_error(test, fcst_df.loc[:23, 'yhat'])
    print('\nProphet模型测试集结果：')
    print('mae:', mae)
    print('mape:', mape)
    print('rmse:', rmse)

    # 输出结果
    fcst_df = fcst_df.rename(columns={'ds': 'Date', 'yhat': 'Prophet', 'yhat_lower':'Prophet-hi-95', 'yhat_upper':'Prophet-lo-95'}, inplace=False)
    fcst_df = fcst_df[['Date', 'Prophet', 'Prophet-hi-95', 'Prophet-lo-95']]
    return fcst_df

def LSTM_NHITS_RNN(df, test_df):

    # 导入数据
    train = df.rename(columns={'Date': 'ds', 'PV_capacity_China_GW': 'y'}, inplace=False)
    test_df=test_df.rename(columns={'Date': 'ds', 'PV_capacity_China_GW': 'y'}, inplace=False)

    # 设置预测步长
    horizon = 24

    # 构建模型
    models = [
        LSTM(h=horizon,  # 预测范围
            input_size=2 * horizon,  # 输入训练集大小
             max_steps=200,  # 训练步数
             scaler_type='standard',  # 数据标准化的方法
             encoder_hidden_size=64,  # LSTM隐藏层的大小
             decoder_hidden_size=64,  # MLP每一层的隐藏单元数量
             random_seed=42 # 设置固定随机种子
             ),
        NHITS(h=horizon,  # 预测范围
              input_size=2 * horizon,  # 输入训练集大小
              max_steps=100,  # 训练步数
              n_freq_downsample=[3, 2, 1],  # 每个堆栈输出的降采样因子
              random_seed=42    # 设置固定随机种子
              ),
        RNN(h=horizon,  # 预测范围
            input_size=2 * horizon,  # 输入训练集大小
            inference_input_size=24,
            scaler_type='standard', # 数据标准化的方法
            encoder_n_layers=3,
            encoder_hidden_size=128,    # 隐藏层的大小
            decoder_hidden_size=128,    # MLP每一层的隐藏单元数量
            decoder_layers=2,   # MLP层数
            max_steps=200,  # 训练步数
            random_seed=42  # 设置固定随机种子
            )
    ]
    nf = NeuralForecast(models=models, freq='ME')

    # 进行四折交叉验证
    cv_df = nf.cross_validation(train, n_windows=4, step_size=horizon)

    # 计算模型在验证集上的MAE, RMSE, MAPE
    cv_evaluation = evaluate(
        cv_df.drop(columns='cutoff'),
        metrics=[mae, mape, rmse]
    )

    # 输出模型在测试集上的表现
    average_metrics = cv_evaluation.groupby('metric')['LSTM'].mean()
    print("LSTM微调后模型验证集结果：")
    print(average_metrics)

    average_metrics = cv_evaluation.groupby('metric')['NHITS'].mean()
    print("NHITS微调后模型验证集结果：")
    print(average_metrics)

    average_metrics = cv_evaluation.groupby('metric')['RNN'].mean()
    print("RNN微调后模型验证集结果：")
    print(average_metrics)

    # 拟合模型
    nf.fit(df=train)

    # 模型预测
    fcst_df = nf.predict()
    future_dt = pd.date_range(start='2023-01-01', end='2024-12-01', freq='MS')

    fcst_df['ds'] = future_dt

    test_full = pd.merge(test_df, fcst_df, 'left', ['unique_id', 'ds'])
    test_evaluation = evaluate(
        test_full,
        metrics=[mae, mape, rmse]
    )

    # 输出模型在验证集上的表现
    average_metrics = test_evaluation.groupby('metric')['LSTM'].mean()
    print("LSTM微调后模型测试集结果：")
    print(average_metrics)

    average_metrics = test_evaluation.groupby('metric')['NHITS'].mean()
    print("NHITS微调后模型测试集结果：")
    print(average_metrics)

    average_metrics = test_evaluation.groupby('metric')['RNN'].mean()
    print("RNN微调后模型测试集结果：")
    print(average_metrics)

    return fcst_df

# 示例用法
if __name__ == "__main__":
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # 关闭oneDNN优化
    logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

    # 导入数据
    df = pd.read_csv("data/DataSet.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    relation_df = df.iloc[:, 2:]
    test_set = pd.read_csv("data/TestSet.csv") #导入测试集
    test_set['Date'] = pd.to_datetime(test_set['Date'])

    # 高相关性特征因素筛选
    pearson = relation_df.corr()
    print("\n特征因素相关性：")
    print(pearson)
    high_relation = pearson.loc[:, "PV_capacity_China_GW"].nlargest(3)
    print("\n高相关性因素：")
    print(high_relation)

    # 筛选训练数据
    input_data = ['unique_id', 'Date'] + high_relation.index.tolist()
    input_df = df.loc[0:155, input_data]

    # 设置Python内置随机种子
    random.seed(42)
    # 设置NumPy随机种子
    np.random.seed(42)

    # TimeGPT设置API，备用API：
    nixtla_client = NixtlaClient(
        api_key='nixak-I4z5OgmYFR1NA8xkdIGWhDrRUO9uN2eeN0eLHCOfLdNMyibtAj7YaeVCoZmpbVXgKnTFqwHPg7IvZYZb'
    )

    # TimeGPT无微调模型
    TimeGPT_no_finetune = TimeGPT_no_finetune(input_df, test_set)

    # TimeGPT微调模型预测
    TimeGPT_finetune = TimeGPT(input_df, test_set)

    # Prophet模型预测
    Prophet_result = Prophet_fcst(input_df, test_set)

    # LSTM模型预测
    LSTM_NHITS_RNN_result = LSTM_NHITS_RNN(input_df, test_set)
    LSTM_NHITS_RNN_result = LSTM_NHITS_RNN_result .rename(columns={'ds':'Date'}, inplace=False)

    # # 导出所有模型的结果
    Final_results = [TimeGPT_no_finetune, TimeGPT_finetune, Prophet_result,LSTM_NHITS_RNN_result]
    final_df = reduce(lambda x, y: pd.merge(x, y, 'left', 'Date'), Final_results)
    final_df.to_csv('results/Result.csv', index=True, encoding='utf-8')