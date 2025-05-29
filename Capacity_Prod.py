import numpy as np
import pandas as pd

# 读取预测数据
df = pd.read_csv("results/Yearly_PV_Capacity.csv")
PV_Capacity = df.loc[:, 'TimeGPT']

Monte_Carlo = 10000

# 初始化数组
Waste = np.zeros((53, Monte_Carlo))
ADD = np.zeros((53, Monte_Carlo))


for idx_MC in range(Monte_Carlo):

    Expect_Life = 25 + np.random.normal(loc=0, scale=1) * 25 * 0.05  # 光伏期望寿命
    Factor = 5.3759 + np.random.normal(loc=0, scale=1) * 5.3759 * 0.05   # Weibull分布形状参数
    x = np.arange(0, 101)

    # 计算Weibull分布概率
    WeibullP = 1 - np.exp(-((x / Expect_Life) ** Factor))
    WeibullP = WeibullP.reshape(-1, 1)  # 转换为列向量
    WeibullP = WeibullP[1:] - WeibullP[:-1]

    # 循环运算每一年的废弃量和生产量
    for idx in range(51):

        next_idx = idx + 1
        # 计算光伏年生产量
        ADD[next_idx, idx_MC] = (PV_Capacity[next_idx] - PV_Capacity[idx] + Waste[next_idx, idx_MC])

        zero_pad = np.zeros((next_idx, 1))
        WeibullP_segment = WeibullP[:53 - next_idx]
        WasteP = np.vstack((zero_pad, WeibullP_segment))

        # 计算光伏累计废弃量
        Waste[:, idx_MC:idx_MC + 1] = ADD[idx, idx_MC] * WasteP + Waste[:, idx_MC:idx_MC + 1]

# 计算蒙特卡洛结果的上分位点
Waste_mean = Waste.mean(axis=1)
Waste_up95 = np.percentile(Waste, 95, axis=1)
Waste_down95 = np.percentile(Waste, 5, axis=1)
Waste_result = np.vstack((Waste_mean[:52], Waste_up95[:52], Waste_down95[:52]))

# 输出计算结果
Result = pd.DataFrame(Waste_result.T, columns=["Waste", "up95", "down95"])
Result['Date'] = df['Date']
Result.to_csv("results/Prod_Waste_Result.csv", index=False)