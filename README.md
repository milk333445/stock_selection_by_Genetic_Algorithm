# stock_selection_by_Genetic_Algorithm
# 多目標遺傳演算法優化 S&P 500 股票預測模型的特徵選擇
## 摘要
- 使用基因演算法選擇最佳的股票特徵指標，並應用於美股市場預測
- 使用了基因演算法和機器學習模型，以預測美股股票的價格波動與報酬率情況，經演化找出同時最佳化波動度和報酬率的股票特徵
- 最後透過波動度和收益率將股票分群，將排名前30%收益率與排名後30%波動度建構投資組合，並觀察總績效以及風險



## 資料
- 資料期間:2019/01/01~2019/10/30
- 股票特徵選擇
- 刪除cos、exp、ad、adosc、obv、sinh




| 股票特徵種類            | 該類別數量 |
| ----------------------- | ---------- |
| Price Volume Indicators |      6      |
| Cycle Indicators        |     5       |
| Math Operators          |      11      |
| Math Transform          |        15    |
| Momentum Indicators     |      30      |
| Pattern Recognition     |      61      |
| Price Transform         |      4      |
| Statistic Functions     |      9      |
| Volatility Indicators   |       3     |
| Volume Indicators       |      3      |
| Overlap Studies         | 17       |
- Cycle Indicators:週期性因素的分析(HT_TRENDLINE、KAMA、TRIX、DPO)
- Math Transform:測量資產價格的速度和幅度變化。它可以顯示股票或其他資產價格在一段時間內的變化趨勢(正弦（Sin）和餘弦（Cos）函數、自然對數（Ln）、算術平方根（Sqrt）)
- Momentum Indicators:測量資產價格的速度和幅度變化。它可以顯示股票或其他資產價格在一段時間內的變化趨勢(RSI、ROC、CMF、CCI、ADX)
- Pattern Recognition:提供了多種圖表形態的技術指標，可以用來偵測股票價格走勢中常見的模式
- Price Transform:Weighted Close Price（WCLPRICE）、Median Price（MEDPRICE）
- Statistic Functions:CORREL、LINEARREG、STDEV、VAR
- Volatility Indicators:衡量股票或市場波動性(Average True Range (ATR)、Bollinger Bands、Chaikin's Volatility (CHV))
- Volume Indicators:AD、CMF、OBV、VWAP、PVI
- Overlap Studies:用過去一段時間的股價資訊，計算出一個或多個股價的平均值或變動率，以及相應的上下限。這些指標可以用來幫助分析股價的走勢、支撐位和阻力位(EMA、MACD、SMA)

## 舉例
### 獲得最佳特徵並透過這些特徵預測隱含波動度和報酬
假設目前股票為:'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG'
![](https://i.imgur.com/WTvG6Jn.png)
### 建立投資組合
![](https://i.imgur.com/Ofe1S2J.png)
根據上表，amzn為最佳投組
## 訓練
- 資料期間:2019/01/01~2019/10/30
- 花費時間:684分鐘43.1秒
- Best fitness: (0.016266901120974033, 0.11610932727839841)
## 建立投資組合
### 第一組投組
1. 訓練+測試集:2019/10/31 ~ 2020/12/31，並計算每支股票平均波動度、平均報酬率
2. 建構2021/01/01~2021/04/30投組
3. 計算投組報酬率、最大回落比、calmar ratio
4. 並與持有大盤做比較
5. 持有投組:共25檔


|  持有標的 |     |      |      |     |
| --- | -------- | ---- | ---- | --- |
|  MRK| JNJ      | PG   | WMT  |  T  |
|  CVX| VZ       |   NKE|   LMT|  MO |
| GILD| SPGI     |   CME|  CCI | D   |
| WFC | BAX      |   FOX| BXP  | WBA |
|  MMC| DLR      | MSCI | RCL  | NDAQ|

7. 投組績效:
![](https://i.imgur.com/7TfCMZ6.png)
8. sp500績效
![](https://i.imgur.com/jgZLWdX.png)


| portfolio return & risk | portfolio | sp500 |
| -------- | -------- | -------- |
| return|    :apple:6.6%      |   6.53%   |
|     mdd     |     :apple:3.71%     |   6.02%   |
| calmar ratio|   :apple:1.78   |  1.08 |



### 第二組投組
1. 訓練+測試集:2020/02/25 ~ 2021/4/30，並計算每支股票平均波動度、平均報酬率
2. 建構2021/05/01~2021/08/31投組
3. 計算投組報酬率、最大回落比、calmar ratio
4. 並與持有大盤做比較
5. 持有投組:共17檔


|  持有標的 |     |      |      |     |
| --- | -------- | ---- | ---- | --- |
|  AAPL| ADBE    | CRM  | AMD  |  AMGN  |
|  HON| APD      |   EW|   DD|  AEP |
| REGN| BSX     |   PGR|  CNC | ANET   |
| VRSN | AME      |   |   |  |
7. 投組績效:
![](https://i.imgur.com/JNx4Lic.png)

8. sp500績效
![](https://i.imgur.com/JlnCZZn.png)

| portfolio return & risk | portfolio | sp500 |
| -------- | -------- | -------- |
| return|    :apple:10.54%      |   6.54%   |
|     mdd     |     :apple:2.03%     |   2.99%   |
| calmar ratio|   :apple:5.2   |  2.19 |




### 第三組投組
1. 訓練+測試集:2020/06/25 ~ 2021/8/31，並計算每支股票平均波動度、平均報酬率
2. 建構2021/09/01~2021/12/31投組
3. 計算投組報酬率、最大回落比、calmar ratio
4. 並與持有大盤做比較
5. 持有投組:共16檔


|  持有標的 |     |      |      |     |
| --- | -------- | ---- | ---- | --- |
|  DIS| VZ    | T  | INTC  |  CVS  |
|  VRTX| BAX   |   FOX|   ECL|  ANSS |
| STZ| KMB     |   CDW|  AIG | MCHP   |
| PENN |       |   |   |  |
6. 投組績效:
![](https://i.imgur.com/miVs0eG.png)
7. sp500績效:
![](https://i.imgur.com/xopP4xc.png)

| portfolio return & risk | portfolio | sp500 |
| -------- | -------- | -------- |
| return|    5.99%      |   :apple:9.62%   |
|     mdd     |     5.37%     |   :apple:3.06%   |
| calmar ratio|   1.12   |  :apple:3.15 |





### 第四組投組
1. 訓練+測試集:2020/10/25 ~ 2021/12/31，並計算每支股票平均波動度、平均報酬率
2. 建構2022/01/01~2022/04/15投組
3. 計算投組報酬率、最大回落比、calmar ratio
4. 並與持有大盤做比較
5. 持有投組:共21檔

| 持有標的 |     |      |      |      |
| -------- | --- | ---- | ---- | ---- |
| TSM      | MDT | C    | LMT  | GILD |
| IBM      | BA  | BDX  | SYK  | D    |
| DUK      | AEP | FOX  | SWKS | FISV |
| ALL      | HIG | PEAK | PSX  | RCL  |
|   PENN       |     |      |      |      |

6. 投組績效:
![](https://i.imgur.com/rGImfuc.png)
7. sp500績效:
![](https://i.imgur.com/HQfB7iw.png)

| portfolio return & risk | portfolio | sp500 |
| -------- | -------- | -------- |
| return|    :apple:5.12%      |   -2.62%   |
|     mdd     |     :apple:5.0%     |   10.74%   |
| calmar ratio|   :apple:1.02   |  -0.24 |

### 第五組投組
1. 訓練+測試集:2021/2/25 ~ 2022/4/15，並計算每支股票平均波動度、平均報酬率
2. 建構2022/4/16~2022/08/16投組
3. 計算投組報酬率、最大回落比、calmar ratio
4. 並與持有大盤做比較
5. 持有投組:共16檔


| 持有標的 |     |      |      |      |
| -------- | --- | ---- | ---- | ---- |
| C        | BA   | GM  | SPG | EBAY |
| FISV     | WBA  | VFC | DLR | ALB  |
| WY       | CTAS | KMB | COF | IR   |
| DLB      |      |     |     |      |

6. 投組績效:
![](https://i.imgur.com/em6YvZc.png)
7. sp500績效:
![](https://i.imgur.com/KECdqJJ.png)

| portfolio return & risk | portfolio | sp500 |
| -------- | -------- | -------- |
| return|    :apple:7.49%      |   3.83%   |
|     mdd     |     :apple:11.91%     |   13.7%  |
| calmar ratio|   :apple:0.63   |  0.28 |

### 第六組投組
1. 訓練+測試集:2021/6/25 ~ 2022/8/16，並計算每支股票平均波動度、平均報酬率
2. 建構2022/8/17~2022/12/31投組
3. 計算投組報酬率、最大回落比、calmar ratio
4. 並與持有大盤做比較
5. 持有投組:共19檔

| 持有標的 |      |      |      |      |
| -------- | ---- | ---- | ---- | ---- |
| MA       | CSCO | AMD  | QCOM | HON  |
| AXP      | GM   | DD   | LUV  | ROST |
| BKNG     | ECL  | ADI  | HLT  | DXCM |
| DOW      | AIG  | MCHP | STE  |      |
6. 投組績效:
![](https://i.imgur.com/NxvSPhT.png)
7. sp500績效:
![](https://i.imgur.com/7WSZZTL.png)

| portfolio return & risk | portfolio | sp500 |
| -------- | -------- | -------- |
| return|    :apple:-1.08%      |   -8.86%   |
|     mdd     |     18.35%     |   :apple:17.15%  |
| calmar ratio|   :apple:-0.06   |  -0.52 |




### 第七組投組
1. 訓練+測試集:2021/10/25 ~ 2022/12/31，並計算每支股票平均波動度、平均報酬率
2. 建構2023/1/1~2023/4/30投組
3. 計算投組報酬率、最大回落比、calmar ratio
4. 並與持有大盤做比較
5. 持有投組:共19檔


| 持有標的 |      |     |     |      |
| -------- | ---- | --- | --- | ---- |
| GOOGL    | GOOG | AMD | PM  | FIS  |
| AXP      | SPGI | PLD | CME | EW   |
| EBAY     | IQV  | EXC | BXP | ANSS |
| KEY      | PSA  | DOW | ESS | PKI  |
|   HAS       |   ZBRA   |     |     |      |
6. 投組績效:
![](https://i.imgur.com/GLNR55n.png)

7. sp500績效:
![](https://i.imgur.com/xPpcJOw.png)

| portfolio return & risk | portfolio | sp500 |
| -------- | -------- | -------- |
| return|    :apple:9.85%      |   8.97%   |
|     mdd     |     8.66%     |   :apple:6.18%  |
| calmar ratio|   1.14   |  :apple:1.45 |


## conclusion
### 回測期間:2021/01/01 ~ 2023/4/30
### 每四個月調整投資組合持股，共調整6次，7個投資組合，績效如下
:apple:**:代表優於sp500指數**
**投組一代表2021/01/01持有投組，依此類推**


| 投資組合 | return | mdd | calmar ratio |
| -------- | ------ | --- | ------------ |
| 投組1    |  :apple:6.6%  | :apple:3.71%    |   :apple:1.78   |
| 投組2    | :apple:10.54%   | :apple:2.03% |   :apple:5.2       |
| 投組3    |  5.99%  |  5.37%   |   1.12     |
| 投組4    |   :apple:5.12%  |  :apple:5.0% |    :apple:1.02     |
| 投組5    |   :apple:7.49%  |  :apple:11.91% |    :apple:0.63      |
| 投組6    |  :apple:-1.08%  |  18.35% |    :apple:-0.06    |
| 投組7    |  :apple:9.85%  | 8.66% |   1.14    |


:crown:**:代表優於投資組合績效** 
| 投資組合 | return | mdd | calmar ratio |
| -------- | ------ | --- | ------------ |
| sp500    |  6.53%  | 6.02% |    1.08     |
| sp500    |  6.54%   | 2.99%  |   2.19      |
| sp500    |  :crown: 9.62%  | :crown: 3.06% |    :crown: 3.15    |
| sp500    |  -2.62%  | 10.74% |     -0.24    |
| sp500    |  3.83%   |  13.7%|    0.28     |
| sp500    |  -8.86% | :crown: 17.15%  |   -0.52    |
| sp500    |   8.97%  |  :crown: 6.18% |   :crown: 1.45     |


```python
#抓美股資料
import pandas_datareader.data as pdr
import datetime
import pandas as pd

#技術指標
import talib
all_ta_label = talib.get_functions()
len(all_ta_label) #有158類
#分類
all_ta_groups = talib.get_function_groups()
all_ta_groups.keys()
all_ta_groups['Cycle Indicators']
table = pd.DataFrame({'技術指標類別名稱:':list(all_ta_groups.keys()),
                      '該類別指標總數:':list(map(lambda x: len(x), all_ta_groups.values()))})

#Abstract API
from talib import abstract
#打包成函數
def get_stock_data(stock_code, start_date, end_date):
    #取得股票資料
    df = pdr.get_data_yahoo([stock_code], start_date, end_date)
    df.columns = [col.lower() for col in df.columns]
    ta_list = talib.get_functions()
    for x in ta_list:
        try:
            # x 為技術指標的代碼，透過迴圈填入，再透過 eval 計算出 output
            output = eval('abstract.'+x+'(df)')
            # 如果輸出是一維資料，幫這個指標取名為 x 本身；多維資料則不需命名
            output.name = x.lower() if type(output) == pd.core.series.Series else None
            # 透過 merge 把輸出結果併入 df DataFrame
            df = pd.merge(df, pd.DataFrame(output), left_on = df.index, right_on = output.index)
            df = df.set_index('key_0')
        except:
            print(x)
    # 將股票價格和報酬率向後滯後一期
    df['return'] = df['adj close'].pct_change()
    df['change'] = (df['adj close'] > df['open']).astype(int)
    # 刪除包含缺失值的行
    data = df
    data = data.drop(['acos', 'asin'], axis=1)
    data = data.dropna()
    data = data.reset_index(drop=True)
    data = data.astype('float')

    return data

#寫機器學習函數
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import copy
def predict_stock_price_ver1(data, keep_cols):
    # 根据 keep_cols 的值删除需要删除的列
    # 用除了漲跌的欄位作為自變數，去預測股價漲跌
    df = copy.deepcopy(data)
    df.fillna(0, inplace=True)
    df['stddev_lagging'] = df['stddev'].shift(-1)
    df = df.dropna()
    scaler = StandardScaler()
    X = df.drop(['stddev_lagging'], axis=1)
    drop_cols = [col for i, col in enumerate(X.columns) if not keep_cols[i]]
    X = X.drop(drop_cols, axis=1)
    X = X.dropna()
    X = scaler.fit_transform(X)
    y = df['stddev_lagging'].values.reshape(-1,1)
    y = scaler.fit_transform(y)
    # 分割資料集為訓練集和測試集
    X_train, y_train = X, y
    # 創建隨機森林模型
    rf = RandomForestRegressor(random_state=10)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_train)
    # 計算測試集的 AUC 分數
    mse_score = mean_squared_error(y_train, y_pred)
    return mse_score
def predict_stock_price_ver2(data, keep_cols):
    # 根据 keep_cols 的值删除需要删除的列
    # 用除了漲跌的欄位作為自變數，去預測股價漲跌
    df = copy.deepcopy(data)
    df.fillna(0, inplace=True)
    df['return_lagging'] = df['return'].shift(-1)
    df = df.dropna()
    scaler = StandardScaler()
    X = df.drop(['return_lagging'], axis=1)
    drop_cols = [col for i, col in enumerate(X.columns) if not keep_cols[i]]
    X = X.drop(drop_cols, axis=1)
    X = X.dropna()
    X = scaler.fit_transform(X)
    y = df['return_lagging'].values.reshape(-1,1)
    y = scaler.fit_transform(y)
    # 分割資料集為訓練集和測試集
    X_train, y_train = X, y
    # 創建隨機森林模型
    rf = RandomForestRegressor(random_state=10)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_train)
    # 計算測試集的 AUC 分數
    mse_score = mean_squared_error(y_train, y_pred)
    return mse_score
def predict_sp500_ver1(sp500_top_30, keep_cols):
    data_combined = None
    for stock_code in sp500_top_30:
        try:
            #取得股票資料
            start_date = datetime.datetime(2020, 1, 1)
            end_date = datetime.datetime(2022, 12, 31)
            data = get_stock_data(stock_code, start_date, end_date)
            if data_combined is None:
                data_combined = data
            else:
                data_combined = data_combined.append(data)
            print(f'股票代碼{stock_code}資料抓取成功')
        except Exception as e:
            print(f"股票代碼 {stock_code} 資料抓取失敗: {e}")
            continue
    if data_combined is not None:
        avg_mse_score_std = predict_stock_price_ver1(data_combined, keep_cols)
        print("預測完成")
        return avg_mse_score_std
    else:
        print("沒有可用的股票資料")
        return None
#算平均mse_ret分數
def predict_sp500_ver2(sp500_top_30, keep_cols):
    data_combined = None
    for stock_code in sp500_top_30:
        try:
            #取得股票資料
            start_date = datetime.datetime(2020, 1, 1)
            end_date = datetime.datetime(2022, 12, 31)
            data = get_stock_data(stock_code, start_date, end_date)
            if data_combined is None:
                data_combined = data
            else:
                data_combined = data_combined.append(data)
            print(f'股票代碼{stock_code}資料抓取成功')
        except Exception as e:
            print(f"股票代碼 {stock_code} 資料抓取失敗: {e}")
            continue
    if data_combined is not None:
        avg_mse_score_ret = predict_stock_price_ver2(data_combined, keep_cols)
        print("預測完成")
        return avg_mse_score_ret
    else:
        print("沒有可用的股票資料")
        return None
    
#寫基因演算法
import deap
from deap import algorithms, base, creator, tools
import random
random.seed(42)
#設定基因和適應度函數
creator.create('FitnessMin', base.Fitness, weights=(-1.0, -1.0))
creator.create('Individual', list, fitness=creator.FitnessMin)

#設定工具箱
toolbox = base.Toolbox()
toolbox.register('attr_bool', random.randint, 0, 1)
toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr_bool, n=180)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
toolbox.register('mate', tools.cxTwoPoint)
toolbox.register('mutate', tools.mutFlipBit, indpb=0.05)

#設定適應度函數
def evaluate_individual(individual):
    keep_cols = individual
    sp500_stock_code = ['AAPL', 'MSFT']
    #計算mse平均值
    avg_mse_score_std = predict_sp500_ver1(sp500_stock_code, keep_cols)
    avg_mse_score_ret = predict_sp500_ver2(sp500_stock_code, keep_cols)

    # 返回一個元組，包含兩個目標
    return (avg_mse_score_std, avg_mse_score_ret)

toolbox.register('evaluate', evaluate_individual)
toolbox.register('select', tools.selNSGA2)
#設定演化參數
population_size = 100
number_of_generations = 50
probability_of_crossover = 0.5
probability_of_mutation = 0.2

#初始化族群
population = toolbox.population(n=population_size)

#執行演化
for generation in range(number_of_generations):
    #交配變異
    offspring = algorithms.varAnd(population, toolbox, cxpb=probability_of_crossover, mutpb=probability_of_mutation)
    #計算每個個體的適應度
    fitnesses = map(toolbox.evaluate, offspring)
    #跟新適應度
    for ind, fit in zip(offspring, fitnesses):
        ind.fitness.values = fit
    #跟新種群
    population = toolbox.select(offspring, k=population_size)

#獲得最佳解
best_individual = tools.selBest(population, k=1)[0]
best_fitness = best_individual.fitness.values

print('Best individual:', best_individual)
print('Best fitness:', best_fitness)

#最佳化之後的特徵選擇
def predict_std(data, keep_cols):
    # 根据 keep_cols 的值删除需要删除的列
    # 用除了漲跌的欄位作為自變數，去預測股價漲跌
    df = copy.deepcopy(data)
    df.fillna(0, inplace=True)
    df['stddev_lagging'] = df['stddev'].shift(-1)
    df = df.dropna()
    scaler = StandardScaler()
    X = df.drop(['stddev_lagging'], axis=1)
    drop_cols = [col for i, col in enumerate(X.columns) if not keep_cols[i]]
    X = X.drop(drop_cols, axis=1)
    X = X.dropna()
    X = scaler.fit_transform(X)
    y = df['stddev_lagging'].values.reshape(-1,1)
    y = scaler.fit_transform(y)
    # 分割資料集為訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    # 創建隨機森林模型
    rf = RandomForestRegressor(random_state=10, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    # 計算平均標準差（stddev）分數
    avg_stddev = np.mean(y_pred)
    return avg_stddev
def predict_ret(data, keep_cols):
    # 根据 keep_cols 的值删除需要删除的列
    # 用除了漲跌的欄位作為自變數，去預測股價漲跌
    df = copy.deepcopy(data)
    df.fillna(0, inplace=True)
    df['return_lagging'] = df['return'].shift(-1)
    df = df.dropna()
    scaler = StandardScaler()
    X = df.drop(['return_lagging'], axis=1)
    drop_cols = [col for i, col in enumerate(X.columns) if not keep_cols[i]]
    X = X.drop(drop_cols, axis=1)
    X = X.dropna()
    X = scaler.fit_transform(X)
    y = df['return_lagging'].values.reshape(-1,1)
    y = scaler.fit_transform(y)
    # 分割資料集為訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    # 創建隨機森林模型
    rf = RandomForestRegressor(random_state=10, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    avg_return = np.mean(y_pred)
    return avg_return

def predict_sp500(sp500_top_250, best_individual):
    data_dict = {}
    for stock_code in sp500_top_250:
        try:
            #取得股票資料
            start_date = datetime.datetime(2018, 10, 31)
            end_date = datetime.datetime(2020, 12, 31)
            data = get_stock_data(stock_code, start_date, end_date)
            # 根據最佳化結果進行特徵選擇
            keep_cols = best_individual
            #進行預測
            avg_std_score = predict_std(data, keep_cols)
            avg_ret_score = predict_ret(data, keep_cols)
            
            print(f"股票代碼 {stock_code} 預測成功")
        except Exception as e:
            print(f"股票代碼 {stock_code} 預測失敗: {e}")
            continue

        #預測結果添加到字典中
        data_dict[stock_code] = {'隱含波動度':avg_std_score, '隱含報酬率':avg_ret_score}
    df = pd.DataFrame.from_dict(data_dict, orient="index")
    

    return df

def filter_stocks(df):
    volatility_quantiles = df['隱含波動度'].quantile([0.3, 0.7])
    df['波動度分類'] = pd.cut(df['隱含波動度'], bins=[-np.inf, volatility_quantiles.iloc[0],
                                            volatility_quantiles.iloc[1],
                                            np.inf],labels=['低波動', '中等波動', '高波動'])
    returns_quantiles = df['隱含報酬率'].quantile([0.3, 0.7])
    df['報酬率分類'] = pd.cut(df['隱含報酬率'], bins=[-np.inf, returns_quantiles.iloc[0], returns_quantiles.iloc[1], np.inf], labels=['低報酬', '中等報酬', '高報酬'])

    # 選取低波動且高報酬的股票代碼
    filtered_stocks = df[(df['波動度分類'] == '低波動') & (df['報酬率分類'] == '高報酬')].index.tolist()

    return filtered_stocks


#視覺化投組資料
def get_stock_data_only(stock_code, start_date, end_date):
    #取得股票資料
    df = pdr.get_data_yahoo([stock_code], start_date, end_date)
    df.columns = [col.lower() for col in df.columns]
    df['return'] = df['adj close'].pct_change()
    df = df.dropna()
    return df

def plot_portfolio_returns(filtered_stocks, start_date, end_date):
    portfolio_returns = []
    portfolio_returns_sp_500=[]
    stock_code_sp_500 = 'SPY'
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    for date in dates:
        # 取得當天的股票資料
        stock = []
        for stock_code in filtered_stocks:
            stock_data = get_stock_data_only(stock_code, date - timedelta(days=1), date + timedelta(days=1))
            if not stock_data.empty:
                stock.append(stock_data.iloc[0]['return'])
            else:
                stock.append(0)     
        avg_return = sum(stock) / len(stock)
        portfolio_returns.append(avg_return)

        stock_data_sp_500 = get_stock_data_only(stock_code_sp_500, date - timedelta(days=1), date + timedelta(days=1))
        if not stock_data_sp_500.empty:
            return_sp_500 = stock_data_sp_500.iloc[0]['return']
        else:
            return_sp_500 = 0
        portfolio_returns_sp_500.append(return_sp_500)
   
    df = pd.DataFrame({'profit':np.cumsum(portfolio_returns), 'sp 500 return':np.cumsum(portfolio_returns_sp_500)}, index=dates)
    df.plot(grid=True, figsize=(16, 6))
    plt.legend()
    plt.ylabel('Profit')
    plt.xlabel('date')
    plt.title('Portfolio Returns')
    plt.show()
    df = df.reset_index(drop=True)

    # 計算總利潤、股權、回撤百分比和回撤
    df['equity'] = df['profit']+1
    df['drawdown_percent'] = (df['equity'] / df['equity'].cummax())-1
    df['drawdown'] = df['equity'] - df['equity'].cummax()
    fig, ax = plt.subplots(figsize = (16,6))
    high_index = df[df['profit'].cummax() == df['profit']].index
    df['profit'].plot(label = 'Total Profit', ax = ax, c = 'r', grid=True)
    plt.fill_between(df['drawdown'].index, df['drawdown'], 0, facecolor = 'r', label = 'Drawdown', alpha = 0.5)
    plt.scatter(high_index, df['profit'].loc[high_index], c = '#02ff0f', label = 'High')
    plt.legend()
    plt.ylabel('Accumulated Profit Return')
    plt.xlabel('Time')
    plt.title('Portfolio Profit & Drawdown',fontsize  = 16)
    plt.show()
    profit = df['profit'].iloc[-1]
    mdd = abs(df['drawdown_percent'].min())
    calmarratio = profit/mdd
    print("portfolio return & risk ")
    print(f'return: ${np.round(profit, 4)*100}%')
    print(f'mdd: {np.round(mdd, 4)*100}%')
    print(f'calmar ratio: {np.round(calmarratio, 2)}')

    #計算持有大盤
    df_sp500 = pd.DataFrame({'profit':np.cumsum(portfolio_returns_sp_500)})
    df_sp500['equity'] = df_sp500['profit']+1
    df_sp500['drawdown_percent'] = (df_sp500['equity'] / df_sp500['equity'].cummax())-1
    df_sp500['drawdown'] = df_sp500['equity'] - df_sp500['equity'].cummax()
    fig, ax = plt.subplots(figsize = (16,6))
    high_index = df_sp500[df_sp500['profit'].cummax() == df_sp500['profit']].index
    df_sp500['profit'].plot(label = 'Total Profit', ax = ax, c = 'r', grid=True)
    plt.fill_between(df_sp500['drawdown'].index, df_sp500['drawdown'], 0, facecolor = 'r', label = 'Drawdown', alpha = 0.5)
    plt.scatter(high_index, df_sp500['profit'].loc[high_index], c = '#02ff0f', label = 'High')
    plt.legend()
    plt.ylabel('Accumulated Profit Return')
    plt.xlabel('Time')
    plt.title('SP500 Profit & Drawdown',fontsize  = 16) 
    plt.ylim(top=df['profit'].max())
    plt.show()
    profit = df_sp500['profit'].iloc[-1]
    mdd = abs(df_sp500['drawdown_percent'].min())
    calmarratio = profit/mdd
    print("sp 500 return & risk ")
    print(f'return: ${np.round(profit, 4)*100}%')
    print(f'mdd: {np.round(mdd, 4)*100}%')
    print(f'calmar ratio: {np.round(calmarratio, 2)}')
    
    return portfolio_returns, portfolio_returns_sp_500
```
