# %%

#抓美股資料
import pandas_datareader.data as pdr
import datetime
import pandas as pd
import yfinance as yf
yf.pdr_override()

# %%
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
table

# %%
#Abstract API
from talib import abstract

# %%
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

# %%
start_date = datetime.datetime(2020, 1, 1)
end_date = datetime.datetime.today()
stock_code = 'UNH'
data = get_stock_data(stock_code, start_date, end_date)
data

# %%
#寫機器學習函數
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import copy

# %%
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    # 創建隨機森林模型
    rf = RandomForestRegressor(random_state=10)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    # 計算測試集的 AUC 分數
    mse_score = mean_squared_error(y_test, y_pred)
    return mse_score

# %%
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    # 創建隨機森林模型
    rf = RandomForestRegressor(random_state=10)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    # 計算測試集的 AUC 分數
    mse_score = mean_squared_error(y_test, y_pred)
    return mse_score

# %%
keep_cols = np.random.randint(2, size=180)
start_date = datetime.datetime(2020, 1, 1)
end_date = datetime.datetime.today()
stock_code = 'AAPL'
data = get_stock_data(stock_code, start_date, end_date)
data

# %%
#算平均mse_std分數
def predict_sp500_ver1(sp500_top_30, keep_cols):
    mse_scores = []
    for stock_code in sp500_top_30:
        try:
            #取得股票資料
            start_date = datetime.datetime(2020, 1, 1)
            end_date = datetime.datetime(2022, 12, 31)
            data = get_stock_data(stock_code, start_date, end_date)
            #進行預測
            mse_score = predict_stock_price_ver1(data, keep_cols)
            mse_scores.append(mse_score)
            print(f"股票代碼 {stock_code} 預測成功")
        except Exception as e:
            print(f"股票代碼 {stock_code} 預測失敗: {e}")
            continue
    avg_mse_score_std = sum(mse_scores)/len(mse_scores)
    return avg_mse_score_std

# %%
import numpy as np
keep_cols = np.random.randint(2, size=180)

# %%
sp500_top_30 = ['AAPL']

# %%
predict_sp500_ver1(sp500_top_30, keep_cols)

# %%
#算平均mse_ret分數
def predict_sp500_ver2(sp500_top_30, keep_cols):
    mse_scores = []
    for stock_code in sp500_top_30:
        try:
            #取得股票資料
            start_date = datetime.datetime(2020, 1, 1)
            end_date = datetime.datetime(2022, 12, 31)
            data = get_stock_data(stock_code, start_date, end_date)
            #進行預測
            mse_score = predict_stock_price_ver2(data, keep_cols)
            mse_scores.append(mse_score)
            print(f"股票代碼 {stock_code} 預測成功")
        except Exception as e:
            print(f"股票代碼 {stock_code} 預測失敗: {e}")
            continue
    avg_mse_score_ret = sum(mse_scores)/len(mse_scores)
    return avg_mse_score_ret

# %%
predict_sp500_ver2(sp500_top_30, keep_cols)

# %% [markdown]
# #雙目標最佳化演算法

# %%
#寫基因演算法
import deap
from deap import algorithms, base, creator, tools
import random
random.seed(42)

# %% [markdown]
# ##第一個適應度函數

# %%
#定義適應度函數1
def fitness_function1(gene):
    # 將基因轉換成 keep_cols
    keep_cols = gene
    sp500_stock_code = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG',
                 'NVDA', 'JPM', 'JNJ', 'V', 'PG',
                 'DIS', 'PYPL', 'BAC',
                'INTC', 'VZ', 'KO', 'CRM', 'PFE', 'CMCSA', 'T',
                'CSCO', 'MRK', 'PEP']
    avg_mse_score_std = predict_sp500_ver1(sp500_stock_code, keep_cols)
    return (avg_mse_score_std,) #在 DEAP 中， FitnessMax 物件的 values 屬性必須是 tuple 型態，所以在定義適應度函數時必須回傳一個 tuple

# %%
#定義適應度函數2
def fitness_function2(gene):
    # 將基因轉換成 keep_cols
    keep_cols = gene
    sp500_stock_code = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG',
                 'NVDA', 'JPM', 'JNJ', 'V', 'PG',
                 'DIS', 'PYPL', 'BAC',
                'INTC', 'VZ', 'KO', 'CRM', 'PFE', 'CMCSA', 'T',
                'CSCO', 'MRK', 'PEP']
    avg_mse_score_ret = predict_sp500_ver2(sp500_stock_code, keep_cols)
    return (avg_mse_score_ret,)

# %% [markdown]
# ##雙目標函數

# %%
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







# %%



