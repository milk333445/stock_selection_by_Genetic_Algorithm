#抓美股資料
import pandas_datareader.data as pdr
import datetime
import pandas as pd
import yfinance as yf
yf.pdr_override()

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
    df['change_lagging'] = df['change'].shift(-1)
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


def predict_stock_price_ver1(data, keep_cols):
    # 根据 keep_cols 的值删除需要删除的列
    # 用除了漲跌的欄位作為自變數，去預測股價漲跌
    data.fillna(0, inplace=True)
    scaler = StandardScaler()
    X = data.drop(['change_lagging'], axis=1)
    drop_cols = [col for i, col in enumerate(X.columns) if not keep_cols[i]]
    X = X.drop(drop_cols, axis=1)
    X = X.dropna()
    X = scaler.fit_transform(X)
    y = data['change_lagging']
    y = y.dropna()
    # 分割資料集為訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    # 創建隨機森林模型
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    # 計算測試集的 AUC 分數
    auc_score = roc_auc_score(y_test, y_pred)
    return auc_score



#算平均auc分數
def predict_sp500_top_30(sp500_top_30, keep_cols):
    auc_scores = []
    for stock_code in sp500_top_30:
        #取得股票資料
        start_date = datetime.datetime(2020, 1, 1)
        end_date = datetime.datetime.today()
        data = get_stock_data(stock_code, start_date, end_date)
        #進行預測
        auc_score = predict_stock_price_ver1(data, keep_cols)
        auc_scores.append(auc_score)
    avg_auc_score = sum(auc_scores)/len(auc_scores)
    return avg_auc_score



#寫基因演算法
import deap
from deap import algorithms, base, creator, tools
import random
random.seed(42)

# 定義基因型和適應度函數
creator.create('FitnessMax', base.Fitness, weights = (1.0,))
creator.create('Individual', list, fitness=creator.FitnessMax)
    

#定義適應度函數
def fitness_function(gene):
    # 將基因轉換成 keep_cols
    keep_cols = gene
    sp500_stock_code = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG',
                 'NVDA', 'JPM', 'JNJ', 'V', 'PG',
                 'DIS', 'PYPL', 'BAC',
                'INTC', 'VZ', 'KO', 'CRM', 'PFE', 'CMCSA', 'T',
                'CSCO', 'MRK', 'PEP']
    avg_auc_score = predict_sp500_top_30(sp500_stock_code, keep_cols)
    return (avg_auc_score,) #在 DEAP 中， FitnessMax 物件的 values 屬性必須是 tuple 型態，所以在定義適應度函數時必須回傳一個 tuple


#設定基因跟適應度函數
toolbox = base.Toolbox() #創建一個新的工具箱 (toolbox)
toolbox.register("attr_bool", random.randint, 0, 1) #定義基因型為一個二進制數，即0或1
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=180)
#建立一個新的基因型(Individual)。這個方法的參數包括
#creator.Individual: 指定新建立的基因型是哪個類別
#toolbox.attr_bool: 指定基因的資料型別是什麼，這裡是布林值(0或1)
#n=180: 指定基因的長度
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#註冊一個 population 函數到工具箱 toolbox 中
#種群中包含多個基因個體，每個個體都是由 creator.Individual 創建的，其基因由 toolbox.individual 函數建立
#tools.initRepeat 函數會重複地調用 toolbox.individual 函數 n 次，以創建一個由 n 個基因個體組成的列表，並將其封裝在一個列表中


toolbox.register('evaluate', fitness_function)
toolbox.register('mate', tools.cxTwoPoint)
toolbox.register('mutate', tools.mutFlipBit, indpb=0.05)
toolbox.register('select', tools.selTournament, tournsize=3)

#設定演化參數
population_size = 100
number_of_generations = 50
probability_of_crossover = 0.5
probability_of_mutation = 0.2

#初始化
populaiton = toolbox.population(n=population_size)
#執行演化
for generation in range(number_of_generations):
    offspring = algorithms.varAnd(populaiton, toolbox, cxpb=probability_of_crossover, mutpb=probability_of_mutation)
    #使用 varAnd 方法來進行交配(mating)和變異(mutation)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring): #這行程式碼是用來將新一代個體的適應度設定到個體上
        ind.fitness.values = fit
    populaiton = toolbox.select(offspring, k=len(populaiton))

#獲得最優解
best_individual = tools.selBest(populaiton, k=1)[0]
best_auc_score = best_individual.fitness.values[0]
best_keep_cols = best_individual