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
- 假設目前股票為:'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG'
![](https://i.imgur.com/WTvG6Jn.png)
### 建立投資組合
![](https://i.imgur.com/Ofe1S2J.png)
- 根據上表，amzn為最佳投組
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


| 投資組合 | return | mdd | calmar ratio |
| -------- | ------ | --- | ------------ |
| sp500    |  6.53%  | 6.02% |    1.08     |
| sp500    |  6.54%   | 2.99%  |   2.19      |
| sp500    |  :apple:9.62%  | :apple:3.06% |    :apple:3.15    |
| sp500    |  -2.62%  | 10.74% |     -0.24    |
| sp500    |  3.83%   |  13.7%|    0.28     |
| sp500    |  -8.86% | :apple:17.15%  |   -0.52    |
| sp500    |   8.97%  |  :apple:6.18% |   :apple:1.45     |
