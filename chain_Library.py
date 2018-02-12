import numpy as np
import pandas as pd
import chainer
from chainer import Chain, Variable
from chainer.datasets import tuple_dataset
import chainer.functions as F
import chainer.links as L
from chainer.cuda import to_gpu
from chainer.cuda import to_cpu

#ニュートラルネットワーククラス
class NeuralNet(chainer.Chain):                 
    def __init__(self, n_units, n_out):     
        super(NeuralNet, self).__init__(        
            l1=L.Linear(None, n_units),         
            l2=L.Linear(n_units, n_units),
            l3=L.Linear(n_units, n_units),
            l4=L.Linear(n_units, n_units),
            l5=L.Linear(n_units, n_units),
            l6=L.Linear(n_units, n_out), 
            bn1=L.BatchNormalization(n_units),
        )

    def __call__(self, x):
        h1 = F.dropout(F.relu(self.bn1(self.l1(x))), 0.7)
        h2 = F.dropout(F.relu(self.bn1(self.l2(h1))), 0.6)
        h3 = F.dropout(F.relu(self.bn1(self.l3(h2))), 0.5)
        h4 = F.dropout(F.relu(self.bn1(self.l4(h3))), 0.4)
        h5 = F.dropout(F.relu(self.bn1(self.l5(h4))), 0.4)
        h6 = F.softmax(self.l6(h5))
        
        return h6
    
    def call(self, x):
        h1 = F.relu(self.bn1(self.l1(x)))
        h2 = F.relu(self.bn1(self.l2(h1)))
        h3 = F.relu(self.bn1(self.l3(h2)))
        h4 = F.relu(self.bn1(self.l4(h3)))
        h5 = F.relu(self.bn1(self.l5(h4)))
        h6 = F.softmax(self.l6(h5))
        
        return h6

# CSVデータ読み込みクラス
class CsvExchange:    
    def __init__(self, filename):
        # CSVデータを読みこむ
        data_dir = "./"
        data = pd.read_csv(data_dir + filename) # FXデータの読み込み
        self.csvData = np.array(data) # pnumpyに変換

    # 何日前までのデータを使用するのか
    # csvデータの読み込み開始位置
    # csvデータの読み込み終了位置
    def getData(self, day_ago, s_size, e_size):
        print(self.csvData[s_size][0])    #25日進んでデータを作っていく開始日
        print(self.csvData[e_size][0])    #25日進んでデータを作っていく開始日

        #為替データの使う部分だけトリミング
        data2 = np.copy(self.csvData[:,1:5])

        #訓練データの配列を確保
        # 1.データ
        x_data = []    

        # 2.ラベル
        t_data = []        

        # 3.当日の終値セットするための配列
        end_price = []   

        # 4.前日の終値セットするための配列
        bend_price = []    

        # ループの開始位置は読み込み開始位置の+25日後から
        j = 0
        for i in range(s_size, e_size + 1):
            # 当日の終値
            end_price.append(np.copy(data2[i][3]))      

            # 前日の終値 
            bend_price.append(np.copy(data2[i - 1][3])) 

            #前日までのday_agoぶんのデータをセットする。
            a = i - day_ago
            tmp_data = np.copy(data2[a:i, 3])              
            #tmp_data = np.reshape(tmp_data, day_ago * 4)    #2次元配列を1次元配列に
            x_data.append(np.copy(tmp_data))
            
            # 答え。前日の終値よりプラスになったら0、マイナスなら1
            if bend_price[j] < end_price[j]:    
                t_data.append(0)
            else:
                t_data.append(1)

            j += 1
                
        # npに変換する
        x_data = np.array(x_data)
        t_data = np.array(t_data)
        end_price = np.array(end_price)
        bend_price = np.array(bend_price)

        # float32に変換する
        x_data = x_data.astype(np.float32)
        t_data = t_data.astype(np.int32)
        end_price = end_price.astype(np.float32)
        bend_price = bend_price.astype(np.float32)
        
        # データセットに変換する
        data = tuple_dataset.TupleDataset(x_data, t_data)
        
        #return x_data, t_data, end_price
        return data, end_price, bend_price

#認識精度の計算をする
def check_accuracy(model,   #ニューラルネットワーク    
                      xs,   #認識するデータ
                      ts,   #答えのラベル
                  gpu_id):  

    # GPUにセット
    x = to_gpu(np.copy(xs))

    #普通に訓練を行う
    ys = model.call(x)                         
    ys = to_cpu(ys.data)
    
    #損失関数を計算する
    loss = F.softmax_cross_entropy(ys, ts)

    #ys.dataの配列の中で一番大きい値(一番確率の高い値)のインデックスを取得する
    ys = np.argmax(ys.data, axis=1)            
    ys = Variable(np.array(ys, "i"))

    #答えラベルと計算結果で一番高いものが＝ならばtrueにする
    cors = (ys.data == ts.data)                

    #trueの合計を計算する
    num_cors = sum(cors)                       

    #正解数/tsの大きさが認識の精度
    accuracy = num_cors / ts.shape[0]          
    
    return accuracy, loss