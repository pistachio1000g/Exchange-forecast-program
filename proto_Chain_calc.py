import numpy as np
import chainer
import chainer.functions as F
import chain_Library
from chainer import serializers
import matplotlib.pyplot as plt
from chainer import serializers
from chainer.cuda import to_gpu
from chainer.cuda import to_cpu

# 推論用ファイル

#モデルの定義とデータ読み込み
gpu_id = 0
model =chain_Library.NeuralNet(784, 2)
serializers.load_npz('fxresult/model_epoch-11120',model)
model.to_gpu(gpu_id)

#データの取得
csvEx = chain_Library.CsvExchange("gbpjpy_d_.csv")
x_data, tzs, tbzs = csvEx.getData(25, 11733, 11990)      #テストデータを取得する
txs, tts = x_data._datasets 

#推論
x = to_gpu(np.copy(txs))
y = model.call(x)
y = to_cpu(y.data)
y_result = y.argmax(axis=1)

#実際に売買した場合の結果を計算
profit = 0
list = []
for i in range(len(y)):
    #出力
    print("tzs=" + str(tzs[i]) + " tbzs = " + str(tbzs[i]))
    
    #機械学習の判断に基づくトレード結果
    if y[i][0] > y[i][1]:
        #ロングサイン
        tmp = tzs[i] - tbzs[i]
        print("損益 = " + str(tzs[i] - tbzs[i]))
    elif y[i][0] < y[i][1]:
        #ショートサイン
        tmp = tbzs[i] - tzs[i]
        print("損益 = " + str(tbzs[i] - tzs[i]))
    else:
        tmp = 0

    # 損切りを少し乱暴にセットしておく    
    if tmp < 0:
        tmp = -0.4
    
    #集計
    profit = profit + tmp   

    #リストにセット    
    list.append(profit) 

#認識精度の計算をする    
accuracy, loss = chain_Library.check_accuracy(model, txs, tts, gpu_id)
    
#結果を表示する
print("結果=" + str(loss))
print("結果=" + str(accuracy))
print("結果=" + str(profit))
print("【x】 = " + str(x.shape))
plt.plot(list)
plt.show()
