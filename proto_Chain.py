import numpy as np
import pandas as pd
import chainer
import chainer.functions as F
import chainer.links as L
import chain_Library
from chainer import Chain, Variable, optimizers, training
from chainer.datasets import tuple_dataset
from chainer import iterators
from chainer.training import extensions
from chainer import serializers
    
#メイン(訓練用ファイル)

# クラスを定義
csvEx = chain_Library.CsvExchange("gbpjpy_d_.csv")

#訓練データを取得する
train, zs, _ = csvEx.getData(25, 8884, 11732)

#テストデータを取得する
test, tzs, _ = csvEx.getData(25, 11733, 11990)      

# バッチサイズを決める
batchsize = 150

# テストデータのバッチサイズを決める
t_batchsize = 11990-11733               

#訓練イテレーター
train_iter = iterators.SerialIterator(train, batchsize)            

#テストイテレーター
test_iter = iterators.SerialIterator(test, t_batchsize, False, False) #テストデータ

# モデルの設定
model = chain_Library.NeuralNet(784, 2)
gpu_id = 0
model = L.Classifier(model)
model.to_gpu(gpu_id) 

# optimizerの学習方法はadamを使用する
optimizer = optimizers.Adam()           

# オプティマイザーに渡す
optimizer.setup(model)                      

# updaterとTrainerの設定
max_epoch = 30000
updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)        #UpdaterにIteratorとOptimizerを渡す
trainer = training.Trainer(updater, (max_epoch, 'epoch'), out='fxresult')      # TrainerにUpdaterを渡す    

#とりあえず設定
trainer.extend(extensions.LogReport())
trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
trainer.extend(extensions.snapshot_object(model.predictor, filename='model_epoch-{.updater.epoch}'))
trainer.extend(extensions.Evaluator(test_iter, model, device=gpu_id))
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))
trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
trainer.extend(extensions.dump_graph('main/loss'))

#訓練開始
trainer.run()
    
    
    
    
    
    
    
    
    
    
    