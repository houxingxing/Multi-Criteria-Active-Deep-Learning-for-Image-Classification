# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 20:21:02 2017

@author: Administrator
"""
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D,ZeroPadding2D
from keras.regularizers import l1l2, l2, activity_l1l2,activity_l2
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
#np.random.seed(1337)  # for reproducibility
import json,random
from keras import backend as K
import h5py,heapq

def build():
    model = Sequential()
    model.add(Convolution2D(32,3, 3,activation='relu',input_shape=(1,28,28)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
#    sgd = SGD(lr=0.09, decay=1e-6, momentum=0.9, nesterov=True)
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def AL_EN(model,proba,data,label,batch_data,batch_label,num,flag):
    print("the number",num)
    pred = model.predict_classes(data)
    if num==batch_size:
        Log_proba = np.log10(proba)
        Entropy_Each_Cell = - np.multiply(proba, Log_proba)
        entropy = np.sum(Entropy_Each_Cell, axis=1)	# summing across columns of the array
        N_index = np.where(np.array(flag)==1)
        print(min(entropy),max(entropy))
        entropy[N_index,]= -1
        index = entropy.argsort()[-num:][::-1]
        print("num",num,entropy[0:5])
        
        threshold = 0.05-0.005*(batch_data.shape[0]-100)/batch_size
        print(threshold)
        pse_index =[]
        if threshold>0:
            for i in range(entropy.shape[0]):
                if i not in index and entropy[i,]>0 and entropy[i,]<threshold:
                    pse_index.append(i)
            n=len(pse_index)
            n1 = int(0.1*n)
            print("pse...........: ",pse_index[0:n1],n)
    else:
        index = np.where(np.array(flag)==0)[0]
        pse_index =[]
        
    flag_arr = np.array(flag)
    flag_arr[index,]=1
    flag = list(flag_arr)
        
#    print(len(index))
#    print(index[:5])
    print(index)
    tmpdata=data[index,:,:,:]
    tmplabel=label[index,:]
    print(tmpdata.shape)
    batch_data,batch_label=np.concatenate([batch_data,tmpdata],axis=0),np.concatenate([batch_label,tmplabel],axis=0)
    
    if len(pse_index)!=0:
        tmpdata=data[pse_index,:,:,:]
        tmplabel=pred[pse_index,]
        tmplabel = np_utils.to_categorical(tmplabel,nb_classes)
        xdata,ylabel= np.concatenate([batch_data,tmpdata],axis=0),np.concatenate([batch_label,tmplabel],axis=0)
    else:
        xdata,ylabel = batch_data,batch_label
    
    return xdata,ylabel,batch_data,batch_label,flag

def AL_BvSB(model,proba,data,label,batch_data,batch_label,num,flag):
    print("the number",num)
    pred = model.predict_classes(data)
    if num==batch_size:
        BvSB = []
        for d in range(proba.shape[0]):
            D = proba[d,:]
            Z = heapq.nlargest(2, D)
            v = np.absolute(np.diff(Z))
            BvSB.append(1-v)
        n=len(BvSB)
        BvSB = np.array(BvSB).reshape((n,))
        N_index = np.where(np.array(flag)==1)
        print(min(BvSB),max(BvSB))
        BvSB[N_index,]= -1
        index = BvSB.argsort()[-num:][::-1]    
        print(BvSB)
        threshold = 0.5-0.0625*(batch_data.shape[0]-100)/batch_size
        pse_index =[]
        if threshold>0:
            for i in range(BvSB.shape[0]):
                if i not in index and BvSB[i,]>0 and BvSB[i,]<threshold:
                    pse_index.append(i)
            n=len(pse_index)
            n1 = int(0.1*n)
            print("pse...........: ",pse_index[0:n1],n)
    else:
        index = np.where(np.array(flag)==0)[0]
        pse_index =[]
        
    flag_arr = np.array(flag)
    flag_arr[index,]=1
    flag = list(flag_arr)
        
#    print(len(index))
#    print(index[:5])
    tmpdata=data[index,:,:,:]
    tmplabel=label[index,:]
    print(tmpdata.shape)
    batch_data,batch_label=np.concatenate([batch_data,tmpdata],axis=0),np.concatenate([batch_label,tmplabel],axis=0)
    
    if len(pse_index)!=0:
        tmpdata=data[pse_index,:,:,:]
        tmplabel=pred[pse_index,]
        tmplabel = np_utils.to_categorical(tmplabel,nb_classes)
        xdata,ylabel= np.concatenate([batch_data,tmpdata],axis=0),np.concatenate([batch_label,tmplabel],axis=0)
    else:
        xdata,ylabel = batch_data,batch_label
    
    return xdata,ylabel,batch_data,batch_label,flag

def AL_LC(model,proba,data,label,batch_data,batch_label,num,flag):
    print("the number",num)
    pred = model.predict_classes(data)
    if num==batch_size:
        lc = []
        for i in range(proba.shape[0]):
            D = proba[i,:]
            maxVal = np.max(D)
            lc.append(1-maxVal)
        n = len(lc)
        lc = np.array(lc).reshape((n,))
        N_index = np.where(np.array(flag)==1)
        lc[N_index,]=-1
        index = lc.argsort()[-num:][::-1]
        print(lc)
        threshold = 0.5-0.0625*(batch_data.shape[0]-100)/batch_size
        pse_index =[]
        if threshold>0:
            for i in range(lc.shape[0]):
                if i not in index and lc[i,]>0 and lc[i]<threshold:
                    pse_index.append(i)
            n=len(pse_index)
            n1 = int(0.1*n)
            print("pse...........: ",pse_index[0:n1])
    else:
        index = np.where(np.array(flag)==0)[0]
        pse_index =[]
        
    flag_arr = np.array(flag)
    flag_arr[index,]=1
    flag = list(flag_arr)
        
#    print(len(index))
#    print(index[:5])
    tmpdata=data[index,:,:,:]
    tmplabel=label[index,:]
    print(tmpdata.shape)
    batch_data,batch_label=np.concatenate([batch_data,tmpdata],axis=0),np.concatenate([batch_label,tmplabel],axis=0)
    
    if len(pse_index)!=0:
        tmpdata=data[pse_index,:,:,:]
        tmplabel=pred[pse_index,]
        tmplabel = np_utils.to_categorical(tmplabel,nb_classes)
        xdata,ylabel= np.concatenate([batch_data,tmpdata],axis=0),np.concatenate([batch_label,tmplabel],axis=0)
    else:
        xdata,ylabel = batch_data,batch_label
    
    return xdata,ylabel,batch_data,batch_label,flag    
    
def select_random(data,label,batch_data,batch_label,num,flag):
    print("num:",num)

    if num==batch_size:
        index=[]
        p_index = np.array(np.where(np.array(flag)==0))[0]
#        p_index = p_index.tolist()
       
        print(p_index , num)
        index = random.sample(list(p_index), num)
           
    else:
        index = np.where(np.array(flag)==0)[0]

    
    flag_arr = np.array(flag)
    flag_arr[index,]=1
    flag = list(flag_arr)
    
    print(len(index))
    tmpdata = data[index,:,:,:]
    tmplabel = label[index,:]
    
    batch_data,batch_label=np.concatenate([batch_data,tmpdata],axis=0),np.concatenate([batch_label,tmplabel],axis=0)
    return batch_data,batch_label,flag
    
def initial(X_train_All,y_train_All):
    X_valid = X_train_All[10000:15000, :, :, :]
    Y_valid = y_train_All[10000:15000]

    X_Pool = X_train_All[20000:21000, :, :, :]
    Y_Pool = y_train_All[20000:21000]


    X_train_All = X_train_All[0:10000, :, :, :]
    y_train_All = y_train_All[0:10000]

    num = 10
    idx_0 = np.array( np.where(y_train_All==0)  ).T
    idx_0 = idx_0[0:num,0]
    X_0 = X_train_All[idx_0, :, :, :]
    y_0 = y_train_All[idx_0,]

    idx_1 = np.array( np.where(y_train_All==1)  ).T
    idx_1 = idx_1[0:num,0]
    X_1 = X_train_All[idx_1, :, :, :]
    y_1 = y_train_All[idx_1,]

    idx_2 = np.array( np.where(y_train_All==2)  ).T
    idx_2 = idx_2[0:num,0]
    X_2 = X_train_All[idx_2, :, :, :]
    y_2 = y_train_All[idx_2,]

    idx_3 = np.array( np.where(y_train_All==3)  ).T
    idx_3 = idx_3[0:num,0]
    X_3 = X_train_All[idx_3, :, :, :]
    y_3 = y_train_All[idx_3,]

    idx_4 = np.array( np.where(y_train_All==4)  ).T
    idx_4 = idx_4[0:num,0]
    X_4 = X_train_All[idx_4, :, :, :]
    y_4 = y_train_All[idx_4,]

    idx_5 = np.array( np.where(y_train_All==5)  ).T
    idx_5 = idx_5[0:num,0]
    X_5 = X_train_All[idx_5, :, :, :]
    y_5 = y_train_All[idx_5,]

    idx_6 = np.array( np.where(y_train_All==6)  ).T
    idx_6 = idx_6[0:num,0]
    X_6 = X_train_All[idx_6, :, :, :]
    y_6 = y_train_All[idx_6,]

    idx_7 = np.array( np.where(y_train_All==7)  ).T
    idx_7 = idx_7[0:num,0]
    X_7 = X_train_All[idx_7, :, :, :]
    y_7 = y_train_All[idx_7,]

    idx_8 = np.array( np.where(y_train_All==8)  ).T
    idx_8 = idx_8[0:num,0]
    X_8 = X_train_All[idx_8, :, :, :]
    y_8 = y_train_All[idx_8,]

    idx_9 = np.array( np.where(y_train_All==9)  ).T
    idx_9 = idx_9[0:num,0]
    X_9 = X_train_All[idx_9, :, :, :]
    y_9 = y_train_All[idx_9,]

    X_train = np.concatenate((X_0, X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8, X_9), axis=0 )
    Y_train = np.concatenate((y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9), axis=0 )
    Y_train = np_utils.to_categorical(Y_train,nb_classes)
    Y_Pool = np_utils.to_categorical(Y_Pool,nb_classes)
    Y_valid = np_utils.to_categorical(Y_valid,nb_classes)

    return X_train,Y_train,X_Pool,Y_Pool,X_valid,Y_valid
        
#preparing
batch_size = 64
nb_classes = 10
nb_epoch = 1

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# convert class vectors to binary class matrices
#Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


## random sampling
#
ct = 0
while ct<5:
    print('Epoch %d----------------------------------------------------------------------'%ct)
    batchdata,batchlabel,traindata,trainlabel,X_valid,Y_valid = initial(X_train,y_train)
    traindata,trainlabel = np.concatenate((batchdata,traindata),axis=0),np.concatenate((batchlabel,trainlabel),axis=0)
    print(traindata.shape)
    print("initial dataset:",batchdata.shape)
    nt = 1000
    nb_epoch = 5
    
    num_train=int(nt/batch_size)
    left=nt%batch_size
    flag=[1]*100+[0]*(nt)
    model=build()
    first=0
    log2={'val_acc':[],'val_loss':[]}
    model.fit(batchdata,batchlabel, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_valid, Y_valid))
    loss,acc = model.evaluate(X_test,Y_test,batch_size = 32,verbose =1)
    print(loss,acc)
    log2['val_acc'].append(acc)
    log2['val_loss'].append(loss)

    for i in range(num_train+1):    
        if i==num_train:
            batchdata,batchlabel,flag=select_random(traindata,trainlabel,batchdata,batchlabel,left,flag)
            model.fit(batchdata,batchlabel, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_valid, Y_valid))
        else:
            batchdata,batchlabel,flag=select_random(traindata,trainlabel,batchdata,batchlabel,batch_size,flag)
            model.fit(batchdata,batchlabel, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_valid, Y_valid))
        loss,acc = model.evaluate(X_test,Y_test,batch_size = 32,verbose =1)
        print(loss,acc)
        log2['val_acc'].append(acc)
        log2['val_loss'].append(loss)

        print(flag.count(0))
    path = 'mnist_random_bs64_0806_%d.json'%ct
    with open(path, 'w') as f:
       json.dump(log2,f)
    ct+=1
    del model



       
ct = 0
while ct<1:
    print('Epoch %d----------------------------------------------------------------------'%ct)
    batchdata,batchlabel,traindata,trainlabel,X_valid,Y_valid = initial(X_train,y_train)
    traindata,trainlabel = np.concatenate((batchdata,traindata),axis=0),np.concatenate((batchlabel,trainlabel),axis=0)
    print(traindata.shape)
    
    nt = 1000
    nb_epoch = 5
    
    num_train=int(nt/batch_size)
    left=nt%batch_size
    flag=[1]*100+[0]*(nt)
    model_active1=build()
    first=0
    log5={'val_acc':[],'val_loss':[]}
    model_active1.fit(batchdata,batchlabel, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_valid, Y_valid))
    loss,acc = model_active1.evaluate(X_test,Y_test,batch_size = 32,verbose =1)
    print(loss,acc)
    log5['val_acc'].append(acc)
    log5['val_loss'].append(loss)


    x=0
    while(x<=num_train):
       if x==num_train:
           proba=model_active1.predict(traindata)
           tdata,tlabel,batchdata,batchlabel,flag = AL_EN(model_active1,proba,traindata,trainlabel,batchdata,batchlabel,left,flag)
           model_active1.fit(tdata,tlabel,nb_epoch=nb_epoch,batch_size=batch_size,validation_data=(X_valid,Y_valid))
       else:
           proba=model_active1.predict(traindata)
           tdata,tlabel,batchdata,batchlabel,flag = AL_EN(model_active1,proba,traindata,trainlabel,batchdata,batchlabel,batch_size,flag)
           model_active1.fit(tdata,tlabel,nb_epoch=nb_epoch,batch_size=batch_size,validation_data=(X_valid,Y_valid))
       loss ,acc = model_active1.evaluate(X_test,Y_test,batch_size = 32,verbose =1)
       print(loss,acc)
       log5['val_acc'].append(acc)
       log5['val_loss'].append(loss)
       x += 1
    path='mnist_Entropy_bs128_CEAL_0806_%d.json'%ct
    with open(path, 'w') as f:
       json.dump(log5,f)
    ct += 1
    del model_active1

ct = 0
while ct<5:
    print('Epoch %d----------------------------------------------------------------------'%ct)
    batchdata,batchlabel,traindata,trainlabel,X_valid,Y_valid = initial(X_train,y_train)
    traindata,trainlabel = np.concatenate((batchdata,traindata),axis=0),np.concatenate((batchlabel,trainlabel),axis=0)
    print(traindata.shape)
    nt = 1000
    nb_epoch = 5
    
    num_train=int(nt/batch_size)
    left=nt%batch_size
    flag=[1]*100+[0]*(nt)
    model_active2=build()
    first=0
    log6={'val_acc':[],'val_loss':[]}
    model_active2.fit(batchdata,batchlabel, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_valid, Y_valid))
    loss,acc = model_active2.evaluate(X_test,Y_test,batch_size = 32,verbose =1)
    print(loss,acc)
    log6['val_acc'].append(acc)
    log6['val_loss'].append(loss)


    x=0
    while(x<=num_train):
       if x==num_train:
           proba=model_active2.predict(traindata)
           tdata,tlabel,batchdata,batchlabel,flag = AL_BvSB(model_active2,proba,traindata,trainlabel,batchdata,batchlabel,left,flag)
           model_active2.fit(tdata,tlabel,nb_epoch=nb_epoch,batch_size=batch_size,validation_data=(X_valid,Y_valid))
       else:
           proba=model_active2.predict(traindata)
           tdata,tlabel,batchdata,batchlabel,flag = AL_BvSB(model_active2,proba,traindata,trainlabel,batchdata,batchlabel,batch_size,flag)
           model_active2.fit(tdata,tlabel,nb_epoch=nb_epoch,batch_size=batch_size,validation_data=(X_valid,Y_valid))
       loss ,acc = model_active2.evaluate(X_test,Y_test,batch_size = 32,verbose =1)
       print(loss,acc)
       log6['val_acc'].append(acc)
       log6['val_loss'].append(loss)
       x += 1
    path='mnist_BvSB_bs128_CEAL_0806_%d.json'%ct
    with open(path, 'w') as f:
       json.dump(log6,f)
    ct += 1
    del model_active2


ct = 0
while ct<5:
    print('Epoch %d----------------------------------------------------------------------'%ct)
    batchdata,batchlabel,traindata,trainlabel,X_valid,Y_valid = initial(X_train,y_train)
    traindata,trainlabel = np.concatenate((batchdata,traindata),axis=0),np.concatenate((batchlabel,trainlabel),axis=0)
    print(traindata.shape)
    nt = 1000
    nb_epoch = 5
    
    num_train=int(nt/batch_size)
    left=nt%batch_size
    flag=[1]*100+[0]*(nt)
    model_active3=build()
    first=0
    log7={'val_acc':[],'val_loss':[]}
    model_active3.fit(batchdata,batchlabel, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_valid, Y_valid))
    loss,acc = model_active3.evaluate(X_test,Y_test,batch_size = 32,verbose =1)
    print(loss,acc)
    log7['val_acc'].append(acc)
    log7['val_loss'].append(loss)


    x=0
    while(x<=num_train):
       if x==num_train:
           proba=model_active3.predict(traindata)
           tdata,tlabel,batchdata,batchlabel,flag = AL_LC(model_active3,proba,traindata,trainlabel,batchdata,batchlabel,left,flag)
           model_active3.fit(tdata,tlabel,nb_epoch=nb_epoch,batch_size=batch_size,validation_data=(X_valid,Y_valid))
       else:
           proba=model_active3.predict(traindata)
           tdata,tlabel,batchdata,batchlabel,flag = AL_LC(model_active3,proba,traindata,trainlabel,batchdata,batchlabel,batch_size,flag)
           model_active3.fit(tdata,tlabel,nb_epoch=nb_epoch,batch_size=batch_size,validation_data=(X_valid,Y_valid))
       loss ,acc = model_active3.evaluate(X_test,Y_test,batch_size = 32,verbose =1)
       print(loss,acc)
       log7['val_acc'].append(acc)
       log7['val_loss'].append(loss)
       x += 1
    path='mnist_LC_bs128_CEAL_0806_%d.json'%ct
    with open(path, 'w') as f:
       json.dump(log7,f)
    ct += 1
    del model_active3