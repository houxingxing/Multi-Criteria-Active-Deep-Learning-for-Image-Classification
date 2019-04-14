# -*- coding: utf-8 -*-
"""
Created on Wed Aug 02 16:05:14 2017

@author: Administrator
"""
from keras.datasets import cifar10,mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.regularizers import l1l2, l2, activity_l1l2,activity_l2
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
np.random.seed(1337)  # for reproducibility
import json,random
from keras import backend as K
import h5py,copy,heapq

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





def uncertainty(proba,flag):
    '''
    Input:
    @proba: probability for all samples, n_sample x nb_class
    @flag: a mark for samples, n_sample x 1, 0 represents unselected
    return:
    @BvSB: the uncertainty measure
    @pse_index: the index of sample to be psesudo labeled
    '''
    n = proba.shape[0]
    P_index = np.where(np.array(flag)==0) #1xN array
    plist = P_index[0]
    BvSB  = -1.0*np.ones((n,),dtype='float64')
    for d in plist:
        D = proba[d,:]
        Z = heapq.nlargest(2, D)
        v = np.absolute(np.diff(Z))
        BvSB[d] = 1-v

    return BvSB


def avg_cosine_distanceM(idx,listed):
    '''
    Input:
    @idx: the index of the sample to be measured density
    @listed: the index of the labeled samples
    Return:
    @The average value of diversity
    @m: the index of k neighbor to the sample to be measured density
    '''

    n = len(listed)
    sumD = sum(1-similarity[idx,listed])
    
    #modified 0805
    list_value = list(similarity[idx,listed])
    kvalue = heapq.nlargest(5, list_value)
    m = []
    for i in range(len(kvalue)):
        ids = list_value.index(kvalue[i])
        v = listed[ids]
        m.append(v)
    return 1.0*sumD/n, m      

def getID(pred,listID):
    '''
    Input:
    @pred: the predicted label for the unlabeled samples
    @listID: the index of k neighbor to current sample
    Return:
    @maxID: the psesudo label
    '''
    pclass=[]
    for i in listID:
        pclass.append(pred[i])
    setClass = set(pclass)
    maxID = -1
    maxCount =-1
    for i in setClass:
        if pclass.count(i)>maxCount:
            maxID =i
            maxCount = pclass.count(i)
#    print("knn class:",pclass,"final ",maxID)
    return maxID
        
        
    
def AL(model,proba,data,label,batch_data,batch_label,num,flag):
    '''
    Input:
    @model: the CNN model
    @proba: probability for all samples, n_sample x nb_class
    @data: all the samples
    @label: the label of all samples
    @batch_data: the labeled set
    @batch_label: the label of the labeled set
    @num: the number of selecting samples
    @flag: a mark for samples, n_sample x 1, 0 represents unselected
    @weight:
    Return:
    @xdata: the labeled set and the psesudo labeling set
    @ylabel: the label of the xdata
    '''
    print("the number",num)
    pred = model.predict_classes(data)
    if num==batch_size:
        BvSB  = uncertainty(proba,flag)
        score =[0.0]*len(flag)
        P_index = np.where(np.array(flag)==0)
        N_index = np.where(np.array(flag)==1)
        for i in P_index[0]:
            score[i] = BvSB[i,]
        score_arr = np.array(score).reshape((len(score),))
        index = score_arr.argsort()[-num:][::-1]
    else:
        index = np.where(np.array(flag)==0)[0]
        pse_index =[]
        
    flag_arr = np.array(flag)
    flag_arr[index,]=1
    flag = list(flag_arr)


    tmpdata=data[index,:,:,:]
    tmplabel=label[index,:]
    print(tmpdata.shape)
    batch_data,batch_label=np.concatenate([batch_data,tmpdata],axis=0),np.concatenate([batch_label,tmplabel],axis=0)

    
    return batch_data,batch_label,flag
 
def AL_psesudo(model,proba,data,label,batch_data,batch_label,num,flag,weight):
    '''
    Input:
    @model: the CNN model
    @proba: probability for all samples, n_sample x nb_class
    @data: all the samples
    @label: the label of all samples
    @batch_data: the labeled set
    @batch_label: the label of the labeled set
    @num: the number of selecting samples
    @flag: a mark for samples, n_sample x 1, 0 represents unselected
    @weight:
    Return:
    @xdata: the labeled set and the psesudo labeling set
    @ylabel: the label of the xdata
    '''
    print("the number",num)
    pred = model.predict_classes(data)
    P_index = np.where(np.array(flag)==0)
    N_index = np.where(np.array(flag)==1)
    x = 1.0*batch_data.shape[0]/data.shape[0]
    if num==batch_size:
        BvSB = uncertainty(proba,flag)
        score =[-1.0]*len(flag)
        for i in P_index[0]:
            distance,kId = avg_cosine_distance(i,list(N_index[0]))
            pse_class = getID(pred,kId)
            w_classes = weight[pse_class]
            # score[i] = BvSB[i,]
            score[i] = (1-x)*BvSB[i,]+x*((1-x)*distance+x*w_classes)
        score_arr = np.array(score).reshape((len(score),))
        index = score_arr.argsort()[-num:][::-1]



        #calculate the entropy
        Log_proba = np.log10(proba)
        Entropy_Each_Cell = - np.multiply(proba, Log_proba)
        En = np.sum(Entropy_Each_Cell, axis=1)	# summing across columns of the array

        # threshold = 0.5-0.0625*(batch_data.shape[0]-100)/batch_size
        threshold = 0.3-0.0375*(batch_data.shape[0]-100)/batch_size  #0107
        pse_index =[]
        if threshold>0:
            for i in P_index[0]:
                if i not in index and En[i,]<threshold:
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

    print "noting:#######################################"
    print "batch_data.shape[0]: %d, len(flag==1):%d,pIndex.shape:%d,index.shape:%d,nIndex.shape:%d,pse_index.shape:%d" %(batch_data.shape[0],flag.count(1),len(P_index[0]),len(index),len(N_index[0]),len(pse_index))

    return xdata,ylabel,batch_data,batch_label,flag




def split(data,label):
    '''
    Input:
    @data: all the samples
    @label: the label of data
    Return:
    @dict_data: the split data dict by label
    @dict_label:
    '''
    dict_data={}
    dict_label={}
    label = np.array([list(label[i]).index(1) for i in range(label.shape[0])])
    print(label.shape)
    num = 800
    idx_0 = np.array( np.where(label==0)  ).T
    idx_0 = idx_0[0:num,0]
    X_0 = data[idx_0, :, :, :]
    y_0 = label[idx_0,]
    print(y_0)
    Y_0 = np_utils.to_categorical(y_0,nb_classes)
    dict_data[0] = X_0
    dict_label[0] = Y_0


    idx_1 = np.array( np.where(label==1)  ).T
    idx_1 = idx_1[0:num,0]
    X_1 = data[idx_1, :, :, :]
    y_1 =label[idx_1,]
    Y_1 = np_utils.to_categorical(y_1,nb_classes)
    dict_data[1] = X_1
    dict_label[1] = Y_1

    idx_2 = np.array( np.where(label==2)  ).T
    idx_2 = idx_2[0:num,0]
    X_2 = data[idx_2, :, :, :]
    y_2 =label[idx_2,]
    Y_2 = np_utils.to_categorical(y_2,nb_classes)
    dict_data[2] = X_2
    dict_label[2] = Y_2    


    idx_3 = np.array( np.where(label==3)  ).T
    idx_3 = idx_3[0:num,0]
    X_3 = data[idx_3, :, :, :]
    y_3 =label[idx_3,]
    Y_3 = np_utils.to_categorical(y_3,nb_classes)
    dict_data[3] = X_3
    dict_label[3] = Y_3



    idx_4 = np.array( np.where(label==4)  ).T
    idx_4 = idx_4[0:num,0]
    X_4 = data[idx_4, :, :, :]
    y_4 =label[idx_4,]
    Y_4 = np_utils.to_categorical(y_4,nb_classes)
    dict_data[4] = X_4
    dict_label[4] = Y_4

    idx_5 = np.array( np.where(label==5)  ).T
    idx_5 = idx_5[0:num,0]
    X_5 = data[idx_5, :, :, :]
    y_5 =label[idx_5,]
    Y_5 = np_utils.to_categorical(y_5,nb_classes)
    dict_data[5] = X_5
    dict_label[5] = Y_5


    idx_6 = np.array( np.where(label==6)  ).T
    idx_6 = idx_6[0:num,0]
    X_6 = data[idx_6, :, :, :]
    y_6 =label[idx_6,]
    Y_6 = np_utils.to_categorical(y_6,nb_classes)
    dict_data[6] = X_6
    dict_label[6] = Y_6

    idx_7 = np.array( np.where(label==7)  ).T
    idx_7 = idx_7[0:num,0]
    X_7 = data[idx_7, :, :, :]
    y_7 =label[idx_7,]
    Y_7 = np_utils.to_categorical(y_7,nb_classes)
    dict_data[7] = X_7
    dict_label[7] = Y_7


    idx_8 = np.array( np.where(label==8)  ).T
    idx_8 = idx_8[0:num,0]
    X_8 = data[idx_8, :, :, :]
    y_8 =label[idx_8,]
    Y_8 = np_utils.to_categorical(y_8,nb_classes)
    dict_data[8] = X_8
    dict_label[8] = Y_8

    idx_9 = np.array( np.where(label==9)  ).T
    idx_9 = idx_9[0:num,0]
    X_9 = data[idx_9, :, :, :]
    y_9 =label[idx_9,]
    Y_9 = np_utils.to_categorical(y_9,nb_classes)
    dict_data[9] = X_9
    dict_label[9] = Y_9
    return dict_data,dict_label 

def evaluate(model,dict_data,dict_label):
    '''
    Input:
    @model: the CNN model
    @dict_data:
    @dict_label:
    Return:
    @Acc: the accuracy of all classes
    '''
    Acc = []
    for i in range(10):
        score,acc = model.evaluate(dict_data[i],dict_label[i])
        Acc.append(acc)
    return Acc


def avg_cosine_distance(idx,listed):
    '''
    Input:
    @idx: the index of the sample to be measured density
    @listed: the index of the labeled samples
    Return:
    @The average value of diversity
    @m: the index of k neighbor to the sample to be measured density
    '''

    n = len(listed)
    sumD = sum(1-similarity[idx,listed])

    #modified 0805
    list_value = list(similarity[idx,listed])
    kvalue = heapq.nlargest(5, list_value)
    m = []
    for i in range(len(kvalue)):
        ids = list_value.index(kvalue[i])
        v = listed[ids]
        m.append(v)
    return 1.0*sumD/n, m

def getID(pred,listID):
    '''
    Input:
    @pred: the predicted label for the unlabeled samples
    @listID: the index of k neighbor to current sample
    Return:
    @maxID: the psesudo label
    '''
    pclass=[]
    for i in listID:
        pclass.append(pred[i])
    setClass = set(pclass)
    maxID = -1
    maxCount =-1
    for i in setClass:
        if pclass.count(i)>maxCount:
            maxID =i
            maxCount = pclass.count(i)
#    print("knn class:",pclass,"final ",maxID)
    return maxID

def getWeight(last_acc,now_acc):
    '''
    Input:
    @last_acc: the accuracy of last round
    @now_acc: the accuracy of current round
    Return:
    @q: the weights of all classes
    '''
    p=copy.deepcopy(now_acc)
    q=[]
    s=0
    print("first ....")
    for i in range(10):
        t=p[i]-last_acc[i]
        if t<0:
            t=0
        if last_acc[i]==0.0:
            t/=0.05
        else:
            t/=last_acc[i]
        s+=t
        q.append(t)
    print("improvement:",q)
    if s!=0.0:
        for i in range(10):
            q[i]/=s

    else:
        q=[0.1]*10
    print("last_acc:",last_acc)
    print("acc:",p)
    print("final weight:",q)
    return q

def getWeightM(last_acc,now_acc):
    '''
    Input:
    @last_acc: the accuracy of last round
    @now_acc: the accuracy of current round
    Return:
    @q: the weights of all classes
    '''
    p=copy.deepcopy(now_acc)
    q=[]
    s=0
    print("first ....")
    for i in range(10):
        t=p[i]-last_acc[i]
        if t<0:
            t=0
        s+=t
        q.append(t)
    print("improvement:",q)
    if s!=0.0:
        for i in range(10):
            q[i]/=s

    else:
        q=[0.1]*10
    print("last_acc:",last_acc)
    print("acc:",p)
    print("final weight:",q)
    return q
    
#preparing
def initial(X_train_All,y_train_All):
    '''
    Input:
    @X_train_All: all samples
    @y_train_All:
    Return:
    @X_train,Y_train: as labeled set
    @X_pool,Y_pool: as  unlabeled set
    @X_valid,Y_valid: as validation set
    '''
    X_valid = X_train_All[10000:20000, :, :, :]
    Y_valid = y_train_All[10000:20000]

    X_Pool = X_train_All[20000:22000, :, :, :]
    Y_Pool = y_train_All[20000:22000]


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


def select_random(data,label,batch_data,batch_label,num,flag):
    '''
    Input:
    @data: the candiant samples
    @label: the label of data
    @batch_data: the labeled data
    @batch_label: the label of batch_data
    @num: the number of sampling
    @flag: a mark for all samples
    Return:
    @batch_data: the labeled data
    @batch_label: the label of batch_data
    @flag: a mark for all samples
    '''
    print("num:",num)
    if num==batch_size:
        p_index = np.array(np.where(np.array(flag)==0))[0]
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

#preparing
batch_size = 128
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
# ct = 0
# while ct<5:
#    print('Epoch %d----------------------------------------------------------------------'%ct)
#    batchdata,batchlabel,traindata,trainlabel,X_valid,Y_valid = initial(X_train,y_train)
#    traindata,trainlabel = np.concatenate((batchdata,traindata),axis=0),np.concatenate((batchlabel,trainlabel),axis=0)
#    print(traindata.shape)
#    print("initial dataset:",batchdata.shape)
#    nt = 2000
#    nb_epoch = 5
#
#    num_train=int(nt/batch_size)
#    left=nt%batch_size
#    flag=[1]*100+[0]*(nt)
#    model=build()
#    first=0
#    log2={'val_acc':[],'val_loss':[]}
#    model.fit(batchdata,batchlabel, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_valid, Y_valid))
#    loss,acc = model.evaluate(X_test,Y_test,batch_size = 32,verbose =1)
#    print(loss,acc)
#    log2['val_acc'].append(acc)
#    log2['val_loss'].append(loss)
#
#    for i in range(num_train+1):
#        if i==num_train:
#            batchdata,batchlabel,flag=select_random(traindata,trainlabel,batchdata,batchlabel,left,flag)
#            model.fit(batchdata,batchlabel, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_valid, Y_valid))
#        else:
#            batchdata,batchlabel,flag=select_random(traindata,trainlabel,batchdata,batchlabel,batch_size,flag)
#            model.fit(batchdata,batchlabel, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_valid, Y_valid))
#        loss,acc = model.evaluate(X_test,Y_test,batch_size = 32,verbose =1)
#        print(loss,acc)
#        log2['val_acc'].append(acc)
#        log2['val_loss'].append(loss)
#
#        print(flag.count(0))
#    path = 'mnist_random_bs128_0104_%d.json'%ct
#    with open(path, 'w') as f:
#       json.dump(log2,f)
#    ct+=1
#    del model
#
#
#
similarity = np.load("diver2100.npy")
#
# ct = 0
# while ct<1:
#     print('Epoch %d----------------------------------------------------------------------'%ct)
#     batchdata,batchlabel,traindata,trainlabel,X_valid,Y_valid = initial(X_train,y_train)
#     traindata,trainlabel = np.concatenate((batchdata,traindata),axis=0),np.concatenate((batchlabel,trainlabel),axis=0)
#     print(traindata.shape)
#
#     nt = 2000
#     nb_epoch = 5
#
#     num_train=int(nt/batch_size)
#     left=nt%batch_size
#     flag=[1]*100+[0]*(nt)
#     model_active1=build()
#     first=0
#     log5={'val_acc':[],'val_loss':[]}
#     model_active1.fit(batchdata,batchlabel, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_valid, Y_valid))
#     loss,acc = model_active1.evaluate(X_test,Y_test,batch_size = 32,verbose =1)
#     print(loss,acc)
#     log5['val_acc'].append(acc)
#     log5['val_loss'].append(loss)
#
#
#     x=0
#     while(x<=num_train):
#        if x==num_train:
#            proba=model_active1.predict(traindata)
#            batchdata,batchlabel,flag= AL(model_active1,proba,traindata,trainlabel,batchdata,batchlabel,left,flag)
#            model_active1.fit(batchdata,batchlabel,nb_epoch=nb_epoch,batch_size=batch_size,validation_data=(X_valid,Y_valid))
#        else:
#            proba=model_active1.predict(traindata)
#            batchdata,batchlabel,flag= AL(model_active1,proba,traindata,trainlabel,batchdata,batchlabel,batch_size,flag)
#            model_active1.fit(batchdata,batchlabel,nb_epoch=nb_epoch,batch_size=batch_size,validation_data=(X_valid,Y_valid))
#        loss ,acc = model_active1.evaluate(X_test,Y_test,batch_size = 32,verbose =1)
#        print(loss,acc)
#        log5['val_acc'].append(acc)
#        log5['val_loss'].append(loss)
#        x += 1
#     path='mnist_AL_bs128_BvSB_0106_%d.json'%ct
#     with open(path, 'w') as f:
#        json.dump(log5,f)
#     ct += 1
#     del model_active1


# AL with psesudo labeling
ct = 0
while ct<5:
    print('Epoch %d----------------------------------------------------------------------'%ct)
    batchdata,batchlabel,traindata,trainlabel,X_valid,Y_valid = initial(X_train,y_train)
    traindata,trainlabel = np.concatenate((batchdata,traindata),axis=0),np.concatenate((batchlabel,trainlabel),axis=0)
    print(traindata.shape)
    dict_valdata,dict_label = split(X_valid,Y_valid)
    nt = 2000
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
    last_acc = [0.0]*10
    Acc = evaluate(model_active1,dict_valdata,dict_label)
    weight_classes = getWeightM(last_acc,Acc)
    last_acc = Acc[:]
    x=0
    while(x<=num_train):
       if x==num_train:
           proba=model_active1.predict(traindata)
           tdata,tlabel,batchdata,batchlabel,flag = AL_psesudo(model_active1,proba,traindata,trainlabel,batchdata,batchlabel,left,flag, weight_classes)
           model_active1.fit(tdata,tlabel,nb_epoch=nb_epoch,batch_size=batch_size,validation_data=(X_valid,Y_valid))
       else:
           proba=model_active1.predict(traindata)
           tdata,tlabel,batchdata,batchlabel,flag = AL_psesudo(model_active1,proba,traindata,trainlabel,batchdata,batchlabel,batch_size,flag, weight_classes)
           model_active1.fit(tdata,tlabel,nb_epoch=nb_epoch,batch_size=batch_size,validation_data=(X_valid,Y_valid))
       loss ,acc = model_active1.evaluate(X_test,Y_test,batch_size = 32,verbose =1)
       print(loss,acc)
       log5['val_acc'].append(acc)
       log5['val_loss'].append(loss)
       Acc = evaluate(model_active1,dict_valdata,dict_label)
       weight_classes = getWeightM(last_acc,Acc)
       last_acc = Acc[:]
       x += 1
    path='mnist_AL_bs128_oursM_0109_%d.json'%ct
    with open(path, 'w') as f:
       json.dump(log5,f)
    ct += 1
    del model_active1



 




