#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 16:10:57 2019

@author: abb22
"""

import numpy as np
from matplotlib import pyplot as plt
import glob

def load(name):
    full_arr=np.load(name)
    T=full_arr[0,:]
    VCI=full_arr[1,:]
    VCI3M=full_arr[2,:]
    
    return T, VCI, VCI3M  

def which_region():
    for i in range(len(glob.glob("im_note/*MODIS.npy"))):
        print('Choose '+str(i)+' for: '+glob.glob("im_note/*MODIS.npy")[i][8:-13])
    num_reg = int(input("Please select a region: "))
    region = glob.glob("im_note/*MODIS.npy")[num_reg]
    print("You have choosen:", region[8:-13])
    return(region)


def plot_vci(X,y,index):
    plt.figure(figsize=(17, 7))
    plt.plot(X,y, linestyle = 'solid', lw = 3, color = 'blue')
    #plt.errorbar(Xtest_use,mean,rms, color = 'red')
    plt.xlabel('Date', size = 20)
    plt.ylabel(index, size = 20)
    plt.tick_params(axis='both', which='major', labelsize=15)
    x_ax = [5114,5479,5844,6210,6575,6940]
    plt.xticks(x_ax, ('1/1/2014','1/1/2015', '1/1/2016', '1/1/2017','1/1/2018','1/1/2019'), size = 18)
    plt.xlim(5000,7200)
    plt.ylim(0,100)

    plt.plot([0,7200],[35,35],color = 'black', lw = 3)
    plt.show()
    
def plot_vci_fc(X,y,Forecast,Sigma,index):
    
    n=len(X)
    nw=len(Forecast)
    x1=np.arange(X[n-1],X[n-1]+7*nw,7)

#    f=np.zeros(4)
#    f[0]=Forecast
#    f[1]=Forecast[1]
#    f[2]=Forecast[3]
#    f[3]=Forecast[5]
#    
#    s=np.zeros(4)
#    s[0]=0
#    s[1]=Sigma[1]
#    s[2]=Sigma[3]
#    s[3]=Sigma[5]

    plt.figure(figsize=(17, 7))
    plt.plot(X,y, linestyle = 'solid', lw = 3, color = 'blue',label = 'data')
    plt.errorbar(x1, Forecast, yerr=Sigma,color='red',lw=3,label='Forecast')
    #plt.fill_between(x1,Forecast-Sigma,Forecast+Sigma, \
    #        color = 'red', label = 'Forecast')
    plt.xlabel('Date', size = 20)
    plt.ylabel(index, size = 20)
    plt.tick_params(axis='both', which='major', labelsize=15)
    x_ax = [6575,6665,6756,6848,6940,7030,7121]
    plt.xticks(x_ax, ('1/1/2018','1/4/2018','1/7/2018','1/10/2018','1/1/2019','1/4/2019','1/7/2019'), size = 18)
    plt.xlim(6575,7200)
    plt.ylim(0,100)

    plt.plot([0,7200],[35,35],color = 'black', lw = 3)
    plt.plot([np.max(X),np.max(X)],[0,100],linestyle = '--',color = 'black', lw = 3,\
            label = 'day of last observation')
    plt.legend(prop={'size': 20},loc=1) 
    plt.show()
    
    print('Forecast:')
    print('VCI3M 4 weeks after last observation =',"%.0f" % Forecast[3])
    if y[n-1]<Forecast[3]:
        print('Trend = Upward')
    if y[n-1]>Forecast[3]:
        print('Trend = Downward')
    
def astro_regress_one(Y,X,nlags):
    
    nobs=len(Y)
    
    Xsegs=[]
    Ysegs=[]
    segstart=0
    nsegs=0
    
    for t in range(nobs-1):
        if not np.isnan(X[t]) and not np.isnan(Y[t]):
            if np.isnan(X[t+1]) or np.isnan(Y[t+1]):
                if t+1-segstart>nlags:
                    Xsegs.append(X[segstart:t+1])
                    Ysegs.append(Y[segstart:t+1])
                    nsegs=nsegs+1
        if np.isnan(X[t]) or np.isnan(Y[t]):
            if not np.isnan(X[t+1]) and not np.isnan(Y[t+1]):
                segstart=t+1
                             
    if not np.isnan(X[nobs-1]) and not np.isnan(Y[nobs-1]):
        if nobs-segstart>nlags:
            Xsegs.append(X[segstart:nobs])
            Ysegs.append(Y[segstart:nobs])
            nsegs=nsegs+1
        
    nobs=0
    for i in range(nsegs):
        nobs=nobs+len(Xsegs[i])
                             
    regressors = np.zeros((nobs-nsegs*nlags,nlags))
    ydep=np.zeros(nobs-nsegs*nlags)
    
    segstart=0
                             
    for i in range(nsegs):  
        XX=Xsegs[i]
        YY=Ysegs[i]
        nobsseg=len(XX)
        ydep[segstart:segstart+nobsseg-nlags] = YY[nlags:]
        for tau in range(nlags): 
            regressors[segstart:segstart+nobsseg-nlags,tau] = XX[nlags-tau-1:nobsseg-tau-1]
        segstart=segstart+nobsseg-nlags
       
    beta=np.zeros(nlags)
    ypred=np.zeros(nobs-nsegs*nlags)
    u=np.zeros(nobs-nsegs*nlags)                          
                             
    regrees = np.linalg.lstsq(regressors,ydep)
    beta=regrees[0]
    ypred = np.dot(regressors,beta)  # keep hold of predicted values
    u = ydep-ypred
    res=np.cov(u)
    
    return beta, u, res, ypred

def astro_predict_one(Y,X,nlags,trainlength):

    nobs=len(Y)
    ntests=nobs-trainlength
    
    ypred=np.zeros(ntests)
    u=np.zeros(ntests)
    
    nopredict=0
    
    for k in range(ntests):
        ret=astro_regress_one(Y[k:k+trainlength],X[k:k+trainlength],nlags)
        beta=ret[0]
        
        predictors = np.zeros(nlags)
        for tau in range(nlags): 
            predictors[tau] = X[k+trainlength-tau-1]
                
        ypred[k]=np.dot(predictors,beta)
        u[k]=Y[k+trainlength]-ypred[k]
        if np.isnan(u[k]):
            nopredict=nopredict+1
        
    respredict=np.sqrt(np.nanvar(u))
       
    k=ntests
    ret=astro_regress_one(Y[k:k+trainlength],X[k:k+trainlength],nlags)
    beta=ret[0]
        
    predictors = np.zeros(nlags)
    for tau in range(nlags): 
        predictors[tau] = X[k+trainlength-tau-1]
                
    forecast=np.dot(predictors,beta)     
    
    return respredict, ypred, nopredict, forecast

def forecast(VCI):
    VCImean=np.nanmean(VCI)
    VCIz=VCI-VCImean
    nlags0=3
    trainlength=200
    l=len(VCI)
    
    VCIpred=np.zeros((9,l))
    Forecast=np.zeros(9)
    Sigma=np.zeros(9)
    Forecast[0]=VCI[l-1]
    
    for i in range(0,8):
    
        Y=VCIz[i:]
        X=VCIz[0:l-i]
        ret=astro_predict_one(Y,X,nlags0,trainlength)
        ypred=ret[1]
        VCIpred[i,trainlength+i:]=ypred
        VCIpred[i,:]=VCIpred[i,:]+VCImean
        Forecast[i+1]=ret[3]+VCImean
        Sigma[i+1]=ret[0]
    
    return Forecast, Sigma

