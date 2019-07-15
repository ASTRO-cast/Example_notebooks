import numpy as np
import matplotlib.pyplot as plt
import torch
import pyro
import pyro.contrib.gp as gp
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import os
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.double)
import glob
import pandas as pd

#import importlib
#importlib.reload(module)

def load_MODIS(name):
    full_arr=np.load(name)
    T=full_arr[0,:]
    VCI=full_arr[1,:]
    VCI3M=full_arr[2,:]
    
    return T, VCI, VCI3M 

def load(name):
    full_arr = np.load(name)
    X = full_arr[0,:] # days since 1/1/2000
    y = full_arr[2,:] # VCI
    use = (full_arr[4,:] != 0) | (X < 7000)
    X=X[use]
    y=y[use]
    return X, y

def load_BOKU(nf):
    mat = pd.read_csv('../VCI_NDMA.csv')
    mat = np.array(mat)
    county = mat[:,0]
    year = mat[:,1]
    month = mat[:,2]
    VCI3M = mat[:,3]

    all_ct = np.unique(county)
    for i in range(np.size(all_ct)):
        print('Choose '+str(i)+' for: ',all_ct[i])
    num_reg = int(input("Please select a region: "))
    region = all_ct[num_reg]
    print("You have choosen:", region)
    use = (county == region)
    Y = year[use][2:].astype(int)
    M = month[use][2:].astype(int)
    V = VCI3M[use][2:].astype(float)
    
    e_all = np.load(nf)
    e_size = e_all[num_reg,:]
    
    if e_size[0] == 0:
        print("\n The uncertainty is not calibrated, please use the calibration box")
    else:
        print("\n The uncertainty is calibrated, but could be recalibrated in the calibration box")


    return Y,M,V, region, e_size,num_reg


def which_region():
    for i in range(len(glob.glob("im_note/*RBFP_an_trip.npy"))):
        print('Choose '+str(i)+' for: '+glob.glob("im_note/*RBFP_an_trip.npy")[i][8:-17])
    num_reg = int(input("Please select a region: "))
    region = glob.glob("im_note/*RBFP_an_trip.npy")[num_reg]
    print("You have choosen:", region[8:-17])
    return(region)

def plot_vci(X,y):
    plt.figure(figsize=(17, 7))
    plt.plot(X,y, linestyle = 'solid', lw = 3, color = 'blue')
    plt.xlabel('Date', size = 20)
    plt.ylabel('weekly VCI', size = 20)
    plt.tick_params(axis='both', which='major', labelsize=15)
    x_ax = [5114,5479,5844,6210,6575,6940]
    plt.xticks(x_ax, ('1/1/2014','1/1/2015', '1/1/2016', '1/1/2017','1/1/2018','1/1/2019'), size = 18)
    plt.xlim(5000,7200)
    plt.ylim(0,100)

    plt.plot([0,8000],[35,35],color = 'black', lw = 3)
    plt.show()
    
def run_GP(X,y):
    k1 = gp.kernels.RBF(input_dim=2, lengthscale=torch.tensor(50.0),\
                   variance = torch.tensor(0.5))
    #k1.set_constraint("lengthscale", torch.distributions.constraints.interval(7.,1000.))
    smoke_test = ('CI' in os.environ)  # ignore; used to check code integrity in the Pyro repo
    pyro.enable_validation(True)       # can help with debugging
    optim = Adam({"lr": 0.01}) 
    
    #pyro.clear_param_store()

    plus_arr = np.max(X)+np.array([7,14,21,28,35,42,49,56])

    X2 = (torch.from_numpy(X))
    y2 = (torch.from_numpy(y-np.mean(y)))

    Xtest_use = np.append(X,plus_arr)
    Xtest_use2 = (torch.from_numpy(Xtest_use))



    gpr = gp.models.GPRegression(X2, y2,k1, noise=torch.tensor(0.01))

    svi = SVI(gpr.model, gpr.guide, optim, loss=Trace_ELBO())
    losses = []

    num_steps = 25

    for k in range(num_steps):
        losses.append(svi.step())



    with torch.no_grad():
      if type(gpr) == gp.models.VariationalSparseGP:
        mean, cov = gpr(Xtest_use2, full_cov=True)
      else:
        mean, cov = gpr(Xtest_use2, full_cov=False, noiseless=False) 

    sd = cov.sqrt().detach().numpy()
    mean = mean.detach().numpy()+np.mean(y)
    
    return mean, Xtest_use 


def run_GP_MODIS(X,y,tt):
    k1 = gp.kernels.RBF(input_dim=2, lengthscale=torch.tensor(50.0),\
                   variance = torch.tensor(0.5))
    #k1.set_constraint("lengthscale", torch.distributions.constraints.interval(7.,1000.))
    smoke_test = ('CI' in os.environ)  # ignore; used to check code integrity in the Pyro repo
    pyro.enable_validation(True)       # can help with debugging
    optim = Adam({"lr": 0.01}) 
    
    #pyro.clear_param_store()

    plus_arr = np.max(X)+np.array([7,14,21,28,35,42,49,56])

    X2 = (torch.from_numpy(X))
    y2 = (torch.from_numpy(y-np.mean(y)))

    Xtest_use = plus_arr
    Xtest_use2 = (torch.from_numpy(Xtest_use))



    gpr = gp.models.GPRegression(X2, y2,k1, noise=torch.tensor(0.01))

    svi = SVI(gpr.model, gpr.guide, optim, loss=Trace_ELBO())
    losses = []

    num_steps = tt

    for k in range(num_steps):
        losses.append(svi.step())


    with torch.no_grad():
      if type(gpr) == gp.models.VariationalSparseGP:
        mean, cov = gpr(Xtest_use2, full_cov=True)
      else:
        mean, cov = gpr(Xtest_use2, full_cov=False, noiseless=False) 
    sd = cov.sqrt().detach().numpy()
    mean = mean.detach().numpy()+np.mean(y)
    
    return mean, Xtest_use

def which_region_m(nf):
    e_all = np.load(nf)

    for i in range(len(glob.glob("im_note/*MODIS.npy"))):
        print('Choose '+str(i)+' for: '+glob.glob("im_note/*MODIS.npy")[i][8:-13])
    num_reg = int(input("Please select a region: "))
    region = glob.glob("im_note/*MODIS.npy")[num_reg]
    print("You have choosen:", region[8:-13])
    e_size = e_all[num_reg,:]
    
    if e_size[0] == 0:
        print("\n The uncertainty is not calibrated, please use the calibration box")
    else:
        print("\n The uncertainty is calibrated, but could be recalibrated in the calibration box")


    return(region,e_size,num_reg)

def calibrate_monthly(Y,V,nf,num):

    test_arr_M = np.empty((3,np.size(Y)))
    test_arr_M[:] = np.nan
    T_M  = np.arange(np.size(Y))

    for n in range(48,np.size(Y)-3):
        run = T_M <= T_M[n]
        VCI = V[run]
        day = T_M[run]
        if np.isfinite(VCI[-1]) == True: 
            mean, Xtest_use = run_GP_MODIS_M(day,VCI,2)
            test_arr_M[0,n+1] = mean[0]
            test_arr_M[1,n+2] = mean[1]
            test_arr_M[2,n+3] = mean[2]
    
    rms = np.zeros(3)
 
    for i in range(3):
        use = np.isfinite(test_arr_M[i,:])
        rms[i] = np.sqrt(np.mean((V[use]-test_arr_M[i,:][use])**2))
        diff2 = np.mean(np.abs(V[use]-test_arr_M[i,:][use]))

    cal = input("Would you like to save the calibration (yes or no): ")
    if cal == 'yes':
        print('Saving calibration......')
        e_all = np.load(nf)
        e_all[num,:] = rms
        np.save(nf,e_all)
        
    return(rms)            
            
            
def calibrate_weekly(T,VCI3M,nf,num):

    test_arr = np.empty((8,np.size(VCI3M)))
    test_arr[:] = np.nan
    for n in range(200,np.size(VCI3M)-8):
        run = T <= T[n]
        VCI = VCI3M[run]
        day = T[run]
        if np.isfinite(VCI[-1]) == True: 
            use = (np.isfinite(VCI)) 
            mean, Xtest_use = run_GP_MODIS(day[use],VCI[use],2)
            test_arr[0,n+1] = mean[0]
            test_arr[1,n+2] = mean[1]
            test_arr[2,n+3] = mean[2]
            test_arr[3,n+4] = mean[3]
            test_arr[4,n+5] = mean[4]
            test_arr[5,n+6] = mean[5]
            test_arr[6,n+7] = mean[6]
            test_arr[7,n+8] = mean[7]

    rms = np.zeros(8)
    for i in range(8):
        use = np.isfinite(VCI3M) &  np.isfinite(test_arr[i,:])
        rms1 = np.sqrt(np.mean((VCI3M[use]-test_arr[i,:][use])**2))
        diff2 = np.mean(np.abs(VCI3M[use]-test_arr[i,:][use]))
        rms[i] = rms1   
        
    cal = input("Would you like to save the calibration (yes or no): ")
    if cal == 'yes':
        print('Saving calibration......')
        e_all = np.load(nf)
        e_all[num,:] = rms
        np.save(nf,e_all)
        
    return(rms)


def plot_vci_plain(X,y,index):
    plt.figure(figsize=(17, 7))
    plt.plot(X,y, linestyle = 'solid', lw = 3, color = 'blue')
    plt.xlabel('Date', size = 20)
    plt.ylabel(index, size = 20)
    plt.tick_params(axis='both', which='major', labelsize=15)
    x_ax = [5114,5479,5844,6210,6575,6940]
    plt.xticks(x_ax, ('1/1/2014','1/1/2015', '1/1/2016', '1/1/2017','1/1/2018','1/1/2019'), size = 18)
    plt.xlim(5000,7200)
    plt.ylim(0,100)

    plt.plot([0,7200],[35,35],color = 'black', lw = 3)
    plt.show()
    

def plot_vci_m(X,y,index,Xtest_use,mean,rms,region):
    plt.figure(figsize=(17, 7))
    plt.plot(X,y, linestyle = 'solid', lw = 3, color = 'blue')
    plt.errorbar(Xtest_use,mean,rms, color = 'red',label='Forecast',lw = 2)

    plt.xlabel('Date', size = 20)
    plt.ylabel(index, size = 20)
    plt.tick_params(axis='both', which='major', labelsize=15)
    x_ax = [5114,5479,5844,6210,6575,6940]
    plt.xticks(x_ax, ('1/1/2014','1/1/2015', '1/1/2016', '1/1/2017','1/1/2018','1/1/2019'), size = 18)
    plt.xlim(5000,7200)
    plt.ylim(0,100)

    plt.plot([0,7200],[35,35],color = 'black', lw = 3)
    plt.plot([0,8000],[35,35],color = 'black', lw = 3)
    plt.plot([np.max(X),np.max(X)],[0,100],linestyle = '--',color = 'black', lw = 3,label = 'day of last observation')
    plt.legend(prop={'size': 20}, loc = 3)
    
    
    plt.show()


    for i in range(np.size(rms)):
        num = mean[i]
        print('week(s) = ',int(i+1),',   VCI3M = ',"%.0f" % num, ',   region = '+region[8:-13])
    
    if mean[3] > y[-1]:
        print('Trend next month:'+'UP')
    else:
        print('Trend next month:'+'DOWN')
    if mean[3] < 35: 
        print('VCI3M < 35 next month')
        
        
        

def run_GP_MODIS_M(X,y,tt):
    k1 = gp.kernels.RBF(input_dim=2, lengthscale=torch.tensor(2.0),\
                   variance = torch.tensor(0.5))
    #k1.set_constraint("lengthscale", torch.distributions.constraints.interval(7.,1000.))
    smoke_test = ('CI' in os.environ)  # ignore; used to check code integrity in the Pyro repo
    pyro.enable_validation(True)       # can help with debugging
    optim = Adam({"lr": 0.01}) 
    
    #pyro.clear_param_store()

    plus_arr = np.max(X)+np.array([1.0,2.0,3.0])

    X2 = (torch.from_numpy(X.astype(float)))
    y2 = (torch.from_numpy(y-np.mean(y)))

    Xtest_use = plus_arr.astype(float)
    Xtest_use2 = (torch.from_numpy(Xtest_use))



    gpr = gp.models.GPRegression(X2, y2,k1, noise=torch.tensor(0.01))

    svi = SVI(gpr.model, gpr.guide, optim, loss=Trace_ELBO())
    losses = []


    for k in range(tt):
        losses.append(svi.step())



    with torch.no_grad():
      if type(gpr) == gp.models.VariationalSparseGP:
        mean, cov = gpr(Xtest_use2, full_cov=True)
      else:
        mean, cov = gpr(Xtest_use2, full_cov=False, noiseless=False) 

    sd = cov.sqrt().detach().numpy()
    mean = mean.detach().numpy()+np.mean(y)
    
    return mean, Xtest_use


    
def plot_vci_fc(Xtest_use,mean,X,y):

    plt.figure(figsize=(17, 7))
    use = Xtest_use >= np.max(X)
    err = np.std(y)
    eb = err*np.array([0,0.5*0.46,0.46,0.46/2+0.66/2,0.66,0.66/2+0.81/2,0.81,0.91,1.0])
    #plt.fill_between(Xtest_use[use],mean[use]-eb,mean[use]+eb, \
    #        color = 'red', label = 'GP prediction')
    plt.errorbar(Xtest_use[use], mean[use], yerr=eb,color='red',lw=3,label='Forecast')

    
    plt.plot(X,y, linestyle = 'solid', lw = 3, color = 'blue',label = 'Landsat VCI')
    plt.xlabel('Date', size = 20)
    plt.ylabel('weekly VCI', size = 20)
    plt.tick_params(axis='both', which='major', labelsize=15)
    x_ax = [6575,6665,6756,6848,6940,7030,7121]
    plt.xticks(x_ax, ('1/1/2018','1/4/2018','1/7/2018','1/10/2018','1/1/2019','1/4/2019','1/7/2019'), size = 18)
    plt.xlim(6575,7150)
    plt.ylim(0,100)

    plt.plot([0,8000],[35,35],color = 'black', lw = 3)
    plt.plot([np.max(X),np.max(X)],[0,100],linestyle = '--',color = 'black', lw = 3,\
            label = 'day of last observation')
    plt.legend(prop={'size': 20}, loc = 3)
    plt.show()
    fc = Xtest_use - np.max(X)
    use = fc > 0
    fc = fc[use]
    for i in range(np.size(fc)):
        num = mean[use][i]
        print('weeks = ',int(fc[i]/7),',   VCI = ',"%.0f" % num)

        
def plot_vci_fc3M(Xtest_use,mean,X,y):
    yf_new = np.zeros(np.size(mean))
    y_new = np.zeros(np.size(y))
    for i in range(12,np.size(mean)):
        yf_new[i] = np.mean(mean[i-12:i])
    for i in range(12,np.size(y)):
        y_new[i] = np.mean(y[i-12:i]) 
        
    plt.figure(figsize=(17, 7))
    use = Xtest_use >= np.max(X)
    err = np.std(y_new)
    eb = err*np.array([0,0.5*0.08,0.08,0.08/2+0.2/2,0.2,0.2/2+0.35/2,0.35,0.42,0.49])
    #plt.fill_between(Xtest_use[use],yf_new[use]-eb,yf_new[use]+eb, \
    #        color = 'red', label = 'GP prediction')
    plt.errorbar(Xtest_use[use], yf_new[use], yerr=eb,color='red',lw=3,label='Forecast')

    #plt.plot(Xtest_use,mean+3*sd, linestyle = '--', lw = 3, color = 'red')
    #plt.plot(Xtest_use,mean-3*sd, linestyle = '--', lw = 3, color = 'red')


    plt.plot(X,y_new, linestyle = 'solid', lw = 3, color = 'blue',label = 'Landsat VCI3M')

    plt.xlabel('Date', size = 20)
    plt.ylabel('VCI3M', size = 20)
    plt.tick_params(axis='both', which='major', labelsize=15)
    x_ax = [6575,6665,6756,6848,6940,7030,7121]
    plt.xticks(x_ax, ('1/1/2018','1/4/2018','1/7/2018','1/10/2018','1/1/2019','1/4/2019','1/7/2019'), size = 18)
    plt.xlim(6575,7150)
    plt.ylim(0,100)

    plt.plot([0,8000],[35,35],color = 'black', lw = 3)
    plt.plot([np.max(X),np.max(X)],[0,100],linestyle = '--',color = 'black', lw = 3,\
            label = 'day of last observation')
    
    
    plt.legend(prop={'size': 20}, loc = 3)

    plt.show()

    fc = Xtest_use - np.max(X)
    use = fc > 0
    fc = fc[use]

    for i in range(np.size(fc)):
        num = yf_new[use][i]
        print('weeks  = ',int(fc[i]/7),',   VCI3M = ',"%.0f" % num)       
        
        
def plot_fc(mean, Xtest_use, V, M ,Y,region,e_bar):
    X = np.arange(np.size(M))

    plt.figure(figsize=(17, 7))
    plt.plot([np.max(X),np.max(X)],[0,100],color = 'black', lw = 3)
    plt.plot([0,800],[35,35],color = 'black', lw = 3)

    plt.plot(X,V, linestyle = 'solid', lw = 3, color = 'blue')
    plt.errorbar(np.append(X[-1],Xtest_use),np.append(V[-1],mean),np.append(0,e_bar), lw = 3, color = 'red')

    x_ax = np.array([10+7*24,6+10+7*24,12+10+7*24,18+10+7*24,10+8*24,6+10+8*24])
    plt.xticks(x_ax, ('Jan 2016','July 2016','Jan 2017','July 2017','Jan 2018','July 2018'), size = 18)   
                      
    plt.ylabel('VCI3M', size = 20)
    
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.xlim(175,210)
    plt.ylim(0,100)
    plt.show()
    
    print('VCI3M = ',"%.0f" % mean[0],' for Month:', M[-1]+1,' Year:',Y[-1], 'region: '+region)
    if mean[0] > V[-1]:
        print('Trend:'+'UP')
    else:
        print('Trend:'+'DOWN')
    if mean[0] < 35: 
        print('VCI3M < 35 next month')
        
def plot_M(Y,M,V):
    plt.figure(figsize=(17, 7))
    X = np.arange(np.size(M))
    plt.plot(X,V, linestyle = 'solid', lw = 3, color = 'blue')
    plt.ylabel('VCI3M', size = 20)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.xlim(0,210)
    plt.ylim(0,100)
    plt.plot([0,800],[35,35],color = 'black', lw = 3)
    
    x_ax = np.array([10,10+24,10+48,10+3*24,10+4*24,10+5*24,10+6*24,10+7*24,10+8*24])
    plt.xticks(x_ax, ('Jan 2002','Jan 2004','Jan 2006','Jan 2008','Jan 2010','Jan 2012','Jan 2014',\
                      'Jan 2016','Jan 2018'), size = 18)
    
    plt.show()