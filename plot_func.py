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

def load(name):
    full_arr = np.load(name)
    X = full_arr[0,:] # days since 1/1/2000
    y = full_arr[2,:] # VCI
    use = X < 6950
    X=X[use]
    y=y[use]
    return X, y

def plot_vci(X,y):
    plt.figure(figsize=(17, 7))
    plt.plot(X,y, linestyle = 'solid', lw = 3, color = 'blue')
    plt.xlabel('Days since 1/1/2000', size = 20)
    plt.ylabel('weekly VCI', size = 20)
    plt.tick_params(axis='both', which='major', labelsize=15)
    x_ax = [5114,5479,5844,6210,6575,6940]
    plt.xticks(x_ax, ('1/1/2014','1/1/2015', '1/1/2016', '1/1/2017','1/1/2018','1/1/2019'), size = 18)
    plt.xlim(5000,7000)
    plt.ylim(0,100)

    plt.plot([0,7000],[50,50],color = 'black', lw = 3)
    plt.show()
    
def run_GP(X,y):
    k1 = gp.kernels.RBF(input_dim=2, lengthscale=torch.tensor(50.0),\
                   variance = torch.tensor(0.5))
    #k1.set_constraint("lengthscale", torch.distributions.constraints.interval(7.,1000.))
    smoke_test = ('CI' in os.environ)  # ignore; used to check code integrity in the Pyro repo
    pyro.enable_validation(True)       # can help with debugging
    optim = Adam({"lr": 0.01}) 
    
    pyro.clear_param_store()

    plus_arr = np.max(X)+np.array([7,14,21,28,35,42,49,56])

    X2 = (torch.from_numpy(X))
    y2 = (torch.from_numpy(y-50))

    Xtest_use = np.append(X,plus_arr)
    Xtest_use2 = (torch.from_numpy(Xtest_use))



    gpr = gp.models.GPRegression(X2, y2,k1, noise=torch.tensor(0.01))

    svi = SVI(gpr.model, gpr.guide, optim, loss=Trace_ELBO())
    losses = []

    num_steps = 10

    for k in range(num_steps):
        losses.append(svi.step())



    with torch.no_grad():
      if type(gpr) == gp.models.VariationalSparseGP:
        mean, cov = gpr(Xtest_use2, full_cov=True)
      else:
        mean, cov = gpr(Xtest_use2, full_cov=False, noiseless=False) 

    sd = cov.sqrt().detach().numpy()
    mean = mean.detach().numpy()+50
    
    return mean, Xtest_use 
    
    
def plot_vci_fc(Xtest_use,mean,X,y):

    plt.figure(figsize=(17, 7))
    plt.plot(Xtest_use,mean, linestyle = 'solid', lw = 3, color = 'red', label = 'GP prediction')
    plt.plot(X,y, linestyle = 'solid', lw = 3, color = 'blue',label = 'Landsat VCI')
    plt.xlabel('Days since 1/1/2000', size = 20)
    plt.ylabel('weekly VCI', size = 20)
    plt.tick_params(axis='both', which='major', labelsize=15)
    x_ax = [5114,5479,5844,6210,6575,6940]
    plt.xticks(x_ax, ('1/1/2014','1/1/2015', '1/1/2016', '1/1/2017','1/1/2018','1/1/2019'), size = 18)
    plt.xlim(5000,7020)
    plt.ylim(0,100)

    plt.plot([0,7020],[50,50],color = 'black', lw = 3)
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
    plt.plot(Xtest_use,yf_new, linestyle = 'solid', lw = 3, color = 'red',label = 'GP prediction')
    #plt.plot(Xtest_use,mean+3*sd, linestyle = '--', lw = 3, color = 'red')
    #plt.plot(Xtest_use,mean-3*sd, linestyle = '--', lw = 3, color = 'red')


    plt.plot(X,y_new, linestyle = 'solid', lw = 3, color = 'blue',label = 'Landsat VCI3M')

    plt.xlabel('Days since 1/1/2000', size = 20)
    plt.ylabel('VCI3M', size = 20)
    plt.tick_params(axis='both', which='major', labelsize=15)
    x_ax = [5114,5479,5844,6210,6575,6940]
    plt.xticks(x_ax, ('1/1/2014','1/1/2015', '1/1/2016', '1/1/2017','1/1/2018','1/1/2019'), size = 18)
    plt.xlim(5000,7020)
    plt.ylim(0,100)

    plt.plot([0,7020],[50,50],color = 'black', lw = 3)
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