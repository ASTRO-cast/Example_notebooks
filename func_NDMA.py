import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import pyro
import pyro.contrib.gp as gp
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import os
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.double)



def load():
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
    return Y,M,V, region

def plot(Y,M,V):
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
    
    
def GP(Y,M,V):  
    X = np.arange(np.size(M))
    k1 = gp.kernels.RBF(input_dim=2, lengthscale=torch.tensor(0.8),\
                   variance = torch.tensor(2.5))
    smoke_test = ('CI' in os.environ)  # ignore; used to check code integrity in the Pyro repo
    pyro.enable_validation(True)       # can help with debugging
    optim = Adam({"lr": 0.01}) 

    pyro.clear_param_store()

    plus_arr = np.max(X)+np.array([0.5,1,1.5,2,2.5])

    X2 = (torch.from_numpy(X.astype(float)))
    y2 = (torch.from_numpy(V-np.mean(V)))

    Xtest_use = np.append(X.astype(float),plus_arr.astype(float))

    Xtest_use2 = (torch.from_numpy(Xtest_use))


    gpr = gp.models.GPRegression(X2, y2,k1, noise=torch.tensor(0.01))

    svi = SVI(gpr.model, gpr.guide, optim, loss=Trace_ELBO())
    losses = []

    num_steps = 500

    for k in range(num_steps):
        losses.append(svi.step())

    with torch.no_grad():
      if type(gpr) == gp.models.VariationalSparseGP:
        mean, cov = gpr(Xtest_use2, full_cov=True)
      else:
        mean, cov = gpr(Xtest_use2, full_cov=False, noiseless=False) 

    sd = cov.sqrt().detach().numpy()
    mean = mean.detach().numpy()+np.mean(V)

    #for param_name in pyro.get_param_store().get_all_param_names():
    #    print('{}={}'.format(param_name,pyro.param(param_name).item()))
        
        
    return mean, Xtest_use, X

def plot_fc(mean, Xtest_use, X, V, M ,Y,region):
    plt.figure(figsize=(17, 7))
    plt.plot([np.max(X),np.max(X)],[0,100],color = 'black', lw = 3)
    plt.plot([0,800],[35,35],color = 'black', lw = 3)

    plt.plot(X,V, linestyle = 'solid', lw = 3, color = 'blue')
    plt.plot(Xtest_use,mean, linestyle = 'solid', lw = 3, color = 'red')

    x_ax = np.array([10+7*24,6+10+7*24,12+10+7*24,18+10+7*24,10+8*24,6+10+8*24])
    plt.xticks(x_ax, ('Jan 2016','July 2016','Jan 2017','July 2017','Jan 2018','July 2018'), size = 18)   
                      
    plt.ylabel('VCI3M', size = 20)
    
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.xlim(175,210)
    plt.ylim(0,100)
    plt.show()
    
    print('VCI3M = ',"%.0f" % mean[-4],' for Month:', M[-1]+1,' Year:',Y[-1], 'region: '+region)
    if mean[-4] > mean[-6]:
        print('Trend:'+'UP')
    else:
        print('Trend:'+'DOWN')
    if mean[-4] < 35: 
        print('VCI3M < 35 next month')


