#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


# ## Functions

# ### 1. Functions for data generation

# In[2]:


def makeb(d_p,N):
    Bs=list()
    Bs.append(np.zeros(d_p*(N-1)))
    for i in range(N-1):
        Bs.append(np.zeros(d_p*(N-1)))
        x=np.random.normal(0,1,d_p)
        u=np.random.uniform(0,1,1)
        b=u**(1./d_p)*x/np.sqrt(np.sum(x**2))
        Bs[i+1][(i*d_p):((i+1)*d_p)]=b        
    return(Bs)


# In[3]:


def makeb2(d_p,N):
    Bs=list()
    x=np.random.normal(0,1,(d_p-1))
    Bs.append(np.zeros(d_p*(N-1)))
    for i in range(N-1):
        Bs.append(np.zeros(d_p*(N-1)))
        Bs[i+1][(i*d_p)]=1.
        Bs[i+1][(i*d_p+1):((i+1)*d_p)]=x  
    return(Bs)
    


# In[4]:


def makedata(d_p,N,T,makeb_style,R):
    Bs_list=list()
    errors=list()
    if makeb_style==0:
        for t in range(T):
            Bs=makeb(d_p,N)
            Bs_list.append(Bs)
            errors.append(np.random.multivariate_normal(np.zeros(N),np.eye(N)*(R**2)))
    else:
        for t in range(T):
            Bs=makeb2(d_p,N)
            Bs_list.append(Bs)
            errors.append(np.random.multivariate_normal(np.zeros(N),np.eye(N)*(R**2)))
    return([Bs_list,errors])


# ### 2. Function for computing the optimal reward and the nonparametric intercept term

# In[5]:


def cumul_opt(T,p_min,p_max,Bs_list,mu):
    reward=np.zeros(T)
    vs=np.zeros(T)
    for t in range(T):
        tru_nonzero=np.array([np.dot(b,mu) for b in Bs_list[t][1:]])
        opt_nonzero=np.argmax(tru_nonzero)
        if tru_nonzero[opt_nonzero]<0:
            pt_opt=p_min
        else:
            pt_opt=p_max            
        reward[t]=tru_nonzero[opt_nonzero]*pt_opt
        # compute nu(t), the nonparametric intercept term
        #vs[t]=0 ## Case (i): nu(t)=0
        vs[t]=-reward[t] ## Case (ii): nu(t)=-b_{a^*(t)}^T mu
        #vs[t]=np.log(t+1) ## Case (iii): nu(t)=log(t+1)
    return([np.cumsum(reward),vs])
    


# ### 3. Functions for algorithms

# In[6]:


### Thompson Sampling algorithm ###

def standard_TS(T,d,v,Bs_list,errors,mu,vs):
    mu_hat=np.zeros(d+1)
    y=np.zeros(d+1)
    reward=np.zeros(T)
    B=np.eye(d+1)
    for t in range(T):
        V=(v**2)*np.linalg.inv(B)
        mu_tilde=np.random.multivariate_normal(mu_hat,V)[1:]
        est=np.array([np.dot(b,mu_tilde) for b in Bs_list[t]])
        action=np.argmax(est)
        reward[t]=np.dot(Bs_list[t][action],mu)
        newb=np.array([1.]+list(Bs_list[t][action]))
        B=B+np.outer(newb,newb)
        y=y+(np.dot(Bs_list[t][action],mu)+vs[t]+errors[t][action])*newb
        mu_hat=np.linalg.inv(B).dot(y)
    print(np.sqrt(np.dot(mu_hat[1:]-mu,mu_hat[1:]-mu)))
    return(np.cumsum(reward))


# In[7]:


### Action-Centered Thompson Sampling algorithm (Greenewald et al., 2017) ###

def action_centered_TS(T,d,v,p_min,p_max,Bs_list,errors,mu,vs):
    mu_hat=np.zeros(d)
    y=np.zeros(d)
    reward=np.zeros(T)
    reward2=np.zeros(T)
    B=np.eye(d)
    B2=np.eye(d+1)
    y2=np.zeros(d+1)
    for t in range(T):
        V=(v**2)*np.linalg.inv(B)
        mu_tilde=np.random.multivariate_normal(mu_hat,V)
        est=np.array([np.dot(b,mu_tilde) for b in Bs_list[t][1:]])
        action=np.argmax(est)+1
        val_mean=np.dot(Bs_list[t][action],mu_hat)
        val_sd=np.sqrt(np.dot(Bs_list[t][action],V.dot(Bs_list[t][action])))
        pt=np.maximum(p_min,np.minimum((1.-norm.cdf(0.,val_mean,val_sd)),p_max))
        reward[t]=np.dot(Bs_list[t][action],mu)*pt
        choice=np.random.choice([0,1],p=[1-pt,pt])
        if choice==1:
            newb=np.array([1.]+list(Bs_list[t][action]))
            B2=B2+np.outer(newb,newb)
            y2=y2+(np.dot(Bs_list[t][action],mu)+vs[t]+errors[t][action])*newb
            intercept=np.linalg.inv(B2).dot(y2)[0]
            y=y+(1-pt)*(-intercept+np.dot(Bs_list[t][action],mu)+vs[t]+errors[t][action])*Bs_list[t][action]
            reward2[t]=np.dot(Bs_list[t][action],mu)
        else:
            newb=np.array([1.]+list(Bs_list[t][0]))
            B2=B2+np.outer(newb,newb)
            y2=y2+(vs[t]+errors[t][0])*newb
            intercept=np.linalg.inv(B2).dot(y2)[0]
            y=y+(-pt)*(-intercept+vs[t]+errors[t][0])*Bs_list[t][action]
        B=B+pt*(1-pt)*np.outer(Bs_list[t][action],Bs_list[t][action])
        mu_hat=np.linalg.inv(B).dot(y)
    print(np.sqrt(np.dot(mu_hat-mu,mu_hat-mu)))
    return(np.cumsum(reward))        


# In[8]:


### Proposed Algorithm ###

## A US patent has been filed for the Proposed_TS algorithm. 
## Please contact to myungheechopaik@snu.ac.kr in case of commercial use. 

def Proposed_TS(T,d,v,N,Bs_list,errors,mu,vs):
    mu_hat=np.zeros(d)
    y=np.zeros(d)
    reward=np.zeros(T)
    B=np.eye(d)
    B2=np.eye(d+1)
    y2=np.zeros(d+1)
    intercept=0
    for t in range(T):
        V=(v**2)*np.linalg.inv(B)
        mu_tilde=np.random.multivariate_normal(mu_hat,V)
        est=np.array([np.dot(b,mu_tilde) for b in Bs_list[t]])
        action=np.argmax(est)
        mu_mc=np.random.multivariate_normal(mu_hat,V,1000)
        est_mc=list((np.dot(Bs_list[t],mu_mc.T)).T)
        ac_mc=list(np.argmax(est_mc,axis=1))
        pi_est=np.array([float(ac_mc.count(n))/len(ac_mc) for n in range(N)])
        b_mean=np.dot(np.transpose(np.array(Bs_list[t])),pi_est)
        B=B+2*np.outer(Bs_list[t][action]-b_mean,Bs_list[t][action]-b_mean)
        B=B+2*np.dot(np.dot(np.transpose(Bs_list[t]),np.diag(pi_est)),Bs_list[t])-2*np.outer(b_mean,b_mean)
        y=y+4*(Bs_list[t][action]-b_mean)*(-intercept+vs[t]+errors[t][action]+np.dot(Bs_list[t][action],mu))
        reward[t]=np.dot(Bs_list[t][action],mu)
        mu_hat=np.linalg.inv(B).dot(y)
        newb=np.array([1.]+list(Bs_list[t][action]))
        B2=B2+np.outer(newb,newb)
        y2=y2+(np.dot(Bs_list[t][action],mu)+vs[t]+errors[t][action])*newb
        intercept=np.linalg.inv(B2).dot(y2)[0]
    print(np.sqrt(np.dot(mu_hat-mu,mu_hat-mu)))
    return(np.cumsum(reward))        


# In[9]:


### BOSE algorithm for N=2 (Krishnamurthy et al., 2018) ###

def Bose(T,d,v,N,Bs_list,errors,mu,vs):
    mu_hat=np.zeros(d)
    y=np.zeros(d)
    reward=np.zeros(T)
    B=np.eye(d)
    B2=np.eye(d+1)
    y2=np.zeros(d+1)
    intercept=0
    for t in range(T):
        survivors=[]
        for i in range(N):
            fail=False
            for j in range(N):
                vec=Bs_list[t][j]-Bs_list[t][i]
                tmp=np.dot(vec,mu_hat)-v*np.sqrt(np.dot(vec,np.linalg.inv(B).dot(vec)))
                if tmp>0:
                    fail=True
                    break
            if not fail:
                survivors.append(i)
        if len(survivors)==1:
            action=survivors[0]
        if len(survivors)>=2:
            action=np.random.choice(survivors)
            new_list=[Bs_list[t][k] for k in survivors]
            pi_est=np.ones(len(survivors))*1./len(survivors)
            b_mean=np.dot(np.transpose(np.array(new_list)),pi_est)
            B=B+np.outer(Bs_list[t][action]-b_mean,Bs_list[t][action]-b_mean)
            y=y+(Bs_list[t][action]-b_mean)*(-intercept+vs[t]+errors[t][action]+np.dot(Bs_list[t][action],mu))
            mu_hat=np.linalg.inv(B).dot(y)
            
        reward[t]=np.dot(Bs_list[t][action],mu)    
        newb=np.array([1.]+list(Bs_list[t][action]))
        B2=B2+np.outer(newb,newb)
        y2=y2+newb*(vs[t]+errors[t][action]+np.dot(Bs_list[t][action],mu))
        intercept=np.linalg.inv(B2).dot(y2)[0]
    print(np.sqrt(np.dot(mu_hat-mu,mu_hat-mu)))
    return(np.cumsum(reward))            


# ## Simulation Settings

# In[10]:


simul_n=50 # number of simulations
T=10000 # number of time steps
R=0.1
d_p=10 # d prime
N=2 # number of actions
d=d_p*(N-1) # dimension of mu
p_min=0.; p_max=1. # p_min and p_max for action-centered TS algorithm
makeb_style=0
mu=2*np.array([-0.275,0.333,-0.045,-0.116,0.122,0.275,-0.333,0.045,0.116,-0.122])


# ### 1. Selection of tuning parameter

# In[11]:


# Candidate values of tuning parameter for each algorithm.

Vstd=[0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.02]
Vac=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2]
Vprop=[0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.2,0.3,0.4]
Vbose=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2]


# In[12]:


# Lists to store the median regret values obtained by each algorithm for each candidate value of tuning parameter.

valStd=[]
valAC=[]
valProp=[]
valBose=[]


# In[13]:


Data_list=[]


# In[14]:


np.random.seed(1)

for simul in range(simul_n):
    Data_list.append(makedata(d_p=d_p,N=N,T=T,makeb_style=makeb_style,R=R))


for tp in range(11):
    
    cumulated_opt=list()
    cumulated_reward_Std=list()
    cumulated_reward_AC=list()
    cumulated_reward_Prop=list()
    cumulated_reward_Bose=list()

    cumulated_regret_Std=list()
    cumulated_regret_AC=list()
    cumulated_regret_Prop=list()
    cumulated_regret_Bose=list()

    for simul in range(simul_n):
    
        Data=Data_list[simul]
        Bs_list=Data[0]
        errors=Data[1]
    
        CO=cumul_opt(T=T,p_min=0.,p_max=1.0,Bs_list=Bs_list,mu=mu)
        cumulated_opt.append(CO[0])
        vs=CO[1]
        cumulated_reward_Std.append(standard_TS(T=T,d=d,v=Vstd[tp],Bs_list=Bs_list,errors=errors,mu=mu,vs=vs))
        cumulated_reward_AC.append(action_centered_TS(T=T,d=d,v=Vac[tp],p_min=p_min,p_max=p_max,Bs_list=Bs_list,errors=errors,mu=mu,vs=vs))
        cumulated_reward_Prop.append(Proposed_TS(T=T,d=d,v=Vprop[tp],N=N,Bs_list=Bs_list,errors=errors,mu=mu,vs=vs))
        cumulated_reward_Bose.append(Bose(T=T,d=d,v=Vbose[tp],N=N,Bs_list=Bs_list,errors=errors,mu=mu,vs=vs))
        
        cumulated_regret_Std.append(cumulated_opt[simul]-cumulated_reward_Std[simul])
        cumulated_regret_AC.append(cumulated_opt[simul]-cumulated_reward_AC[simul])
        cumulated_regret_Prop.append(cumulated_opt[simul]-cumulated_reward_Prop[simul])
        cumulated_regret_Bose.append(cumulated_opt[simul]-cumulated_reward_Bose[simul])
    
        print(simul)
        
    valStd.append(np.median(cumulated_regret_Std,axis=0)[9999])
    valAC.append(np.median(cumulated_regret_AC,axis=0)[9999])
    valProp.append(np.median(cumulated_regret_Prop,axis=0)[9999])
    valBose.append(np.median(cumulated_regret_Bose,axis=0)[9999])
    
    


# In[15]:


np.round(valStd,2)


# In[16]:


np.round(valAC,2)


# In[17]:


np.round(valProp,2)


# In[18]:


np.round(valBose,2)


# ### 2. Simulation

# In[19]:


cumulated_opt=list()
cumulated_reward_Std=list()
cumulated_reward_AC=list()
cumulated_reward_Prop=list()
cumulated_reward_Bose=list()

cumulated_regret_Std=list()
cumulated_regret_AC=list()
cumulated_regret_Prop=list()
cumulated_regret_Bose=list()


# In[20]:


Data_list=[]


# In[21]:



np.random.seed(1)

for simul in range(simul_n):
    Data_list.append(makedata(d_p=d_p,N=N,T=T,makeb_style=makeb_style,R=R))


for simul in range(simul_n):
    
    Data=Data_list[simul]
    Bs_list=Data[0]
    errors=Data[1]
    
    CO=cumul_opt(T=T,p_min=0.,p_max=1.0,Bs_list=Bs_list,mu=mu)
    cumulated_opt.append(CO[0])
    vs=CO[1]
    cumulated_reward_Std.append(standard_TS(T=T,d=d,v=0.005,Bs_list=Bs_list,errors=errors,mu=mu,vs=vs))
    cumulated_reward_AC.append(action_centered_TS(T=T,d=d,v=0.04,p_min=p_min,p_max=p_max,Bs_list=Bs_list,errors=errors,mu=mu,vs=vs))
    cumulated_reward_Prop.append(Proposed_TS(T=T,d=d,v=0.12,N=N,Bs_list=Bs_list,errors=errors,mu=mu,vs=vs))
    cumulated_reward_Bose.append(Bose(T=T,d=d,v=0.04,N=N,Bs_list=Bs_list,errors=errors,mu=mu,vs=vs))
    
    cumulated_regret_Std.append(cumulated_opt[simul]-cumulated_reward_Std[simul])
    cumulated_regret_AC.append(cumulated_opt[simul]-cumulated_reward_AC[simul])
    cumulated_regret_Prop.append(cumulated_opt[simul]-cumulated_reward_Prop[simul])
    cumulated_regret_Bose.append(cumulated_opt[simul]-cumulated_reward_Bose[simul])
    
    print(simul)
    


# In[22]:


np.median(cumulated_regret_Std,axis=0)[9999]


# In[23]:


np.median(cumulated_regret_AC,axis=0)[9999]


# In[24]:


np.median(cumulated_regret_Prop,axis=0)[9999]


# In[25]:


np.median(cumulated_regret_Bose,axis=0)[9999]


# In[26]:


steps=np.arange(1,T+1)

plt.plot(steps,np.median(cumulated_regret_Std,axis=0),'r',label='Standard TS')
plt.plot(steps,np.median(cumulated_regret_AC,axis=0),'b',label='Action Centered TS')
plt.plot(steps,np.median(cumulated_regret_Prop,axis=0),'g',label='Proposed TS')
plt.plot(steps,np.median(cumulated_regret_Bose,axis=0),'k',label='BOSE')


plt.xlabel('Decision Point')
plt.ylabel('Cumulative Regret')
plt.title('Experiment Result')
plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.2),fancybox=True,ncol=5)
plt.show()


# In[27]:


steps=np.arange(1,T+1)

plt.plot(steps,np.median(cumulated_regret_Std,axis=0),'r',label='Standard TS')
plt.plot(steps,np.percentile(cumulated_regret_Std,25,axis=0),'r',linestyle='--')
plt.plot(steps,np.percentile(cumulated_regret_Std,75,axis=0),'r',linestyle='--')

plt.plot(steps,np.median(cumulated_regret_AC,axis=0),'b',label='Action Centered TS')
plt.plot(steps,np.percentile(cumulated_regret_AC,25,axis=0),'b',linestyle='--')
plt.plot(steps,np.percentile(cumulated_regret_AC,75,axis=0),'b',linestyle='--')

plt.plot(steps,np.median(cumulated_regret_Prop,axis=0),'g',label='Proposed TS')
plt.plot(steps,np.percentile(cumulated_regret_Prop,25,axis=0),'g',linestyle='--')
plt.plot(steps,np.percentile(cumulated_regret_Prop,75,axis=0),'g',linestyle='--')

plt.plot(steps,np.median(cumulated_regret_Bose,axis=0),'k',label='BOSE')
plt.plot(steps,np.percentile(cumulated_regret_Bose,25,axis=0),'k',linestyle='--')
plt.plot(steps,np.percentile(cumulated_regret_Bose,75,axis=0),'k',linestyle='--')

plt.xlabel('Decision Point')
plt.ylabel('Cumulative Regret')
plt.title('Experiment Result')
plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.2),fancybox=True,ncol=5)
plt.show()

