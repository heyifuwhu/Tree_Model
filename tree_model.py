# =============================================================================
# Binomial tree
# =============================================================================

import numpy as np
import pandas as pd
from math import log as log
from math import sqrt as sqrt
from math import exp as exp
from scipy.stats import norm
from numpy import nan
import math

# Additive tree
## Additive binomial tree model of European Option
def additive_binomial_European(Option_type,S,K,r,vol,T,N):
    
    """
    
    """
    # precompute constants
    dt = T/N 
    nu = r - 0.5 * vol**2
    xu = math.sqrt(dt * (vol**2)+ (nu**2) * (dt ** 2))
    xd = -xu
    pu = 0.5 +0.5*((nu*dt)/xu)
    pd = 1-pu
    discount = np.exp(-r*dt)
    
    # initialize asset price
    St = S*np.exp(np.asarray([(N-i)*xu + i*xd for i in range(N+1)]))
    # initialize value at maturity
    if Option_type == "call":
        C = np.where(St >= K, St-K,0)
    if Option_type == "put":
        C = np.where(K >= St, K-St,0)
   
    # calculate the option price
    while (len(C) >1):
        C = discount*(pu*C[:-1]+pd*C[1:])
    return C[0]

## Additive binomial tree model of American Option
def additive_binomial_American(Option_type,S,K,r,vol,T,N):
    
    """
    
    """
    # precompute constants
    dt = T/N 
    nu = r - 0.5 * vol**2
    xu = math.sqrt(dt * (vol**2)+ (nu**2) * (dt ** 2))
    xd = -xu
    pu = 0.5 +0.5*((nu*dt)/xu)
    pd = 1-pu
    discount = np.exp(-r*dt)
    
    # initialize asset price
    St = S*np.exp(np.asarray([(N-i)*xu + i*xd for i in range(N+1)]))
    # initialize value at maturity 
    if Option_type == "call":
        C = np.where(St >= K, St-K,0)
        while (len(C) >1):
            C = discount*(pu*C[:-1]+pd*C[1:])
            St = np.exp(xd)*St[:-1]
            C = np.where(C >= (St-K), C, St-K)

    if Option_type == "put":
        C = np.where(K >= St, K-St,0)
        while (len(C) >1):
            C = discount*(pu*C[:-1]+pd*C[1:])
            St = np.exp(xd)*St[:-1]
            C = np.where(C >= (K-St), C, K-St)
    return C[0]


S=100
K=100
r=0.06
vol=0.2
T=1
N=3
print(f'European_call: {additive_binomial_European("call",S,K,r,vol,T,N)}')
print(f'European_put: {additive_binomial_European("put",S,K,r,vol,T,N)}')
print(f'American_call: {additive_binomial_American("call",S,K,r,vol,T,N)}')
print(f'American_put: {additive_binomial_American("put",S,K,r,vol,T,N)}')


# Mutiplicative tree
## Multiplicative tree model of European Option
def multiplicative_tree_European(Option_type,S,K,r,N,T,u,d):
    dt = T/N
    p = (np.exp(r*dt)-d)/(u-d)
    disc = np.exp(-r*dt)
    
    # initialize asset price
    St = np.asarray([(S * u**(N-i)) * (d**i) for i in range(N+1)])
    
    # call
    if Option_type == "call":
        #initialize value at maturity
        C = np.where(St >= K, St-K,0)
        while(len(C) >1):
            C = disc*(p*C[:-1]+(1-p)*C[1:])
   
    # put
    elif Option_type == "put":
        #initialize value at maturity
        C = np.where(St <= K, K-St,0)
        while(len(C) >1):
            C = disc*(p*C[:-1]+(1-p)*C[1:])
   
    return C[0]

## Multiplicative tree model of American Option
def multiplicative_tree_American(Option_type,S,K,r,N,T,u,d):
    dt = T/N
    p = (np.exp(r*dt)-d)/(u-d)
    disc = np.exp(-r*dt)
    
    # initialize asset price
    St = np.asarray([(S * u**(N-i)) * (d**i) for i in range(N+1)])
    
    # call
    if Option_type == "call":
        #initialize value at maturity
        C = np.where(St >= K, St-K,0)
        while(len(C) >1):
            C = disc*(p*C[:-1]+(1-p)*C[1:])
            St = St[:-1]/u
            C = np.where(C>(St-K),C,St-K)
   
    # put
    elif Option_type == "put":
        #initialize value at maturity
        C = np.where(St <= K, K-St,0)
        while(len(C) >1):
            C = disc*(p*C[:-1]+(1-p)*C[1:])
            St = St[:-1]/u
            C = np.where(C>(K-St),C,K-St)
   
    return C[0]


S=100
K=100
T=1
r=0.06
N=3
u=1.1
d=1/u
print(f'European_call: {multiplicative_tree_European("call",S,K,r,N,T,u,d)}')
print(f'European_put: {multiplicative_tree_European("put",S,K,r,N,T,u,d)}')
print(f'American_call: {multiplicative_tree_American("call",S,K,r,N,T,u,d)}')
print(f'American_put: {multiplicative_tree_American("put",S,K,r,N,T,u,d)}')

# Generalized fomulation of multiplicative binomial tree
## General formulation of multiplicative tree model of European Option
def gen_multiplicative_tree_European(Option_type,S,K,r,N,T,vol):
    dt = T/N
    u = np.exp(vol*np.sqrt(dt))
    d = 1/u
    p = 0.5 + 0.5*np.sqrt(dt)*(r - 0.5*vol**2)/vol
    disc = np.exp(-r*dt)
    
    # initialize asset price
    St = np.asarray([(S * u**(N-i)) * (d**i) for i in range(N+1)])
    
    # call
    if Option_type == "call":
        #initialize value at maturity
        C = np.where(St >= K, St-K,0)
        while(len(C) >1):
            C = disc*(p*C[:-1]+(1-p)*C[1:])
   
    # put
    elif Option_type == "put":
        #initialize value at maturity
        C = np.where(St <= K, K-St,0)
        while(len(C) >1):
            C = disc*(p*C[:-1]+(1-p)*C[1:])
   
    return C[0]

## General formulation of multiplicative tree model of Amecican Option
def gen_multiplicative_tree_American(Option_type,S,K,r,N,T,vol):
    dt = T/N
    u = np.exp(vol*np.sqrt(dt))
    d = 1/u
    p = 0.5 + 0.5*np.sqrt(dt)*(r - 0.5*vol**2)/vol
    disc = np.exp(-r*dt)
    
    # initialize asset price
    St = np.asarray([(S * u**(N-i)) * (d**i) for i in range(N+1)])
    
    # call
    if Option_type == "call":
        #initialize value at maturity
        C = np.where(St >= K, St-K,0)
        while(len(C) >1):
            C = disc*(p*C[:-1]+(1-p)*C[1:])
            St = St[:-1]/u
            C = np.where(C>(St-K),C,St-K)
   
    # put
    elif Option_type == "put":
        #initialize value at maturity
        C = np.where(St <= K, K-St,0)
        while(len(C) >1):
            C = disc*(p*C[:-1]+(1-p)*C[1:])
            St = St[:-1]/u
            C = np.where(C>(K-St),C,K-St)
   
    return C[0]

S=100
K=100
T=1
r=0.06
N=3
vol = 0.25

print(f'European_call: {gen_multiplicative_tree_European("call",S,K,r,N,T,vol)}')
print(f'European_put: {gen_multiplicative_tree_European("put",S,K,r,N,T,vol)}')
print(f'American_call: {gen_multiplicative_tree_European("call",S,K,r,N,T,vol)}')
print(f'American_put: {gen_multiplicative_tree_American("put",S,K,r,N,T,vol)}')


## use implied volatility to verify the tree model

def BSM(Option_type,S,K,vol,T,r):
    """
    Black-Scholes Model
    
    Option_type: "call" or "put"
    S: spot price
    K: strike price
    vol : volatility
    T : Maturity
    r: risk-free rate
    
    """
    d1 = (log(S / K) + (r + 0.5 * vol**2) * T )/ (vol * sqrt(T))
    d2 = d1 - vol * sqrt(T)
          
    if Option_type == "call":
        Option = S * norm.cdf(d1) - K * exp(-r * T)*norm.cdf(d2)
        return Option
    elif Option_type == "put":
        Option = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return Option
    else:
        return "Error: parameter Option_type only takes in 'call or 'put'" 
    

def bisection(left, right, f, epsilon):
    """
    bisection root-finding method
    
    left : the left side of the interval 
    right : the right side of the interval
    f : the function
    epsilon : the error to tolerate
    
    """
    if f(left) * f(right) > 0:
        return nan
        #return "No root exists in ({},{})".format(a,b)
    elif abs(f(left)) < epsilon:
        return left
    elif abs(f(right)) < epsilon:
        return right
    else:
        while abs(f((left + right) / 2)) > epsilon:
            if f(left)* f((left + right) / 2) < 0:
                right=(left + right) / 2
            elif f(left)* f((left + right) / 2) > 0:
                left=(left + right) / 2 
        return (left + right) / 2

def implied_vol(Option_type,S,K,T,r,p_market):
    def f(x):
        return BSM(Option_type,S,K,x,T,r) - p_market
    return f



path = "/Users/yifuhe/Desktop/"
data = pd.read_csv(path+"SPY_data2.csv")

data["Strike"]
data1 =data[(data["Expiry"]=="2019-02-15") & (data["Strike"] >= 264) & (data["Strike"] <= 274)]
data2 =data[(data["Expiry"]=="2019-03-15") & (data["Strike"] >= 263) & (data["Strike"] <= 282)]
data3 =data[(data["Expiry"]=="2019-04-18") & (data["Strike"] >= 268) & (data["Strike"] <= 277)]   

data1 = pd.concat([data1,data1,data3],axis=0).reset_index()
data1["Option"] = (data1["Ask"] + data1["Bid"])/2
data1["implied_vol"] = 0
for i in range(len(data1)):
    data1.loc[i,"implied_vol"] = bisection(0.01,3,implied_vol(
    data1["Type"][i],
    float(data1["Underlying_Price"][i]),
    float(data1["Strike"][i]),
    float(data1["Maturity"][i]),
    0.0075,
    float(data1["Option"][i])),
    1.e-6)

    
for i in range(len(data1)):
    data1.loc[i,"American"] = additive_binomial_American(data1["Type"][i],
             data1["Underlying_Price"][i],
             data1["Strike"][i],
             0.0075,
             data1["implied_vol"][i],
             data1["Maturity"][i],200)
for i in range(len(data1)):
    data1.loc[i,"European"] = additive_binomial_European(data1["Type"][i],
             data1["Underlying_Price"][i],
             data1["Strike"][i],
             0.0075,
             data1["implied_vol"][i],
             data1["Maturity"][i],200)
    
print("I only show the first 10 rows")
data1.loc[0:9,["European", "American", "Market_P"]]    


## see the error decreases when the number of steps increases
import matplotlib.pyplot as plt
%matplotlib inline 
Option_type ="put"
S = 100
K = 100
T = 1
r = 0.06
vol =0.2
error = []
x = [10,20,30,40,50,100,150,200,250,300,350,400]
for N in x :
    error.append(abs(additive_binomial_European(Option_type,S,K,r,vol,T,N) -
                     BSM(Option_type,S,K,vol,T,r) ))
print(error)
plt.plot(x,error)


## bonus 
## calculate the volatility throuth binomial tree
for i in range(len(data1)): 
    data1.loc[i,'American_Binomial_vol']=bisection(0.01,3,implied_vol(
    data1["Type"][i],
    float(data1["Underlying_Price"][i]),
    float(data1["Strike"][i]),
    float(data1["Maturity"][i]),
    0.0075,
    float(data1["American"][i])),
    1.e-6)

print("I only show the first 10 rows")
data1.loc[0:9,["implied_vol","American_Binomial_vol"]] 



# =============================================================================
# Trinomial tree         
# =============================================================================
# trinomial tree model of European option
def trinomial_tree_European_dividend(Option_type,S,K,r,vol,T,N,div,dx):
    """
    
    """
    
    # precompute constants
    dt = T/N
    nu = r - div - 0.5*vol**2
    pu = 0.5*( (vol**2 * dt + (nu*dt)**2)/(dx**2) + (nu*dt)/(dx))
    pd = 0.5*( (vol**2 * dt + (nu*dt)**2)/(dx**2) - (nu*dt)/(dx))
    pm = 1 - pu - pd
    disc = np.exp(-r*dt)
    
    # initialize asset price
    St = S * np.exp(np.asarray([(N-i)*dx for i in range(2*N+1)]))
    
    # consider option type : call
    if Option_type == "call":
        # initialize option value
        Ct = np.where(St >= K, St-K,0)
        while(len(Ct)>1):
            Ct = disc*(pu*Ct[:-2] + pm*Ct[1:-1] + pd*Ct[2:])
    
    # consider option type : call
    if Option_type == "put":
         # initialize option value
        Ct = np.where(K >= St, K-St,0)
        while(len(Ct)>1):
            Ct = disc*(pu*Ct[:-2] + pm*Ct[1:-1] + pd*Ct[2:])
    return Ct[0]

# trinomial tree model of American option        
def trinomial_tree_American_dividend(Option_type,S,K,r,vol,T,N,div,dx):
    """
    
    """
    
    # precompute constants
    dt = T/N
    nu = r - div - 0.5*vol**2
    pu = 0.5*( (vol**2 * dt + (nu*dt)**2)/(dx**2) + (nu*dt)/(dx))
    pd = 0.5*( (vol**2 * dt + (nu*dt)**2)/(dx**2) - (nu*dt)/(dx))
    pm = 1 - pu - pd
    disc = np.exp(-r*dt)
    
    # initialize asset price
    St = S * np.exp(np.asarray([(N-i)*dx for i in range(2*N+1)]))
    
    # consider option type
    if Option_type == "call":
        # initialize option value
        Ct = np.where(St >= K, St-K,0)
        while(len(Ct)>1):
            St = St[1:-1]
            Ct = disc*(pu*Ct[:-2] + pm*Ct[1:-1] + pd*Ct[2:])
            Ct = np.where(Ct >= (St-K),Ct,(St-K))
            
    
    if Option_type == "put":
         # initialize option value
        Ct = np.where(K >= St, K-St,0)
        while(len(Ct)>1):
            St = St[1:-1]
            Ct = disc*(pu*Ct[:-2] + pm*Ct[1:-1] + pd*Ct[2:])
            Ct = np.where(Ct >= (K-St),Ct,(K-St))
    return Ct[0]


Option_type = "call"
S = 100
K = 100
T = 1
r = 0.06
div = 0.03
N = 3
dx = 0.2
vol =0.2
stability = dx >= vol* np.sqrt(3*(T/N))

print(f'Stability condition is {stability}')
print("European_call: ")
print(trinomial_tree_European_dividend("call",S,K,r,vol,T,N,div,dx))
print("European_put: ")
print(trinomial_tree_European_dividend("put",S,K,r,vol,T,N,div,dx))
print("American_call: ")
print(trinomial_tree_American_dividend("call",S,K,r,vol,T,N,div,dx))
print("American_put: ")
print(trinomial_tree_American_dividend("put",S,K,r,vol,T,N,div,dx))


## compare the value with BSM
def BSM_dividend(Option_type,S,K,vol,div,expir,r):
    d1 = (log(S / K) + (r- div+ 0.5 * vol**2) * expir) / (vol * sqrt(expir))
    d2 = d1 - vol * sqrt(expir)
    if Option_type == "call":
        option_price = S *exp(- div*expir)* norm.cdf(d1) - K * exp(-r * expir)*norm.cdf(d2)
        return option_price
    if Option_type == "put":
        option_price = K * exp(-r * expir) * norm.cdf(-d2) - S *exp(- div*expir)* norm.cdf(-d1)
        return option_price



Option_type = "call"
S = 100
K = 100
T = 1
r = 0.06
div = 0.03
N = 1000
dx = 0.2
vol =0.25
stability = dx >= vol* np.sqrt(3*(T/N))
print(f'dx is 0.2, N is 500 and stability condition is {stability}')
print(f'European trinomial call: {trinomial_tree_European_dividend(Option_type,S,K,r,vol,T,N,div,dx)}')
print(f'American trinomial put: {trinomial_tree_American_dividend(Option_type,S,K,r,vol,T,N,div,dx)}')
print(f'BSM: {BSM_dividend(Option_type,S,K,vol,div,T,r)}')

## see the error decreases when the number of steps increases
import matplotlib.pyplot as plt
%matplotlib inline 
Option_type =="put"
S = 100
K = 100
T = 1
r = 0.06
vol =0.2
div = 0.03
dx = 0.25
stability = dx >= vol* np.sqrt(3*(T/10))

print(f'Stability condition is {stability}')

error = []
x = [10,20,30,30,50,100,150,200,250,300,350,400]
for N in x :
    dx = vol*np.sqrt(3*(T/N))
    error.append(abs(trinomial_tree_European_dividend(Option_type,S,K,r,vol,T,N,div,dx) - BSM_dividend(Option_type,S,K,vol,div,T,r)))
   
print(error)
plt.plot(x,error)


##

