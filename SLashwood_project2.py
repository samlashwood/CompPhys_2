# -*- coding: utf-8 -*-
"""
@author: sl4516

The following code is part of a project undertaken in numerical integration.

The code below was used to produce all of the results used in my report, with the accuracies changed in some places.
"""

import numpy as np
import scipy as sp
import copy
import matplotlib.pyplot as plt

np.random.seed(1)                                                              #Seed random distributions for reproducability

def extended_trap_rule(f,a,b,epsilon):                                         #Extended Trapezium rule function, taking integrand fucntion, upper and lower limits, and accuracy as arguements
    
    i, fns=2,3                                                                 #i is the current number of rectangle sections, fns the # of funct. evals
    
    I1=(b-a)*0.5*(f(a)+f(b))                                                   #Compute first 2 estimations to comapre for given accuracy
    I2=0.5*I1+(b-a)*0.5*f((a+b)/2.)
    
    while np.abs((I2-I1)/I1)>=epsilon:                                         #Continue until user specified accuracy is reached (relative error comparison to achieve this)
        I1=copy.copy(I2)                                                       #Update for comparison
        
        I2=0.5*copy.copy(I1)                                                   #Step size halved
        h=((b-a)/(2**i))                                                       #New width
        for n in [x for x in range(2**(i-1))]:                                 #For all terms that must be added
            I2+=h*f((a+(2*n+1)*h))                                             #Add in 'middle' terms for each existing integral, hence 2n+1
        
        fns+=2**(i-1)                                                          #Counting the number of function evaluations
        i+=1
    
    #err=-(1/(12*(2**i)))*((b-a)**3)*(sp.misc.derivative(f,b)-sp.misc.derivative(f,a))                #Error estimate on method, if Scipy works
    return I2, fns#, np.abs(err)                                                #Returning the estimate, number of function evaluations and maybe error if Scipy works!


def extended_simpsons_rule(f,a,b,epsilon):                                     #Extended Simpson's rule function, taking integrand fucntion, upper and lower limits, and accuracy as arguements
    
    i, fns=3,5                                                                 #i is the current number of rectangle sections, fns the # of funct. evals
    
    I0=(b-a)*0.5*(f(a)+f(b))                                                   #Compute first 3 estimations to comapre for given accuracy
    I1=0.5*I0+(b-a)*0.5*f((a+b)/2.)
    I2=0.5*I1+(b-a)*0.25*(f(a+((b-a)*0.25))+f(a+((b-a)*0.75)))
    
    S1=(4*I1/3.)-(I0/3.)                                                       #Use first 3 trap. rule estimations to obtian first 2 Simpson's estimates to comapre for given accuracy 
    S2=(4*I2/3.)-(I1/3.)
    
    while np.abs((S2-S1)/S1)>=epsilon:                                         #Continue until user specified accuracy is reached (relative error comparison to achieve this)
        I1=copy.copy(I2)                                                       #Trap. rule implementation as above
        
        I2=0.5*copy.copy(I1)
        h=((b-a)/(2**i))
        for n in [x for x in range(2**(i-1))]:
            I2+=h*f((a+(2*n+1)*h))
        
        S1=copy.copy(S2)
        S2=(4*I2/3.)-(I1/3.)                                                   #Update Simpson's estimate for comparison
        
        fns+=2**(i-1)
        i+=1
    
    #err=((b-a)/180.)*(((b-a)/(2**i))**4)*sp.misc.derivative(f,a,n=4, order=5)  #Error estimate on method, if Scipy works
    return S2, fns#, np.abs(err)                                                #Returning the estimate, number of function evaluations and maybe error if Scipy works!


def psi_sq(z):                                                                 #The given function to estimate, with the square removing complex exponential phase element
    return np.abs(((np.pi)**-0.25)*np.exp(0.5*(-z**2)))**2

print("Extended Trap. rule: ", extended_trap_rule(psi_sq,0,2,10**-6))          #Estimates of given integral for trap. and Simpson's rules.
print("Extended Simpson's rule: ", extended_simpsons_rule(psi_sq,0,2,10**-6))


def monte_carlo_int(f,a,b,epsilon,deviates,PDF):                               #Implement generalised Monte Carlo method to integrate function with same arguements as before AND functions for importance sampling (devaites from transformation method and weighting PDF)
    N=100
    xcoords=list(deviates(np.random.uniform(a,b,100)))                         #Start with 100 points considered which are deviates with the deviates function
    sums, sum_prev, errlist=0,1,[]
    
    for i in range(len(xcoords)):
        temps1=f((xcoords[i]))/PDF(xcoords[i])                                 #Add first set of points' value of function(point)/ weight_PDF(point) to sum and list for error 
        sums+=temps1
        errlist.append(temps1)
    
    while np.abs((sums-sum_prev)/sum_prev)>=epsilon or sum_prev==0:            #Continue until user specified accuracy is reached in sums (relative error comparison to achieve this)
        sum_prev=copy.copy(sums)                                               #Accuracy of estimate can be specified above as the 4th function arguement, left as 10**-3
        
        xcoords1=list(deviates(np.random.uniform(a,b,100)))
        xcoords+=xcoords1
        
        for i in range(len(xcoords1)):                                         #Repeat previous step of point sums for another 100 points 
            temps1=f((xcoords1[i]))/PDF(xcoords1[i])
            sums+=temps1
            errlist.append(temps1)
        
        N+=100
    
    finalsum=(sums/(N))                                                        #Monte Carlo estimate
    errors=[(x-finalsum)**2 for x in errlist]
    error=np.sqrt(sum(errors)/(N-1))/N                                         #Calculation of error on estimate
    return finalsum,N,xcoords,error                                            #Return estimate, Number of function evaluations, deviates considered (for importance sampling validation) and error 


def monte_carlo_int_metropolis(f,a,b,epsilon):                                 #Metropolis Monte Carlo with adaptive step sizing, taking number of times to conisder stepping as 'steps' argument (and same other arguments)
    sumstot, errtot=[], []
    
    for xcoord in np.random.uniform(a,b,10):
        N, sums, sum_prev, xcoordall, accepts, errlist=0,0,1,[],[],[]          #Initialise variables annd lists used: N=# of funct. evaluations, sums and sum1= 2 consecutive MC totals, then list of all coordinates, accepted coordinates and function values  
        xcoord=np.random.uniform(a,b)                                          #Pick a random point in the domain
        xcoordall.append(xcoord)                                               #Add to list to keep track of all locations considered
        
        point=xcoord                                                           #Initialise 'point' as current point
        
        np.random.seed(10)                                                     #Re-seed generator for probability comparisons- reproducability but not correlated to previous number
        while sum_prev==0 or np.abs((sums-sum_prev)/sum_prev)>=epsilon:        #Accuracy of estimate can be specified above as the 4th function arguement, left as 10**-3                                                 
            
            xcoordall.append(point)                                            #Process current point by adding function value to sum and point to lists for error and all points
            sum_prev=copy.copy(sums)
            
            temp=compare(f,point,a,b)                                          #Call compare function (see function)
            sums+=np.abs(temp[1])
            errlist.append(temp[1])
            
            N+=1                                                               #Counting number of steps taken
            if temp[2]>np.random.uniform(0,1):                                 #Generate random number to examine whether to move given probability of moving
                point=temp[0]                                                  #If move to be taken, update current point
                accepts.append(point)
        
        finalsum=(sums/(N-1))
        errors=[(x-finalsum)**2 for x in errlist]                              #Calculate final sum and errors
        error=np.sqrt(sum(errors)/(N-1))/N
        sumstot.append(finalsum)
        errtot.append(error)
    
    finalsum1=np.mean(sumstot)
    finalerror=np.mean(errtot)
    return finalsum1,N,xcoordall,finalerror,accepts                            #Return estimate, number of function evaluations, points evaluated for validation and error


def compare(f,x,a,b):                                                          #Function for metroplois algorithm that returns a step from the current positon and the probability of taking that step
    p=x+np.random.normal()*0.1                                                 #Small step away from given point
    if p<a or p>b:
        return x,f(x),0                                                        #If step takes point out of domain, probability of taking step is 0                                                            
    else:
        return p,f(x)/Linear_dev(x),(Linear_dev(p)/Linear_dev(x))              #Return the new point, the weighted function evaluated at the previous point and the probability of stepping to new point


def Flat_dev(x):                                                               #Weighting function for flat importance sampling
    return x 
def Flat_trans(x):                                                             #Uniform normalised transformation function for flat sampling
    return 0.5
def Linear_trans(a):
    x=a/2.                                                                     #Devaites transformation for linear importance sampling. Divide by 2 first such that we transform from [0,1}.
    return (2*0.98-np.sqrt(4*0.98**2-8*0.48*(x)))/(2*0.48)
def Linear_dev(x):                                                             #Linear weighting function
    return (0.98-0.48*x)


def monte_carlo_int_alternate(f,a,b,epsilon):                                  #Define function to estimate integral using area comaprison- alternative (rejection) method
    V=1.2                                                                      #Area of 2*0.6 grid
    N=100
    I1=copy.copy(V)
    I2=0.5*I1+(b-a)*0.5*f((a+b)/2.)
    
    xcoords=list(np.random.uniform(a,b,100))
    ycoords=list([0.6*x for x in np.random.uniform(0,1,100)])                  #Generate x,y coordinates in 2*0.6 box. [below] Accept all points 
    
    accepts=[[xcoords[n],ycoords[n]] for n in range(len(xcoords)) if f(xcoords[n])>=ycoords[n] and np.abs(f(xcoords[n]))>=np.abs(ycoords[n])]
    
    while np.abs((I2-I1)/I1)>=epsilon:                                         #Compare such that user specified accuracy is reached
        I1=copy.copy(I2)                                                       #Accuracy of estimate can be specified above as the 4th function argument, left as 10**-3
        xcoords1=list(np.random.uniform(a,b,100))                              #Repeat previous steps to consider more random coordinates
        ycoords1=list([0.6*x for x in np.random.uniform(0,1,100)])
        
        xcoords+=xcoords1
        ycoords+=ycoords1
        accepts+=[[xcoords1[n],ycoords1[n]] for n in range(len(xcoords1)) if f(xcoords1[n])>=np.abs(f(xcoords1[n])-ycoords1[n]) and np.abs(f(xcoords1[n]))>=np.abs(ycoords1[n])]
        
        I2=len(accepts)*(V/N)                                                  #Calculate estimate
        N+=100
    
    err=np.sqrt((I2/V)*(1-(I2/V))/N)
    return I2,N,accepts,xcoords,ycoords,err                                    #Return estimate, number of function evaluations, accpeted coordinates and all coordinates


lisa=monte_carlo_int(psi_sq,0,2,10**-3,Flat_dev,Flat_trans)                    
print("Flat importance sampling: ",lisa[0],lisa[3],lisa[1])                    #Run monte carlo method with flat importance sampling


fig1, ax1 = plt.subplots()
lisa1=monte_carlo_int(psi_sq,0,2,10**-3,Linear_trans,Linear_dev)               #Run monte carlo method with linear importance sampling and produce histogram of considered deviates
print("Linear importance sampling: ",lisa1[0],lisa1[3],lisa1[1])               #Accuracy of estimate can be specified in above line as the 4th function arguement, left as 10**-3

plt.hist(lisa1[2],100, density=1)
plt.plot(np.arange(0,2,0.01),[Linear_dev(a) for a in np.arange(0,2,0.01)], label="Linear weighting")
plt.title("Distribution of random deviates for linear importance sampling")
plt.legend(loc='upper right',prop={'size': 9})


fig1, ax1 = plt.subplots()
lisa2=monte_carlo_int_metropolis(psi_sq,0,2,10**-6)                            #Run monte carlo method with adaptive step sizing and produce histogram of considered deviates
print("Metropolis algorithm: ",lisa2[0],lisa2[3],lisa2[1])                     #Specify number of steps to take, not accuracy, as 4th function argument

plt.hist(lisa2[2],100, density=1)
plt.plot(np.arange(0,2,0.01),[Linear_dev(a) for a in np.arange(0,2,0.01)], label="Linear weighting")
plt.plot(np.arange(0,2,0.01),[np.real(psi_sq(a)) for a in np.arange(0,2,0.01)], label="Integrand function")
plt.title("Distribution of points considered in Metropolis algorithm steps")
plt.legend(loc='upper right',prop={'size': 9})


#adaptive=[monte_carlo_int_metropolis(psi_sq,0,2,10**-6)[0] for i in range(10)] #Run metropolis algorithm multiple times to obtain average
#print("Mean of metropolis iterations: ", np.mean(adaptive))


fig1, ax1 = plt.subplots()
lisaalt=monte_carlo_int_alternate(psi_sq,0,2,10**-3)                           #Run alternative area comparison monte carlo method and plot points under the integrand
xs=[aa[0] for aa in lisaalt[2]]
ys=[aa[1] for aa in lisaalt[2]]
print("Alternative (rejection) method: ", lisaalt[0],lisaalt[1])

plt.plot(xs,ys,"r+")
#plt.plot(lisa[3],lisa[4],"b+")                                                 #Plot of all points considered
plt.plot(np.arange(0,2,0.01),[np.real(psi_sq(a)) for a in np.arange(0,2,0.01)], label="Integrand function")
plt.title("Accepted points from alternate Monte Carlo method")
plt.legend(loc='upper right',prop={'size': 9})

#print(sp.integrate.quad(psi_sq,0,2))                                          #Validate results with given Scipy result

print("==========EXAMINE DIFFERENT ACCURACIES=========")
metnos=[]

for k in [10**(-1*a) for a in range(2,6)]:                                     #Examine each estimate at differing accuracies
    print("accuracy: ", k)
    
    func1=monte_carlo_int(psi_sq,0,2,k,Flat_dev,Flat_trans)
    func2=monte_carlo_int(psi_sq,0,2,k,Linear_trans,Linear_dev)
    func3=monte_carlo_int_metropolis(psi_sq,0,2,k)
    
    print("Flat importance sampling: ",func1[0],func1[3],func1[1])
    print("Linear importance sampling: ",func2[0],func2[3],func2[1])
    print("Metropolis algorithm: ",func3[0],func3[3],func3[1])
    print("")
    metnos.append(func3[1])
    

fig1, ax1 = plt.subplots()
plt.plot(range(2,6),[np.log(x) for x in metnos])                               #Plot the number of evaluations against accuracy
plt.title("Variation of number of function evaluations with accuracy")
plt.xlabel("Accuracy exponent (10^-x)")
plt.ylabel("ln(number of function evaluations)")
