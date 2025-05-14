#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 13:15:23 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from tqdm import tqdm 

SIZE_TITLE = 24
SIZE_LABELS = 24
SIZE_TICKS = 20
SIZE_LEGEND = 18


"""


# =============================================================================
# CASE UNIFORM & ONES: VARYING XI
# =============================================================================
c = 0.8

n = 900
rho = 10 * np.log( n ) / n
rho = 0.05
k = 3
fileName = 'optimal_rate_homo_uniform_one_n_' + str( n ) + '_k_' + str(k) + '_rho_' + str(rho) + '_c_' + str(c) + '.pdf'
savefig = False

xi_range = np.linspace(0.01, 1 , 100)
chernoff = [ ]
for _ in tqdm( range( len(xi_range ) ) ):
    xi = xi_range[ _ ]
    chernoff.append( compute_chernoff_numerically( n, k, rho, xi, in_law = 'uniform', out_law = 'uniform', in_parameter = c, out_parameter = c, method = 'elaine' ) )
    
    #chernoff.append( compute_chernoff_via_formula( n, k, rho, xi, in_law = 'uniform', out_law = 'one', in_parameter = c, out_parameter = c ) )

plt.plot( xi_range, chernoff )
plt.xlabel( r'xi$', fontsize = SIZE_LABELS )
plt.ylabel( 'Optimal error rate', fontsize = SIZE_LABELS )
plt.xticks( fontsize = SIZE_TICKS )
plt.yticks( fontsize = SIZE_TICKS )
if savefig:
    plt.savefig( fileName, bbox_inches='tight' )
plt.show( )



# =============================================================================
# CASE UNIFORM & ONES: VARYING C
# =============================================================================
n = 900
rho = 10 * np.log( n ) / n
rho = 0.05
k = 3
xi = 1
fileName = 'optimal_rate_homo_uniform_one_n_' + str( n ) + '_k_' + str(k) + '_rho_' + str(rho) + '_xi_' + str(xi) + '.pdf'
savefig = False

c_range = np.linspace(0.01, 1 , 100)
chernoff = [ ]
for _ in tqdm( range( len(c_range ) ) ):
    c = c_range[ _ ]
    chernoff.append( compute_chernoff_numerically( n, k, rho, xi, in_law = 'uniform', out_law = 'one', in_parameter = c, out_parameter = c, method = 'elaine' ) )
    
    #chernoff.append( compute_chernoff_via_formula( n, k, rho, xi, in_law = 'uniform', out_law = 'one', in_parameter = c, out_parameter = c ) )


plt.plot( c_range, chernoff )
plt.xlabel( r'$c$', fontsize = SIZE_LABELS )
plt.ylabel( 'Optimal error rate', fontsize = SIZE_LABELS )
plt.xticks( fontsize = SIZE_TICKS )
plt.yticks( fontsize = SIZE_TICKS )
if savefig:
    plt.savefig( fileName, bbox_inches='tight' )
plt.show( )




# =============================================================================
# OTHER CASES: VARYING XI
# =============================================================================

scenario = 'exponential'
in_parameter = 1
out_parameter = 1 

n = 900
rho = 10 * np.log( n ) / n
rho = 0.05
k = 3
xi = 1
fileName = 'optimal_rate_' + scenario + '_n_' + str( n ) + '_k_' + str(k) + '_rho_' + str(rho) + '.pdf'
savefig = False

xi_range = np.linspace(0.01, 1 , 100)
chernoff = [ ]
for _ in tqdm( range( len(xi_range ) ) ):
    xi = xi_range[ _ ]
    chernoff.append( compute_chernoff_numerically( n, k, rho, xi, in_law = scenario, out_law = scenario, in_parameter = in_parameter, out_parameter = out_parameter, method = 'elaine' ) )
    

plt.plot( xi_range, chernoff )
plt.xlabel( r'$xi$', fontsize = SIZE_LABELS )
plt.ylabel( 'Optimal error rate', fontsize = SIZE_LABELS )
plt.xticks( fontsize = SIZE_TICKS )
plt.yticks( fontsize = SIZE_TICKS )
if savefig:
    plt.savefig( fileName, bbox_inches='tight' )
plt.show( )


#compute_chernoff_numerically( n, k, rho, xi, in_law = 'uniform', out_law = 'one', in_parameter = 1, out_parameter = 1 )
#compute_chernoff_via_formula( n, k, rho, xi, in_law = 'uniform', out_law = 'one', in_parameter = 1, out_parameter = 1 )
"""

      

def compute_chernoff_numerically( n, k, rho, xi, in_law = 'uniform', out_law = 'uniform', in_parameter = 1, out_parameter = 1, method = 'integral' ):
    coeff = n * rho / k
    gamma_in = gamma( in_parameter, law = in_law )
    gamma_out = gamma( out_parameter, law = out_law )
    
    if out_law == 'one':
        
        delta = lambda x : x + xi - 2 * gamma_in * np.sqrt( xi) * np.sqrt( x )
        
        if in_law == 'uniform':
            
            if method == 'integral':
                func = lambda x : np.exp( - coeff * delta(x) ) / (2*in_parameter)
                res, err = sp.integrate.quad( func , 1-in_parameter, 1+in_parameter ) 
            
            elif method.lower() == 'laplace':
                res = sp.optimize.minimize_scalar( delta, bounds=( (1-in_parameter, 1+in_parameter) ) )
                
                if res.x >= 1+in_parameter - 1e-5:
                    print( 'The maximum is greater than the integration bound: could be a problem. Xi : ', xi )
                
                if res.x <= 1-in_parameter + 1e-5:
                    print( 'The maximum is lower than the integration bound: could be a problem. Xi : ', xi )
                
                #print ( in_parameter, (1+in_parameter, res.x - (1-in_parameter)), res.x )
                    
                res = np.exp( - coeff * res.fun ) / (2*in_parameter)
                res = res * np.sqrt( 2 * np.pi / ( coeff * np.abs( 1 / ( 2*xi*gamma_in**2 ) ) ) )
        
            elif method == 'elaine':
                M = n * rho / k 
                u1 = np.sqrt( M ) * ( np.sqrt(1-in_parameter) - gamma_in * np.sqrt(xi) )
                u2 = np.sqrt( M ) * ( np.sqrt(1+in_parameter) - gamma_in * np.sqrt(xi) )
                J1 = np.exp( - M * xi * ( 1 - gamma_in**2 ) ) / ( 2 * in_parameter * M ) 
                J2 =  np.exp( - u2**2 ) - np.exp( - u1**2 ) + gamma_in * np.sqrt(xi * M * np.pi ) * ( sp.special.erf(u2) - sp.special.erf(u1) )
                res = J1 * J2
        
    else:
        
        delta = lambda x : x[0] + xi * x[1] - 2 * gamma_in * gamma_out * np.sqrt( xi) * np.sqrt( x[0] * x[1] )
        
        if in_law == 'uniform' and out_law == 'uniform':
            func = lambda x0, x1 : np.exp( - coeff * delta( [x0, x1] ) ) / ( 2 * in_parameter ) / ( 2 * out_parameter )
            res, err = sp.integrate.nquad( func , [ [1-in_parameter, 1+in_parameter], [1-out_parameter, 1+out_parameter] ] )
            
        elif in_law == 'exponential' and out_law == 'exponential':
            func = lambda x0, x1 : np.exp( - coeff * delta( [x0, x1] ) ) * in_parameter * out_parameter * np.exp( - in_parameter * x0 ) * np.exp( - out_parameter * x1 )
            res, err = sp.integrate.nquad( func , [ [0, np.inf], [0, np.inf] ] )
        
        elif in_law == 'lognormal' and out_law == 'lognormal':
            pdf_in = lambda x : sp.stats.lognorm.pdf(x, s = in_parameter, scale = np.exp(-in_parameter**2/2 ) )
            pdf_out = lambda x : sp.stats.lognorm.pdf(x, s = out_parameter, scale = np.exp(-out_parameter**2/2 ) )

            func = lambda x0, x1 : np.exp( - coeff * delta( [x0, x1] ) ) * pdf_in( x0 ) * pdf_out( x1 ) 
            res, err = sp.integrate.nquad( func , [ [0, np.inf], [0, np.inf] ] )
                
    #print( 'The error is ', err )
    return res


    

def compute_chernoff_via_formula( n, k, rho, xi, in_law = 'uniform', out_law = 'one', in_parameter = 1, out_parameter = 1 ):
    
    coeff = n * rho / k
    gamma_in = gamma( in_parameter, law = in_law )
    
    if out_law == 'one':
        if in_law == 'uniform':
            return np.exp( -coeff * xi * (1-gamma_in**2) ) / ( 2*in_parameter )
        
    else:
        raise TypeError( 'This probability distribution is not implemented' )



def gamma( parameter, law = 'uniform' ):
    if law == 'uniform':
        #func = lambda x : np.sqrt(x) * 1/(2*parameter)
        #res, err = sp.integrate.quad( func, 1-parameter, 1+parameter )
        return 1/( 3 * parameter ) * ( ( 1+parameter )**(3/2) - ( 1-parameter )**(3/2) )
    
    elif law == 'exponential':
        func = lambda x : np.sqrt(x) * parameter * np.exp( - parameter * x )
        res, err = sp.integrate.quad( func, 0, np.inf )
        return res 
    
    elif law == 'one' or law == 'ones':
        return 1
    
    elif law == 'lognormal':
        pdf = lambda x : sp.stats.lognorm.pdf(x, s = parameter, scale = np.exp(-parameter**2/2 ) )
        func = lambda x : np.sqrt(x) * pdf( x )
        res, err = sp.integrate.quad( func, 0, np.inf )
        return res 
    
    else:
        raise TypeError( 'This probability distribution is not implemented' )



def f_uniform_uniform( x , y, xi, c ):
    gamma_c = gamma(c)
    return x + xi * y - 2 * gamma_c**2 * np.sqrt(xi) * np.sqrt(x * y) 

def f_uniform_one( x, xi, c ):
    gamma_c = gamma(c)
    return x + xi - 2 * gamma_c * np.sqrt(xi) * np.sqrt(x) 


def compute_chernoff( xi_range, c = 0.5, model = 'uniform_uniform' ):
    
    results = [ ]
    
    for xi in xi_range:
        if model == 'uniform_uniform':
            
            fun = lambda var: f_uniform_uniform( var[0], var[1], xi = xi, c = c )
            res = sp.optimize.minimize( fun, x0=[0.5,0.5], bounds=((1-c, 1+c), (1-c,1+c)) )
            
        elif model == 'uniform_one':
            fun = lambda var: f_uniform_one(var, xi = xi, c = c )
            res = sp.optimize.minimize_scalar( fun, bounds=( (1-c, 1+c) ) )
            
        results.append( res.fun )
        
    return results


def compute_chernoff_theoretic( xi_range, c = 0.5, model = 'uniform_one' ):
    gamma_c = gamma(c)
    return [ xi * (1-gamma_c**2) for xi in xi_range ]


"""
# Create grid
x = np.linspace(1-c, 1+c , 400)
y = np.linspace(1-c, 1+c, 400)
X, Y = np.meshgrid(x, y)
Z = f_uniform_uniform( X, Y, xi, c )

# Plot
plt.figure(figsize=(8, 6))
contours = plt.contour(X, Y, Z, levels=50, cmap='viridis')
plt.clabel(contours, inline=True, fontsize=8)
plt.colorbar()

# Mark the minimum
fun = lambda var: f_uniform_uniform(var[0], var[1], xi = xi, c = c )
res = sp.optimize.minimize( fun, x0=[0.5,0.5], bounds=((1-c, 1+c), (1-c,1+c)) )

min_x, min_y = res.x[0], res.x[1]
plt.plot(min_x, min_y, 'ro')  # red dot at the minimum
plt.text(min_x + 0.5, min_y, 'Minimum', color='red')

plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Contour plot with minimum located')
plt.grid(True)
plt.show()


"""