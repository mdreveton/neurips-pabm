#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 12:03:41 2024
"""

import networkx as nx
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from tqdm import tqdm


import utils as utils
import clustering as clustering

SIZE_TITLE = 24
SIZE_LABELS = 24
SIZE_TICKS = 20
SIZE_LEGEND = 18


"""
# =============================================================================
# HOMOGENEOUS SCENARIOS
# =============================================================================

###### VARYING C

scenario = 'homo-uniform-uniform'
scenario = 'homo-uniform-one'

n = 900
n_clusters = 3
sizes = [ n//n_clusters for dummy in range( n_clusters ) ]
edge_density = 0.05
nAverage = 10

xi = 1
c_range = [ 0,0.2,0.4,0.6,0.8,1 ]

results_mean, results_std = runScenario_c( scenario, n, sizes, edge_density, xi, c_range, nAverage = nAverage, metric = 'accuracy' )

fileName = str(scenario) + '_c_n_' + str( n ) + '_k_' + str(n_clusters) + '_density_' + str(edge_density) + '_xi_' + str(xi) + '_nAverage' + str(nAverage) + '.pdf'
plotFigure( c_range, results_mean, accuracy_err = results_std, methods = [ 'sbm', 'dcbm', 'pabm', 'osc'], 
               xticks = None, yticks = None,
               xlabel = "c", ylabel = "Accuracy",
               savefig = False, fileName = fileName )


###### VARYING XI

scenario = 'exponential'
scenario = 'homo-uniform-one'


n = 900
n_clusters = 3
sizes = [ n//n_clusters for dummy in range( n_clusters ) ]
edge_density = 0.05
nAverage = 10

xi_range = [ 0, 0.2, 0.4, 0.6, 0.8, 1 ]
c = 0.8

results_mean, results_std = runScenario_xi( scenario, n, sizes, edge_density, xi_range, c, nAverage = 10, metric = 'accuracy' , homogeneous = True )

fileName = str(scenario) + '_xi' + '_n_' + str(n) + '_k_' + str(n_clusters) + '_density_' + str(edge_density) + '_nAverage_' + str(nAverage) + '.pdf'
plotFigure( xi_range, results_mean, accuracy_err = results_std, methods = [ 'sbm', 'dcbm', 'pabm', 'osc'], 
               xticks = None, yticks = None,
               xlabel = r"$xi$", ylabel = "Accuracy",
               savefig = False, fileName = fileName )



# =============================================================================
# HETEROGENEOUS SCENARIOS
# =============================================================================

scenario = 'exponential'

n = 2000
n_clusters = 5
sizes = [ n//n_clusters for dummy in range( n_clusters ) ]
edge_density = 0.05
nAverage = 10

n = 2000
n_clusters = 5
edge_density = 0.05

xi_range = [ 0,0.2,0.4,0.6,0.8,1 ]
xi_range = np.linspace(0, 1 , 11)
c = 0.5

results_mean, results_std = runScenario_xi( scenario, n, sizes, edge_density, xi_range, c, nAverage = nAverage, metric = 'accuracy' , homogeneous = False )
    
fileName = 'hetero_' + str(scenario) + '_xi_n_' + str(n) + '_k_' + str(n_clusters) + '_density_' + str(edge_density) + '_nAverage' + str(nAverage) + '.pdf'
plotFigure( xi_range, results_mean, accuracy_err = results_std, methods = [ 'sbm', 'dcbm', 'pabm', 'osc'], 
               xticks = None, yticks = None,
               xlabel = r'$\xi$', ylabel = "Accuracy",
               savefig = False, fileName = fileName )


"""



# =============================================================================
# CODE TO OBTAIN RESULTS OF THE DIFFERENT SCENARIOS
# =============================================================================


def runScenario_c( scenario, n, sizes, edge_density, community_strength, c_range, nAverage = 10, metric = 'accuracy' ):
    
    results_mean = { 'sbm' : [ ], 'dcbm' : [ ], 'pabm' : [ ], 'osc' : [ ], 'sklearn' : [] }
    results_std = { 'sbm' : [ ], 'dcbm' : [ ], 'pabm' : [ ], 'osc' : [ ], 'sklearn' : [] }

    algos = results_mean.keys( )
    
    for dummy in tqdm( range( len( c_range ) ) ):
        c = c_range[ dummy ]
        results_given_c = scenario_homogeneous( scenario, n, sizes, edge_density, community_strength, c = c, nAverage = nAverage, metric = metric )
        
        for algo in algos:
            results_mean[ algo ].append( np.mean( results_given_c[ algo ] ) )
            results_std[ algo ].append( np.std( results_given_c[ algo ] ) )
    
    return results_mean, results_std


def runScenario_xi( scenario, n, sizes, edge_density, community_strength_range, c, nAverage = 10, metric = 'accuracy' , homogeneous = True ):
    
    results_mean = { 'sbm' : [ ], 'dcbm' : [ ], 'pabm' : [ ], 'osc' : [ ], 'sklearn' : [] }
    results_std = { 'sbm' : [ ], 'dcbm' : [ ], 'pabm' : [ ], 'osc' : [ ], 'sklearn' : [] }

    algos = results_mean.keys( )
    
    for dummy in tqdm( range( len( community_strength_range ) ) ):
        community_strength = community_strength_range[ dummy ]
        
        if homogeneous == True:
            results_given_xi = scenario_homogeneous( scenario, n, sizes, edge_density, community_strength, c = c, nAverage = nAverage, metric = metric )
        
        else:
            results_given_xi = scenario_inhomogeneous( scenario, n, sizes, edge_density, community_strength, sigma = c, nAverage = nAverage, metric = metric )
            
        for algo in algos:
            results_mean[ algo ].append( np.mean( results_given_xi[ algo ] ) )
            results_std[ algo ].append( np.std( results_given_xi[ algo ] ) )
    
    return results_mean, results_std



def scenario_homogeneous( scenario, n, sizes, edge_density, xi, c = 1, nAverage = 10, metric = 'accuracy'):
    
    results = { 'sbm' : [ ], 'dcbm' : [ ], 'pabm' : [ ], 'osc' : [ ], 'sklearn' : [ ] }
    
    algos = results.keys( ) 
    
    n_clusters = len(sizes)
    labels_true = [ ]
    for community in range( n_clusters ):
        labels_true = labels_true + [ community+1 for i in range( sizes[community] ) ]
    labels_true = np.array( labels_true, dtype = int )

    for run in range( nAverage ):
        if scenario == 'homo-uniform-uniform':
            theta_in = np.random.uniform( 1-c, 1 + c, size = n )
            #theta_out = theta_in
            #theta_out = np.ones( n )
            theta_out = np.random.uniform( 1-c, 1 + c, size = n )
        
        elif scenario == 'homo-uniform-one':
            theta_in = np.random.uniform( 1-c, 1 + c, size = n )
            theta_out = np.ones( n )
        
        elif scenario == 2:
            theta_in = 2 * np.random.beta( 2, 2, size = n )
            theta_out = 2 * np.random.beta( 2, 2, size = n )
            
        elif scenario == 3:
            #theta_in = 2 * np.random.beta( 2, 2, size = n )
            theta_in = sp.stats.truncpareto.rvs( 1.5, 100, loc = 0, scale = 1, size = n )
            theta_in /= np.mean( theta_in )
            theta_out = theta_in * ( np.ones( n ) +  np.random.uniform( -c, c, size = n ) )

        elif scenario == 'pareto':
            mean = np.mean ( truncatedDistributions( 'pareto', parameter = 1.5, min_value = 0.05, max_value = 50, size = 1000 ) )
            theta_in = truncatedDistributions( 'pareto', parameter = 1.5, min_value = 0.05, max_value = 50, size = n ) / mean
            theta_out = np.ones( n )
            
        elif scenario == 'lognormal':
            mean = np.mean ( truncatedDistributions( 'lognormal', parameter = 1, min_value = 0.05, max_value = 5, size = 1000 ) )
            theta_in = truncatedDistributions( 'lognormal', parameter = 1, min_value = 0.05, max_value = 5, size = n ) / mean
            theta_out = np.ones( n )
        
        elif scenario == 'exponential':
            mean = np.mean ( truncatedDistributions( 'exponential', parameter = 1, min_value = 0.05, max_value = 10, size = 1000 ) )
            theta_in = truncatedDistributions( 'exponential', parameter = 1, min_value = 0.1, max_value = 10, size = n ) / mean
            theta_out = np.ones( n )
        
        else:
            raise TypeError( 'Please provide a scenario implemented' )
        
        P = utils.generateP_of_homogeneousPABM( labels_true, np.sqrt( edge_density ), np.sqrt( edge_density * xi ), theta_in, theta_out )
        A = utils.generateBernoulliAdjacency( P )
        
        predicted_clusterings = getClusterings( A, n_clusters )
        
        for algo in algos:
            results[ algo ].append( utils.computePartitionMetric( labels_true, predicted_clusterings[ algo ], metric = metric ) )
        
    results_mean, results_std = dict( ), dict( )

    for algo in algos:
        results_mean[ algo ] = np.mean( results[ algo ] )
        results_std[ algo ] = np.std( results[ algo ] )
    
    return results 



def scenario_inhomogeneous( scenario, n, sizes, edge_density, xi, sigma = 1, nAverage = 10, metric = 'accuracy', min_value = 0.1, max_value = 10 ):
    
    results = { 'sbm' : [ ], 'dcbm' : [ ], 'pabm' : [ ], 'osc' : [ ], 'sklearn' : [ ] }
    
    algos = results.keys( ) 
    
    n_clusters = len( sizes )
    labels_true = [ ]
    
    for community in range( n_clusters ):
        labels_true = labels_true + [ community+1 for i in range( sizes[community] ) ]
    labels_true = np.array( labels_true, dtype = int )

    for run in range( nAverage ):
        
        if scenario == 'pareto':
            Lambdas = [ [ ] for a in range( n_clusters ) ]
            mean = np.mean ( truncatedDistributions( 'pareto', parameter = 1.5, min_value = min_value, max_value = max_value, size = 1000 ) )
            for a in range( n_clusters ):
                #Lambdas[ a ] = [ np.random.lognormal( mean = 1, sigma = 1, size = sizes[ a ] ) for b in range(n_clusters) ]
                #Lambdas[ a ] = [ 1/2 * np.random.beta( 2, 2, size = sizes[ a ] ) for b in range( n_clusters ) ]   
                #mean = np.mean ( sp.stats.truncpareto.rvs( 1.5, 5, size = 1000 ) )
                #Lambdas[ a ] = [ sp.stats.truncpareto.rvs( 1.5, 5, size = sizes[ a ] ) / mean for b in range(n_clusters) ]
                mean = 1
                Lambdas[ a ] = [ truncatedDistributions( 'pareto', parameter = 1.5, min_value = min_value, max_value = max_value, size = sizes[ a ] ) / mean for b in range(n_clusters) ]

                
        if scenario == 'lognormal':
            Lambdas = [ [ ] for a in range( n_clusters ) ]
            mean = np.mean ( truncatedDistributions( 'lognormal', parameter = 1, min_value = min_value, max_value = max_value, size = 1000 ) )
            for a in range( n_clusters ):
                #Lambdas[ a ] = [ np.exp( -sigma**2 / 2) * np.random.lognormal( mean = 0, sigma = sigma, size = sizes[ a ] ) for b in range(n_clusters) ]
                mean = 1
                Lambdas[ a ] = [ truncatedDistributions( 'lognormal', parameter = 1, min_value = min_value, max_value = max_value, size = sizes[ a ] ) / mean for b in range(n_clusters) ]

        if scenario == 'exponential':
            Lambdas = [ [ ] for a in range( n_clusters ) ]
            #mean = np.mean ( sp.stats.truncexpon.rvs( 5, scale = 1, size = 1000 ) )
            mean = np.mean ( truncatedDistributions( 'exponential', parameter = 1, min_value = min_value, max_value = max_value, size = 1000 ) )
            for a in range( n_clusters ):
                #Lambdas[ a ] = [  sp.stats.truncexpon.rvs( 5, size = sizes[ a ] ) / mean for b in range(n_clusters) ]
                mean = 1
                Lambdas[ a ] = [ truncatedDistributions( 'exponential', parameter = 1, min_value = min_value, max_value = max_value, size = sizes[ a ] ) / mean for b in range(n_clusters) ]

        for a in range( n_clusters ):
            for b in range( n_clusters ):
                Lambdas[ a ][ b ] *= np.sqrt( edge_density )
                if a!= b:
                    Lambdas[ a ][ b ] *= np.sqrt( xi )
        
        P = utils.generateP_inhomogeneousPABM( sizes, Lambdas )
        A = utils.generateBernoulliAdjacency( P )
        
        predicted_clusterings = getClusterings( A, n_clusters )
        
        for algo in algos:
            results[ algo ].append( utils.computePartitionMetric( labels_true, predicted_clusterings[ algo ], metric = metric ) )
        
    results_mean, results_std = dict( ), dict( )

    for algo in algos:
        results_mean[ algo ] = np.mean( results[ algo ] )
        results_std[ algo ] = np.std( results[ algo ] )
    
    return results 






# =============================================================================
# CODE TO PLOT FIGURES
# =============================================================================

def plotFigure( x, accuracy_mean, accuracy_err = None, methods = None, 
               xticks = None, yticks = None,
               xlabel = "x", ylabel = "Accuracy",
               savefig = False, fileName = "fig.pdf" ):
    
    if methods is None and accuracy_err is None:
        plt.plot( x, accuracy_mean, linestyle = '-.', marker = '.' )
        
    elif methods is None and accuracy_err is not None:
        plt.errorbar( x, accuracy_mean, yerr = accuracy_err, linestyle = '-.' )
        
    elif methods is not None and accuracy_err is None:
        for method in methods:
            plt.plot( x, accuracy_mean[ method ], linestyle = '-.', marker = '.', label = method )
            legend = plt.legend( loc=0,  fancybox = True, fontsize = SIZE_LEGEND )
            plt.setp( legend.get_title(),fontsize = SIZE_LEGEND )
    
    elif methods is not None and accuracy_err is not None:
        for method in methods:
            plt.errorbar( x, accuracy_mean[ method ], yerr = accuracy_err[ method ], linestyle = '-.', label = method )
            legend = plt.legend( loc=0,  fancybox = True, fontsize = SIZE_LEGEND )
            plt.setp( legend.get_title(),fontsize = SIZE_LEGEND )

    plt.xlabel( xlabel, fontsize = SIZE_LABELS )
    plt.ylabel( ylabel, fontsize = SIZE_LABELS )
    
    if xticks != None:
        plt.xticks( xticks, fontsize = SIZE_TICKS )
    else:
        plt.xticks( fontsize = SIZE_TICKS )
    
    if yticks != None:
        plt.yticks( yticks, fontsize = SIZE_TICKS )
    else:
        plt.yticks( fontsize = SIZE_TICKS )

    if(savefig):
        plt.savefig( fileName, bbox_inches='tight' )
    plt.show( )



# =============================================================================
# CODE TO OBTAIN THE PREDICTED CLUSTERINGS AND CORRESPONDING METRICS (ACCURACY ETC.)
# =============================================================================


def getClusterings( A, n_clusters ):
    
    z_bm = clustering.spectralClustering_bm( A , n_clusters )
    z_dcbm = clustering.spectralClustering_dcbm( A , n_clusters )
    z_pabm = clustering.spectralClustering_pabm( A, n_clusters )
    z_osc = clustering.orthogonalSpectralClustering( A, n_clusters )
    z_sklearn = clustering.graph_clustering( A, n_clusters, variant = 'sklearn' )

    return { 'sbm' : z_bm, 'dcbm' : z_dcbm, 'pabm' : z_pabm, 'osc' : z_osc, 'sklearn' : z_sklearn }



def truncatedDistributions( distribution, parameter, min_value = 0, max_value = 10, size = 1, max_trials = 100 ):
    
    if distribution == 'exponential':
        samples = sp.stats.expon.rvs( scale = parameter, size = max_trials * size )
        
    elif distribution == 'pareto':
        samples = sp.stats.pareto.rvs( parameter, size = max_trials * size )
    
    elif distribution == 'lognormal':
        samples = sp.stats.lognorm.rvs( parameter, size = max_trials * size )
    
    samples = list( samples )
    next_admissible_element = 0
    result = [ ]

    while len( result ) < size and next_admissible_element != None:
        next_admissible_element = next( ( x for x in samples if x > min_value ), None )
        samples.remove( next_admissible_element )
        if next_admissible_element < max_value:
            result.append( next_admissible_element )
        else:
            result.append( max_value )
            
    if len( result ) == size:
            return np.asarray( result )
    else:
        raise TypeError( 'The function was not able to return a list of the given size within a reasonable number of tries')
            
    

