#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 12:03:41 2024
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans, SpectralClustering


import utils as utils
import clustering as clustering
import thresholdedCosineSpectralClustering as tcsc
import greedySubspaceProjectionClustering as gspc

import selfrepresentation as selfrepresentation


SIZE_TITLE = 24
SIZE_LABELS = 24
SIZE_TICKS = 20
SIZE_LEGEND = 18

algorithms = ['sbm', 'dcbm', 'pabm', 'osc', 'tcsc', 'rtcsc', 'gspc', 'sklearn' ]

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
nAverage = 2

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

results_mean, results_std = runScenario_xi( scenario, n, sizes, edge_density, xi_range, c, nAverage = nAverage, metric = 'accuracy' , homogeneous = True )

fileName = str(scenario) + '_xi' + '_n_' + str(n) + '_k_' + str(n_clusters) + '_density_' + str(edge_density) + '_nAverage_' + str(nAverage) + '.pdf'
plotFigure( xi_range, results_mean, accuracy_err = results_std, methods = [ 'sbm', 'dcbm', 'pabm', 'osc'], 
               xticks = None, yticks = None,
               xlabel = r"$xi$", ylabel = "Accuracy",
               savefig = False, fileName = fileName )



# =============================================================================
# HETEROGENEOUS SCENARIOS
# =============================================================================

scenario = 'pareto' #choices: pareto, exponential, lognormal

n = 1600
n_clusters = 4
sizes = [ n//n_clusters for dummy in range( n_clusters ) ]
edge_density = 0.04
nAverage = 10

#xi_range = np.linspace(0, 1 , 11)
xi_range = [ 0,0.2,0.4,0.6,0.8,1 ]
xi_range[0] = 0.01 #To avoid problem with disconnected graphs
c = 0.5

results_mean, results_std = runScenario_xi( scenario, n, sizes, edge_density, xi_range, c, nAverage = nAverage, metric = 'accuracy' , homogeneous = False )
    
fileName = 'hetero_' + str(scenario) + '_xi_n_' + str(n) + '_k_' + str(n_clusters) + '_density_' + str(edge_density) + '_nAverage' + str(nAverage) + '.pdf'
plotFigure( xi_range, results_mean, accuracy_err = results_std, methods = [ 'sbm', 'dcbm', 'pabm', 'osc'], 
               xticks = None, yticks = None,
               xlabel = r'$xi$', ylabel = "Accuracy",
               savefig = False, fileName = fileName )



------------------------

scenario = 'pareto'

n = 2000
n_clusters = 5
sizes = [ n//n_clusters for dummy in range( n_clusters ) ]
edge_density = 0.02
nAverage = 2

xi = 1
shape_range = [ 0.5, 1, 1.5, 2, 2.5]

results_mean, results_std = runScenario_c( scenario, n, sizes, edge_density, xi, shape_range, nAverage = nAverage, metric = 'accuracy', homogeneous = False )

fileName = str(scenario) + '_c_n_' + str( n ) + '_k_' + str(n_clusters) + '_density_' + str(edge_density) + '_xi_' + str(xi) + '_nAverage' + str(nAverage) + '.pdf'

plotFigure( shape_range, results_mean, accuracy_err = results_std, methods = [ 'sbm', 'dcbm', 'pabm', 'osc', 'tcsc', 'gspc'], 
               xticks = shape_range, yticks = None,
               xlabel = r'$\sigma$', ylabel = "Accuracy",
               savefig = False, fileName = fileName )



# =============================================================================
# REBUTTAL: ACCURACY AS A FUNCTION OF THE EMBEDDING DIMENSION
# =============================================================================

scenario = 'homo-uniform-one' 
#choices: homo-uniform-one, homo-uniform-uniform, hetero-pareto


n = 900
n_clusters = 3
sizes = [ n//n_clusters for dummy in range( n_clusters ) ]
edge_density = 0.05
xi = 1
nAverage = 2


extra_parameter = [1]  
#This correspond to the parameter(s) for the distribution used
#e.g., the c for uniform, the exponent for Pareto, etc. 

P = generate_P_for_different_scenarios( scenario, sizes, edge_density, xi, extra_parameter )


results_mean, results_std = varying_embedding_dimension( n, sizes, P, nAverage = nAverage )
fileName = 'embeddingDimension_' + str(scenario) + '_n_' + str(n) + '_k_' + str(n_clusters) + '_density_' + str(edge_density) + '_nAverage' + str(nAverage) + '.pdf'


plotFigure( range( n_clusters, n_clusters**2+1 + n_clusters ), results_mean, accuracy_err = results_std, methods = [ 'sbm', 'dcbm', 'pabm' ], 
               xticks = None, yticks = None,
               xlabel = 'embedding dimension', ylabel = "Accuracy",
               savefig = False, fileName = fileName )


"""



def initialize_empty_dics( algorithms ):
    res = dict( )
    for algo in algorithms:
        res[ algo ] = [ ]
    return res 



def generate_P_for_different_scenarios( scenario, sizes, edge_density, xi, extra_parameter, min_value = 0.05, max_value = 5 ):
    n = sum(sizes)
    n_clusters = len( sizes )
    
    
    if scenario == 'homo-uniform-one':
        interactions = 'homogeneous'
        c = extra_parameter[0]
        theta_in = np.random.uniform( 1-c, 1 + c, size = n )
        theta_out = np.ones( n )

    
    elif scenario == 'homo-uniform-uniform':
        interactions = 'homogeneous'
        c = extra_parameter[0]
        theta_in = np.random.uniform( 1-c, 1 + c, size = n )
        theta_out = np.random.uniform( 1-c, 1 + c, size = n )

    elif scenario == 'hetero-uniform':
        interactions = 'heterogeneous'
        c = extra_parameter[ 0 ]
        Lambdas = [ [ ] for a in range( n_clusters ) ]
        for a in range( n_clusters ):
            Lambdas[ a ] = [ np.random.uniform( 1-c, 1 + c, size = sizes[ a ] ) for b in range(n_clusters) ]
        #for a in range( n_clusters ):
        #    Lambdas[a][1] = np.ones( sizes[ a ] ) 

    elif scenario == 'hetero-exponential-ones':
        interactions = 'heterogeneous'
        Lambdas = [ [ ] for a in range( n_clusters ) ]
        mean = np.mean ( truncatedDistributions( 'exponential', parameter = 1, min_value = min_value, max_value = max_value, size = 1000 ) )
        for a in range( n_clusters ):
            #mean = 1
            Lambdasa = [ ]
            for b in range( n_clusters ):
                if a==b:
                    Lambdasa.append( truncatedDistributions( 'exponential', parameter = 1, min_value = min_value, max_value = max_value, size = sizes[ a ] ) / mean )
                else:
                    Lambdasa.append( np.ones( sizes[a] ) )
            Lambdas[ a ] = Lambdasa
    

    elif scenario == 'hetero-pareto':
        interactions = 'heterogeneous'
        Lambdas = [ [ ] for a in range( n_clusters ) ]
        mean = np.mean ( truncatedDistributions( 'pareto', parameter = 1.5, min_value = min_value, max_value = max_value, size = 1000 ) )
        for a in range( n_clusters ):
            #mean = 1
            Lambdas[ a ] = [ truncatedDistributions( 'pareto', parameter = 1.5, min_value = min_value, max_value = max_value, size = sizes[ a ] ) / mean for b in range(n_clusters) ]
    
    elif scenario == 'hetero-exponential':
        interactions = 'heterogeneous'
        Lambdas = [ [ ] for a in range( n_clusters ) ]
        #mean = np.mean ( sp.stats.truncexpon.rvs( 5, scale = 1, size = 1000 ) )
        mean = np.mean ( truncatedDistributions( 'exponential', parameter = 1, min_value = min_value, max_value = max_value, size = 1000 ) )
        for a in range( n_clusters ):
            #Lambdas[ a ] = [  sp.stats.truncexpon.rvs( 5, size = sizes[ a ] ) / mean for b in range(n_clusters) ]
            mean = 1
            Lambdas[ a ] = [ truncatedDistributions( 'exponential', parameter = 1, min_value = min_value, max_value = max_value, size = sizes[ a ] ) / mean for b in range(n_clusters) ]
    
    
    if interactions == 'homogeneous':
        P = utils.generateP_homogeneousPABM( sizes, np.sqrt( edge_density ), np.sqrt( edge_density * xi ), theta_in, theta_out )
    
    else:
        for a in range( n_clusters ):
            for b in range( n_clusters ):
                Lambdas[ a ][ b ] *= np.sqrt( edge_density )
                if a!= b:
                    Lambdas[ a ][ b ] *= np.sqrt( xi )
        P = utils.generateP_inhomogeneousPABM( sizes, Lambdas )
    
    return P 

# =============================================================================
# CODE TO OBTAIN RESULTS OF THE DIFFERENT SCENARIOS
# =============================================================================


def varying_embedding_dimension( n, sizes, P, nAverage = 10, metric = 'accuracy' , algorithms = [ 'sbm', 'dcbm', 'pabm', 'osc', 'sklearn', 'tcsc' ] ):
    
    results_mean, results_std = dict(), dict()
    for version in algorithms:
        results_mean[ version ] = [ ]
        results_std[ version ] = [ ]
    
    n_clusters = len( sizes )
    embedding_dimension_range = range( n_clusters, n_clusters**2+1 + + n_clusters )
    
    labels_true = [ ]
    for community in range( n_clusters ):
        labels_true = labels_true + [ community+1 for i in range( sizes[community] ) ]
    labels_true = np.array( labels_true, dtype = int )

    for dummy in tqdm( range( len( embedding_dimension_range ) ) ):
        embedding_dimension = embedding_dimension_range[ dummy ]
        
        results = dict( )
        for version in algorithms: 
            results[ version ] = [ ]
        
        for run in range( nAverage ):
            A = utils.generateBernoulliAdjacency( P )
            vals, vecs = sp.sparse.linalg.eigsh( A.astype(float), k = embedding_dimension, which = 'LM' )
            
            for version in algorithms:
                
                if version == 'sbm':
                    z = KMeans(n_clusters = n_clusters, n_init = 'auto' ).fit_predict( vecs @ np.diag( vals ) ) + np.ones( n )
                
                elif version == 'dcbm':
                    hatP = vecs @ np.diag( vals ) @ vecs.T
                    hatP_rowNormalized = hatP
                    for i in range( n ):
                        if np.linalg.norm( hatP[i,:], ord = 1) != 0:
                            hatP_rowNormalized[i,:] = hatP[i,:] / np.linalg.norm( hatP[i,:], ord = 1)
                    z = KMeans(n_clusters = n_clusters, n_init = 'auto' ).fit_predict( hatP_rowNormalized ) + np.ones( n )        

                elif version == 'pabm':
                    model = selfrepresentation.ElasticNetSubspaceClustering( n_clusters = n_clusters ,algorithm = 'lasso_lars',gamma=50 ).fit( vecs @ np.diag( vals ) @ vecs.T )
                    z = model.labels_ + np.ones( n )
                
                elif version == 'tcsc':
                    z = tcsc.ThresholdedCosineSpectralClustering( A, n_clusters, number_eigenvectors = embedding_dimension )
                
                elif version == 'rtcsc' or version == 'r-tcsc':
                    z = tcsc.RefinedThresholdedCosineSpectralClustering( A, n_clusters, number_eigenvectors = embedding_dimension )

                elif version == 'sklearn':
                    z = SpectralClustering(n_clusters = n_clusters, affinity='precomputed', n_components = embedding_dimension).fit_predict( A ) + np.ones( n )
                
                elif version == 'osc':
                    vals_osc, vecs_osc = sp.sparse.linalg.eigsh( A.astype(float), k = embedding_dimension, which = 'BE' )
                    B = np.sqrt( n ) * vecs_osc @ vecs_osc.T
                    clustering_ = SpectralClustering( n_clusters = n_clusters, affinity='precomputed').fit( np.abs(B) )
                    z = clustering_.labels_ + np.ones( n )

                results[ version ].append( utils.computePartitionMetric( labels_true, z, metric = metric ) )
    
        for version in algorithms:
            results_mean[ version ].append( np.mean( results[ version ] ) )
            results_std[ version ].append( np.std( results[ version ] ) )
            
    return results_mean, results_std



def runScenario_c( algorithms, scenario, n, sizes, edge_density, community_strength, c_range, nAverage = 10, metric = 'accuracy', homogeneous = True ):
    
    results_mean = initialize_empty_dics( algorithms )
    results_std = initialize_empty_dics( algorithms )

    algos = results_mean.keys( )
    
    for dummy in tqdm( range( len( c_range ) ) ):
        c = c_range[ dummy ]
        if homogeneous:
            results_given_c = scenario_homogeneous( algorithms, scenario, n, sizes, edge_density, community_strength, c = c, nAverage = nAverage, metric = metric )
        else:
            results_given_c = scenario_inhomogeneous( algorithms, scenario, n, sizes, edge_density, community_strength, sigma = c, nAverage = nAverage, metric = metric )
            
        for algo in algos:
            results_mean[ algo ].append( np.mean( results_given_c[ algo ] ) )
            results_std[ algo ].append( np.std( results_given_c[ algo ] ) )
    
    return results_mean, results_std


def runScenario_xi( algorithms, scenario, n, sizes, edge_density, community_strength_range, c, nAverage = 10, metric = 'accuracy', homogeneous = True ):
    
    results_mean = initialize_empty_dics( algorithms )
    results_std = initialize_empty_dics( algorithms )
    
    for dummy in tqdm( range( len( community_strength_range ) ) ):
        community_strength = community_strength_range[ dummy ]
        
        if homogeneous == True:
            results_given_xi = scenario_homogeneous( algorithms, scenario, n, sizes, edge_density, community_strength, c = c, nAverage = nAverage, metric = metric )
        
        else:
            results_given_xi = scenario_inhomogeneous( algorithms, scenario, n, sizes, edge_density, community_strength, sigma = c, nAverage = nAverage, metric = metric )
            
        for algo in algorithms:
            results_mean[ algo ].append( np.mean( results_given_xi[ algo ] ) )
            results_std[ algo ].append( np.std( results_given_xi[ algo ] ) )
    
    return results_mean, results_std



def scenario_homogeneous( algorithms, scenario, n, sizes, edge_density, xi, c = 1, nAverage = 10, metric = 'accuracy', min_value = 0.1, max_value = 10):
    
    results = initialize_empty_dics( algorithms )
        
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
            mean = np.mean ( truncatedDistributions( 'pareto', parameter = 1.5, min_value = min_value, max_value = max_value, size = 1000 ) )
            theta_in = truncatedDistributions( 'pareto', parameter = 1.5, min_value = min_value, max_value = max_value, size = n ) / mean
            theta_out = np.ones( n )
            
        elif scenario == 'lognormal':
            mean = np.mean ( truncatedDistributions( 'lognormal', parameter = 1, min_value = min_value, max_value = max_value, size = 1000 ) )
            theta_in = truncatedDistributions( 'lognormal', parameter = 1, min_value = min_value, max_value = max_value, size = n ) / mean
            theta_out = np.ones( n )
        
        elif scenario == 'exponential':
            mean = np.mean ( truncatedDistributions( 'exponential', parameter = 1, min_value = min_value, max_value = max_value, size = 1000 ) )
            theta_in = truncatedDistributions( 'exponential', parameter = 1, min_value = min_value, max_value = max_value, size = n ) / mean
            theta_out = np.ones( n )
        
        else:
            raise TypeError( 'Please provide a scenario implemented' )
        
        P = utils.generateP_of_homogeneousPABM( labels_true, np.sqrt( edge_density ), np.sqrt( edge_density * xi ), theta_in, theta_out )
        A = utils.generateBernoulliAdjacency( P )
        
        predicted_clusterings = getClusterings( A, n_clusters, algorithms=algorithms )
        
        for algo in algorithms:
            results[ algo ].append( utils.computePartitionMetric( labels_true, predicted_clusterings[ algo ], metric = metric ) )
        
    results_mean, results_std = dict( ), dict( )

    for algo in algorithms:
        results_mean[ algo ] = np.mean( results[ algo ] )
        results_std[ algo ] = np.std( results[ algo ] )
    
    return results 



def scenario_inhomogeneous( algorithms, scenario, n, sizes, edge_density, xi, sigma = 1, nAverage = 10, metric = 'accuracy', min_value = 0.1, max_value = 10 ):
    
    results = initialize_empty_dics( algorithms )
    
    
    n_clusters = len( sizes )
    labels_true = [ ]
    
    for community in range( n_clusters ):
        labels_true = labels_true + [ community+1 for i in range( sizes[community] ) ]
    labels_true = np.array( labels_true, dtype = int )

    for run in range( nAverage ):
        
        if scenario == 'pareto':
            Lambdas = [ [ ] for a in range( n_clusters ) ]
            mean = np.mean ( truncatedDistributions( 'pareto', parameter = sigma, min_value = min_value, max_value = max_value, size = 1000 ) )
            for a in range( n_clusters ):
                #Lambdas[ a ] = [ np.random.lognormal( mean = 1, sigma = 1, size = sizes[ a ] ) for b in range(n_clusters) ]
                #Lambdas[ a ] = [ 1/2 * np.random.beta( 2, 2, size = sizes[ a ] ) for b in range( n_clusters ) ]   
                #mean = np.mean ( sp.stats.truncpareto.rvs( 1.5, 5, size = 1000 ) )
                #Lambdas[ a ] = [ sp.stats.truncpareto.rvs( 1.5, 5, size = sizes[ a ] ) / mean for b in range(n_clusters) ]
                #mean = 1
                Lambdas[ a ] = [ truncatedDistributions( 'pareto', parameter = sigma, min_value = min_value, max_value = max_value, size = sizes[ a ] ) / mean for b in range(n_clusters) ]

                
        if scenario == 'lognormal':
            Lambdas = [ [ ] for a in range( n_clusters ) ]
            mean = np.mean ( truncatedDistributions( 'lognormal', parameter = sigma, min_value = min_value, max_value = max_value, size = 1000 ) )
            for a in range( n_clusters ):
                #Lambdas[ a ] = [ np.exp( -sigma**2 / 2) * np.random.lognormal( mean = 0, sigma = sigma, size = sizes[ a ] ) for b in range(n_clusters) ]
                Lambdas[ a ] = [ truncatedDistributions( 'lognormal', parameter = sigma, min_value = min_value, max_value = max_value, size = sizes[ a ] ) / mean for b in range(n_clusters) ]

        if scenario == 'exponential':
            Lambdas = [ [ ] for a in range( n_clusters ) ]
            #mean = np.mean ( sp.stats.truncexpon.rvs( 5, scale = 1, size = 1000 ) )
            mean = np.mean ( truncatedDistributions( 'exponential', parameter = 1, min_value = min_value, max_value = max_value, size = 1000 ) )
            for a in range( n_clusters ):
                #Lambdas[ a ] = [  sp.stats.truncexpon.rvs( 5, size = sizes[ a ] ) / mean for b in range(n_clusters) ]
                Lambdas[ a ] = [ truncatedDistributions( 'exponential', parameter = 1, min_value = min_value, max_value = max_value, size = sizes[ a ] ) / mean for b in range(n_clusters) ]

        rateMatrix = np.zeros( ( n_clusters, n_clusters ) )
        for a in range( n_clusters ):
            for b in range( n_clusters ):
                if a == b:
                    rateMatrix[ a, b ] = edge_density 
                else:
                    rateMatrix[ a, b ] = edge_density * xi
        
        P = utils.generateP_inhomogeneousPABM( sizes, rateMatrix, Lambdas )
        A = utils.generateBernoulliAdjacency( P )
        
        predicted_clusterings = getClusterings( A, n_clusters, algorithms = algorithms )
        
        for algo in algorithms:
            results[ algo ].append( utils.computePartitionMetric( labels_true, predicted_clusterings[ algo ], metric = metric ) )
        
    results_mean, results_std = dict( ), dict( )

    for algo in algorithms:
        results_mean[ algo ] = np.mean( results[ algo ] )
        results_std[ algo ] = np.std( results[ algo ] )
    
    return results 



def results_std_to_ste( results_std, nAverage ):
    results_ste = dict( )
    for algo in results_std.keys( ):
        results_ste[ algo ] = np.asarray( results_std[ algo ] ) / np.sqrt( nAverage )
    return results_ste




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


def getClusterings( A, n_clusters, algorithms = ['sbm', 'dcbm', 'pabm', 'osc', 'tcsc', 'gspc', 'sklearn'], verbose = False ):
    clusterings = dict( )
    
    for algorithm in algorithms:
        if verbose:
            print( 'Running algorithm :' , algorithm )
            
        if algorithm == 'tcsc':
            z = tcsc.ThresholdedCosineSpectralClustering(A, n_clusters)
        elif algorithm == 'r-tcsc' or algorithm == 'rtcsc':
            z = tcsc.RefinedThresholdedCosineSpectralClustering( A, n_clusters )
        elif algorithm == 'gspc':
            z = gspc.GreedySubspaceProjectionClustering(A, n_clusters, model = 'pabm' )            
        else:
            z = clustering.graph_clustering( A , n_clusters , variant = algorithm )
        
        clusterings[ algorithm ] = z
 
    return clusterings


def truncatedDistributions( distribution, parameter, min_value = 0, max_value = 10, size = 1, max_trials = 100 ):
    
    if distribution == 'exponential':
        samples = sp.stats.expon.rvs( scale = 1/parameter, size = max_trials * size )
        
    elif distribution == 'pareto':
        samples = sp.stats.pareto.rvs( parameter, size = max_trials * size ) - 1
    
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
            
    

