#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 12:03:41 2024

@author: maximilien dreveton
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from tqdm import tqdm


import utils as utils
import clustering as clustering
import thresholdedCosineSpectralClustering as tcsc
import greedySubspaceProjectionClustering as gspc

from sklearn.cluster import KMeans, SpectralClustering
import selfrepresentation as selfrepresentation


SIZE_TITLE = 24
SIZE_LABELS = 24
SIZE_TICKS = 20
SIZE_LEGEND = 18


__homogeneous_scenarios__ = [ 'homo-uniform-one', 'homo-uniform-uniform', 'homo-beta-beta', 'homo-beta-one' ]

__heterogeneous_scenarios__ = [ 'hetero-uniform', 'hetero-pareto', 'hetero-exponential', 
                               'hetero-lognormal', 'hetero-exponential-ones',
                               'hetero-beta' ]

__scenarios_implemented__ = __homogeneous_scenarios__ + __heterogeneous_scenarios__

__algorithms_implemented__ = [ 'sbm', 'dcbm', 'pabm', 'osc', 'sklearn', 'tcsc', 'gspc' ]


"""
# =============================================================================
# HOMOGENEOUS SCENARIOS: FIGURE 1a AND 1b
# =============================================================================


###### VARYING XI

scenario = 'homo-uniform-uniform'

n = 1200
n_clusters = 3
sizes = [ n//n_clusters for dummy in range( n_clusters ) ]
edge_density = 0.1
nAverage = 2

xi_range = [ 0, 0.2, 0.4, 0.6, 0.8, 1 ]
c = 0.8

results_mean, results_std = varyingCommunityStrength( scenario, n, sizes, edge_density, xi_range, c, nAverage = nAverage, metric = 'accuracy', homogeneous = True )

fileName = str(scenario) + '_xi' + '_n_' + str(n) + '_k_' + str(n_clusters) + '_density_' + str(edge_density) + '_nAverage_' + str(nAverage) + '.pdf'
plotFigure( xi_range, results_mean, accuracy_err = results_std, methods = [ 'sbm', 'dcbm', 'pabm', 'osc'], 
               xticks = None, yticks = None,
               xlabel = r"$xi$", ylabel = "Accuracy",
               savefig = False, fileName = fileName )


###### VARYING C

scenario = 'homo-uniform-uniform' #other possibilities are in __homogeneous_scenarios__

n = 1200
n_clusters = 3
sizes = [ n//n_clusters for dummy in range( n_clusters ) ]
edge_density = 0.1
nAverage = 2

xi = 1
c_range = [ 0,0.2,0.4,0.6,0.8,1 ]

results_mean, results_std = varyingUniformDistributionParameter( scenario, n, sizes, edge_density, xi, c_range, nAverage = nAverage, metric = 'accuracy' )

fileName = str(scenario) + '_c_n_' + str( n ) + '_k_' + str(n_clusters) + '_density_' + str(edge_density) + '_xi_' + str(xi) + '_nAverage' + str(nAverage) + '.pdf'
plotFigure( c_range, results_mean, accuracy_err = results_std, methods = [ 'sbm', 'dcbm', 'pabm', 'osc'], 
               xticks = None, yticks = None,
               xlabel = "c", ylabel = "Accuracy",
               savefig = False, fileName = fileName )





# =============================================================================
# HETEROGENEOUS SCENARIOS (FIGURE .. IN APPENDIX)
# =============================================================================

scenario = 'hetero-pareto'  #all possibilities are in __heterogeneous_scenarios__
#To reproduce the Figure use pareto, exponential, and lognormal

nAverage = 2

n = 2000
n_clusters = 5
sizes = [ n//n_clusters for dummy in range( n_clusters ) ]
edge_density = 0.05

xi_range = [ 0,0.2,0.4,0.6,0.8,1 ]
xi_range = np.linspace(0, 1 , 11)
xi_range[0] = 0.01 #To avoid problem with disconnected graphs
c = 0.5

results_mean, results_std = varyingCommunityStrength( scenario, n, sizes, edge_density, xi_range, c, nAverage = nAverage, metric = 'accuracy' , homogeneous = False )
    
fileName = 'hetero_' + str(scenario) + '_xi_n_' + str(n) + '_k_' + str(n_clusters) + '_density_' + str(edge_density) + '_nAverage' + str(nAverage) + '.pdf'
plotFigure( xi_range, results_mean, accuracy_err = results_std, methods = [ 'sbm', 'dcbm', 'pabm', 'osc'], 
               xticks = None, yticks = None,
               xlabel = r'$xi$', ylabel = "Accuracy",
               savefig = False, fileName = fileName )




# =============================================================================
# ACCURACY AS A FUNCTION OF THE EMBEDDING DIMENSION
# =============================================================================


scenario = 'homo-uniform-one' #choices in __scenarios_implemented__

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

results_mean, results_std = varyingEmbeddingDimension( n, sizes, P, nAverage = nAverage )
fileName = 'embeddingDimension_' + str(scenario) + '_n_' + str(n) + '_k_' + str(n_clusters) + '_density_' + str(edge_density) + '_nAverage' + str(nAverage) + '.pdf'

plotFigure( range( n_clusters, n_clusters**2+1 + n_clusters ), results_mean, accuracy_err = results_std, methods = [ 'sbm', 'dcbm', 'pabm',  'osc' ], 
               xticks = None, yticks = None,
               xlabel = 'embedding dimension', ylabel = "Accuracy",
               savefig = False, fileName = fileName )




# =============================================================================
# TRIALS P WITH DIFFERENT RANKS
# =============================================================================

n = 900
n_clusters = 3
sizes = [ n//n_clusters for dummy in range( n_clusters ) ]
labels_true = utils.generate_labels( sizes )

intra_edge_density = 0.1
xi = 1
c = 0.8

rateMatrix = utils.generateHomoegeneousRateMatrix( n_clusters, intra_edge_density, intra_edge_density*xi )    

Lambdas = [ [ ] for _ in range(n_clusters) ] 

for a in range( n_clusters ):
    Lambdas[ a ] = [ np.ones( sizes[a] ) for b in range(n_clusters) ]
for a in range(n_clusters):
    Lambdas[a][a] = np.random.uniform( 1-c, 1 + c, size = sizes[a] ) 
#Lambdas is a list of list of arrays such that the element Lambdas(a,b) is an array providing the popularity coefficient of the vertices in community a to the vertices in community b


Lambdas = [ [ ] for _ in range(n_clusters) ] 
for a in range( n_clusters ):
    Lambdas[ a ] = [ np.random.uniform( 1-c, 1 + c, size = sizes[a] ) for b in range(n_clusters) ]



for a in range( n_clusters ):
    for b in range( n_clusters ):
        if a==b:
            Lambdas[a][b] *= np.sqrt(p)
        else:
            Lambdas[a][b] *= np.sqrt(q)
            
P = utils.generateP_inhomogeneousPABM( sizes, rateMatrix, Lambdas )
A = utils.generateBernoulliAdjacency( P )



-----

xi = 0.5
edge_density = 0.1


theta_in = np.random.uniform( 1-c, 1 + c, size = n )
theta_out = np.random.uniform( 1-c, 1 + c, size = n )

P = utils.generateP_homogeneousPABM( sizes, np.sqrt( edge_density ), np.sqrt( edge_density * xi ), theta_in, theta_out )
A = utils.generateBernoulliAdjacency( P )


"""







def initialize_empty_dics( algos = __algorithms_implemented__ ):
    res = dict( )
    for algo in algos:
        res[ algo ] = [ ]
    return res 



def generate_P_for_different_scenarios( scenario, sizes, intra_edge_density, xi, extra_parameter, min_value = 0.01, max_value = 4 ):
    n = sum(sizes)
    n_clusters = len( sizes )
    
    if scenario not in __scenarios_implemented__:
        raise TypeError( 'This scenario is not implemented' )
    
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
        
    elif scenario == 'homo-beta-beta':
        interactions = 'homogeneous'
        theta_in = 2 * np.random.beta( 2, 2, size = n )
        theta_out = 2 * np.random.beta( 2, 2, size = n )

    elif scenario == 'homo-beta-one':
        interactions = 'homogeneous'
        theta_in = 2 * np.random.beta( 2, 2, size = n )
        theta_out = np.ones( n )

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
            Lambdasa = [ ]
            for b in range( n_clusters ):
                if a==b:
                    Lambdasa.append( truncatedDistributions( 'exponential', parameter = 1, min_value = min_value, max_value = max_value, size = sizes[ a ] ) / mean )
                else:
                    Lambdasa.append( np.ones( sizes[a] ) )
            Lambdas[ a ] = Lambdasa

    elif scenario == 'hetero-beta':
        interactions = 'heterogeneous'
        Lambdas = [ [ ] for a in range( n_clusters ) ]
        for a in range( n_clusters ):
            Lambdas[ a ] = [ 2 * np.random.beta( 2, 2, size = sizes[ a ] ) for b in range(n_clusters) ]

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
        mean = np.mean ( truncatedDistributions( 'exponential', parameter = 1, min_value = min_value, max_value = max_value, size = 1000 ) )
        mean = 1
        for a in range( n_clusters ):
            Lambdas[ a ] = [ truncatedDistributions( 'exponential', parameter = 1, min_value = min_value, max_value = max_value, size = sizes[ a ] ) / mean for b in range(n_clusters) ]
    
    p = intra_edge_density
    q = intra_edge_density * xi

    if interactions == 'homogeneous':
        P = utils.generateP_homogeneousPABM( sizes, p, q, theta_in, theta_out )
    
    else:
        rateMatrix = utils.generateHomoegeneousRateMatrix( n_clusters, p, q )
        P = utils.generateP_inhomogeneousPABM( sizes, rateMatrix, Lambdas )
    
    return P 

# =============================================================================
# CODE TO OBTAIN RESULTS OF THE DIFFERENT SCENARIOS
# =============================================================================


def varyingEmbeddingDimension( n, sizes, P, nAverage = 10, metric = 'accuracy' , algos = [ 'sbm', 'dcbm', 'pabm', 'osc' ] ):
    
    results_mean, results_std = dict(), dict()
    for version in algos:
        results_mean[ version ] = [ ]
        results_std[ version ] = [ ]
    
    n_clusters = len( sizes )
    embedding_dimension_range = range( n_clusters, n_clusters**2+1 + n_clusters )
    
    labels_true = utils.generate_labels( sizes )

    for dummy in tqdm( range( len( embedding_dimension_range ) ) ):
        embedding_dimension = embedding_dimension_range[ dummy ]
        print( 'The embedding dimension is : ', embedding_dimension )
        
        results = dict( )
        for version in algos: 
            results[ version ] = [ ]
        
        for run in range( nAverage ):
            A = utils.generateBernoulliAdjacency( P )
            vals, vecs = sp.sparse.linalg.eigsh( A.astype(float), k = embedding_dimension, which = 'LM' )
            
            for version in algos:
                
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
                    model = selfrepresentation.ElasticNetSubspaceClustering( n_clusters = n_clusters ,algorithm = 'lasso_lars',gamma=50 ).fit( vecs @ np.diag( vals ) )
                    z = model.labels_ + np.ones( n )
                    
                elif version == 'osc':
                    vals, vecs = sp.sparse.linalg.eigsh( A.astype(float), k = embedding_dimension, which = 'BE' )
                    B = np.sqrt( n ) * vecs @ vecs.T
                    clustering_ = SpectralClustering( n_clusters = n_clusters, affinity = 'precomputed').fit( np.abs(B) )
                    z = clustering_.labels_ + np.ones( n )
                
                results[ version ].append( utils.computePartitionMetric( labels_true, z, metric = metric ) )
    
        for version in algos:
            results_mean[ version ].append( np.mean( results[ version ] ) )
            results_std[ version ].append( np.std( results[ version ] ) )
            
    return results_mean, results_std



def varyingUniformDistributionParameter( scenario, n, sizes, edge_density, community_strength, c_range, nAverage = 10, metric = 'accuracy' ):
    
    if scenario not in [ 'homo-uniform-one', 'homo-uniform-uniform', 'hetero-uniform']:
        raise TypeError( 'The scenario should involve the uniform distribution' )
    
    results_mean = initialize_empty_dics( )
    results_std = initialize_empty_dics( )

    algos = results_mean.keys( )
    
    for dummy in tqdm( range( len( c_range ) ) ):
        c = c_range[ dummy ]
        
        P = generate_P_for_different_scenarios( scenario, sizes, edge_density, community_strength, extra_parameter = [c], min_value = 0.1, max_value = 5 )
        labels_true = utils.generate_labels( sizes )
        
        results_given_c = run_experiments( P, labels_true, algos, nAverage = nAverage, metric = metric )
        
        for algo in algos:
            results_mean[ algo ].append( np.mean( results_given_c[ algo ] ) )
            results_std[ algo ].append( np.std( results_given_c[ algo ] ) )
    
    return results_mean, results_std


def varyingCommunityStrength( scenario, n, sizes, edge_density, community_strength_range, c, nAverage = 10, metric = 'accuracy' , homogeneous = True ):
    
    if scenario not in __scenarios_implemented__:
        raise TypeError( 'The scenario is not implemented' )
    
    results_mean = initialize_empty_dics( )
    results_std = initialize_empty_dics( )
    algos = results_mean.keys( )
    
    for dummy in tqdm( range( len( community_strength_range ) ) ):
        community_strength = community_strength_range[ dummy ]
        P = generate_P_for_different_scenarios( scenario, sizes, edge_density, community_strength, extra_parameter = [c] )
        labels_true = utils.generate_labels( sizes )

        results_given_xi = run_experiments( P, labels_true, algos, nAverage = nAverage, metric = metric )
                    
        for algo in algos:
            results_mean[ algo ].append( np.mean( results_given_xi[ algo ] ) )
            results_std[ algo ].append( np.std( results_given_xi[ algo ] ) )
    
    return results_mean, results_std



def run_experiments( P, labels_true, algos, nAverage = 5, metric = 'accuracy' ):
    
    n_clusters = len( set(labels_true ) )
    results = initialize_empty_dics( algos )

    for run in range( nAverage ):

        A = utils.generateBernoulliAdjacency( P )
        predicted_clusterings = getClusterings( A, n_clusters )
    
        for algo in algos:
            results[ algo ].append( utils.computePartitionMetric( labels_true, predicted_clusterings[ algo ], metric = metric ) )

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
    z_tcsc = tcsc.RefinedThresholdedCosineSpectralClustering(A, n_clusters)
    z_gspc = gspc.GreedySubspaceProjectionClustering( A, n_clusters, model = 'pabm' )
 
    return { 'sbm' : z_bm, 'dcbm' : z_dcbm, 'pabm' : z_pabm, 'osc' : z_osc, 'sklearn' : z_sklearn, 'tcsc' : z_tcsc, 'gspc' : z_gspc }



def truncatedDistributions( distribution, parameter, min_value = 0.01, max_value = 10, size = 1, max_trials = 100 ):
    
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
            
    

