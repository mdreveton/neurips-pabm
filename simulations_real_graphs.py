#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 15:47:24 2025
"""

import networkx as nx
import utils as utils
import pandas as pd
import numpy as np

import datasets as datasets
import clustering as clustering

import time
from tqdm import tqdm

import thresholdedCosineSpectralClustering as tcsc
import greedySubspaceProjectionClustering as gspc


gml_datasets = [ 'citeseer', 'cora', 'liveJournal-top2', 'politicalBlogs' ]
gl_datasets = [ 'mnist', 'fashionmnist', 'cifar10' ]

#__algorithms__ = [ 'bm', 'dcbm', 'pabm-k', 'pabm-2k', 'pabm-ksquared', 'osc', 'sklearn', 'tcsc' ]
__algorithms__ = ['bm', 'dcbm', 'pabm', 'osc', 'tcsc', 'gspc', 'sklearn' ]


"""
# =============================================================================
# REPRODUCE RESULTS OF TABLE 1
# =============================================================================

algorithms = ['sbm', 'dcbm', 'pabm', 'osc', 'tcsc', 'rtcsc', 'gspc', 'sklearn' ]

dataset_names = [ 'politicalBlogs', 'liveJournal-top2', 'citeseer', 'cora', 'mnist', 'fashionmnist', 'cifar10' ]
results = runExperiments( gml_dataset, algorithms = algorithms, saveResults = True, filename = 'real_data', nAverage = 5, verbose = True )


df_statistics = pd.DataFrame(  )

df_accuracy = pd.DataFrame( )
df_ami = pd.DataFrame( )
df_ari = pd.DataFrame( )
df_time = pd.DataFrame( )

for dataset_name in gml_datasets:
    
    print( dataset_name )
    A, labels_true = datasets.getRealGraph( dataset_name , n = 10000 )
    
    
    #G = readFromGml( dataset_name )
    #labels_true = get_communities( G, community_label = 'community' )
    n_clusters = len( set( labels_true ) )
    #A = nx.adjacency_matrix( G )

    graph_statistics = getGraphStatistics( nx.from_scipy_sparse_array( A ), n_clusters, dataset_name = dataset_name )
    df_statistics = pd.concat( [ df_statistics, graph_statistics ] )
        
    predicted_clusterings, time_execution = getClusterings( A, n_clusters, algorithms = algorithms )
    
    accuracies = getClusteringMetrics( predicted_clusterings, labels_true, dataset_name = dataset_name, metric = 'accuracy' )
    ami = getClusteringMetrics( predicted_clusterings, labels_true, dataset_name = dataset_name, metric = 'ami' )
    ari = getClusteringMetrics( predicted_clusterings, labels_true, dataset_name = dataset_name, metric = 'ari' )
    #cc = getClusteringMetrics( predicted_clusterings, labels_true, dataset_name = dataset_name, metric = 'correlation coefficient' )
    
    df_accuracy = pd.concat( [ df_accuracy, accuracies ] )
    df_ami = pd.concat( [ df_ami, ami ] )
    df_ari = pd.concat( [ df_ari, ari ] )
    #df_cc = pd.concat( [ df_cc, cc ] )
    
    time_execution[ 'name' ] = dataset_name
    time_execution = pd.DataFrame( data = [ time_execution ] )

    df_time = pd.concat( [ df_time, time_execution ] )

    
df_accuracy.to_csv( 'gml_clusteringAccuracy.csv', index = False )
df_ami.to_csv( 'gml_clusteringAMI.csv', index = False )
df_ari.to_csv( 'gml_clusteringARI.csv', index = False )
#df_cc.to_csv( 'gml_clustering_correlationCoeff.csv', index = False )
df_time.to_csv( 'gml_executionTime.csv', index = False )

df_statistics.to_csv( 'gml_graphStatistics.csv', index = False )

"""


def initialize_empty_dics( algorithms ):
    res = dict( )
    for algo in algorithms:
        res[ algo ] = [ ]
    return res 


def runExperiments_old( dataset_names, algorithms = __algorithms__, saveResults = False, filename = 'gml', nAverage = 1 ):
    df_statistics = pd.DataFrame(  )

    df_accuracy = pd.DataFrame( )
    df_ami = pd.DataFrame( )
    df_ari = pd.DataFrame( )
    df_cc = pd.DataFrame( )
    df_time = pd.DataFrame( )

    for dataset_name in dataset_names:
        
        print( dataset_name )
        A, labels_true = datasets.getRealGraph( dataset_name , n = 10000 )
        n_clusters = len( set( labels_true ) )

        graph_statistics = getGraphStatistics( nx.from_scipy_sparse_array( A ), n_clusters, dataset_name = dataset_name )
        df_statistics = pd.concat( [ df_statistics, graph_statistics ] )
        
        for i in range( nAverage ):
            predicted_clusterings, time_execution = getClusterings( A, n_clusters, algorithms = algorithms )
        
            accuracies = getClusteringMetrics( predicted_clusterings, labels_true, dataset_name = dataset_name, metric = 'accuracy' )
            ami = getClusteringMetrics( predicted_clusterings, labels_true, dataset_name = dataset_name, metric = 'ami' )
            ari = getClusteringMetrics( predicted_clusterings, labels_true, dataset_name = dataset_name, metric = 'ari' )
            #cc = getClusteringMetrics( predicted_clusterings, labels_true, dataset_name = dataset_name, metric = 'correlation coefficient' )
        
        df_accuracy = pd.concat( [ df_accuracy, accuracies ] )
        df_ami = pd.concat( [ df_ami, ami ] )
        df_ari = pd.concat( [ df_ari, ari ] )
        #df_cc = pd.concat( [ df_cc, cc ] )
        
        time_execution[ 'name' ] = dataset_name
        time_execution = pd.DataFrame( data = [ time_execution ] )

        df_time = pd.concat( [ df_time, time_execution ] )

    if saveResults:
        print( 'Saving results...' )
        df_accuracy.to_csv( filename + '_clusteringAccuracy.csv', index = False )
        df_ami.to_csv( filename + '_clusteringAMI.csv', index = False )
        df_ari.to_csv( filename + '_clusteringARI.csv', index = False )
        df_time.to_csv( filename + '_executionTime.csv', index = False )
        df_statistics.to_csv( filename + '_graphStatistics.csv', index = False )
    
    return {
        'statistics' : df_statistics,
        'accuracy' : df_accuracy,
        'ami' : df_ami,
        'ari' : df_ari,
        'time' : df_time 
    }



def getClusterings( A, n_clusters, algorithms = __algorithms__, verbose = False ):
    
    time_execution = dict( )
    clusterings = dict( )

    for algorithm in algorithms:
        if verbose:
            print( 'Running algorithm :' , algorithm )
        start_time = time.time( )
        if algorithm == 'tcsc':
            z = tcsc.ThresholdedCosineSpectralClustering(A, n_clusters)
        elif algorithm == 'r-tcsc' or algorithm == 'rtcsc':
            z = tcsc.RefinedThresholdedCosineSpectralClustering(A, n_clusters)
        elif algorithm == 'gspc':
            z = gspc.GreedySubspaceProjectionClustering(A, n_clusters, model = 'pabm' )            
        else:
            z = clustering.graph_clustering( A , n_clusters , variant = algorithm )
        end_time = time.time( )
        
        clusterings[ algorithm ] = z
        time_execution[ algorithm ] = end_time - start_time
        if verbose:
            print( 'Algorithm took ', end_time - start_time , ' seconds.' )
    
    #z_dcbm = clustering.spectralClustering_dcbm( A , n_clusters )
    #z_pabm_k = clustering.spectralClustering_pabm( A, n_clusters, infer_rank = n_clusters )
    #z_pabm_2k = clustering.spectralClustering_pabm( A, n_clusters, infer_rank = 2 * n_clusters )
    #z_pabm_ksquared = clustering.spectralClustering_pabm( A, n_clusters, infer_rank = n_clusters * n_clusters )
    #z_osc = clustering.orthogonalSpectralClustering( A, n_clusters )

    #return { 'bm' : z_bm, 'dcbm' : z_dcbm, 'pabm-k' : z_pabm_k, 'pabm-2k' : z_pabm_2k, 'pabm-ksquared' : z_pabm_ksquared, 'osc' : z_osc }

    return clusterings, time_execution
    


def runExperimentsSingleDataset( dataset_name, algorithms = __algorithms__, nAverage = 1, verbose = False ):
    A, labels_true = datasets.getRealGraph( dataset_name , n = 10000 )
    n_clusters = len( set( labels_true ) )

    graph_statistics = getGraphStatistics( nx.from_scipy_sparse_array( A ), n_clusters, dataset_name = dataset_name )
    
    accuracies, ami, ari, time_execution = initialize_empty_dics( algorithms ), initialize_empty_dics( algorithms ), initialize_empty_dics( algorithms ), initialize_empty_dics( algorithms )

    if verbose:
        iteration = tqdm( range( nAverage) )
    else:
        iteration = range( nAverage)
    
    for _ in iteration:
        predicted_clusterings, time_execution_single_run = getClusterings( A, n_clusters, algorithms = algorithms )
        
        for algo in algorithms:
            accuracies[algo].append( utils.computePartitionMetric( labels_true, predicted_clusterings[ algo ], metric = 'accuracy' ) )
            ami[algo].append( utils.computePartitionMetric( labels_true, predicted_clusterings[ algo ], metric = 'ami' ) )
            ari[algo].append( utils.computePartitionMetric( labels_true, predicted_clusterings[ algo ], metric = 'ari' ) )
            time_execution[algo].append( time_execution_single_run[algo] )

    accuracies_mean = { algo : np.mean( accuracies[ algo ] ) for algo in algorithms }
    ami_mean = { algo : np.mean( ami[ algo ] ) for algo in algorithms }
    ari_mean = { algo : np.mean( ari[ algo ] ) for algo in algorithms }
    time_execution_mean = { algo : np.mean( time_execution[ algo ] ) for algo in algorithms }

    accuracies_std = { algo : np.std( accuracies[ algo ] ) for algo in algorithms }
    ami_std = { algo : np.std( ami[ algo ] ) for algo in algorithms }
    ari_std = { algo : np.std( ari[ algo ] ) for algo in algorithms }
    time_execution_std = { algo : np.std( time_execution[ algo ] ) for algo in algorithms }

    return {
        'statistics' : graph_statistics,
        'accuracy_mean' : accuracies_mean,
        'accuracy_std' : accuracies_std,
        'ami_mean' : ami_mean,
        'ami_std' : ami_std,
        'ari_mean' : ari_mean,
        'ari_std' : ari_std,
        'execution_time_mean' : time_execution_mean,
        'execution_time_std' : time_execution_std,
    }




def runExperiments( dataset_names, algorithms = __algorithms__, saveResults = False, filename = '', nAverage = 1, verbose = False ):
    df_statistics = pd.DataFrame(  )

    
    df_accuracy_mean = pd.DataFrame( )
    df_accuracy_std = pd.DataFrame( )
    df_ari_mean = pd.DataFrame( )
    df_ari_std = pd.DataFrame( )
    df_ami_mean = pd.DataFrame( )
    df_ami_std = pd.DataFrame( )
    
    df_time_mean, df_time_std = pd.DataFrame( ), pd.DataFrame( )

    for dataset_name in dataset_names:
        if verbose:
            print( dataset_name )
        results = runExperimentsSingleDataset( dataset_name, algorithms = algorithms, nAverage = nAverage, verbose = verbose )
        
        df_statistics = pd.concat( [ df_statistics, results['statistics'] ] )
        
        accuracy_mean = {'name' : dataset_name }
        accuracy_mean.update( results['accuracy_mean'] )
        df_accuracy_mean = pd.concat( [ df_accuracy_mean, pd.DataFrame( data = [ accuracy_mean ] ) ] )
        
        accuracy_std = {'name' : dataset_name }
        accuracy_std.update( results['accuracy_std'] )
        df_accuracy_std = pd.concat( [ df_accuracy_std, pd.DataFrame( data = [ accuracy_std ] ) ] )

        ari_mean = {'name' : dataset_name }
        ari_mean.update( results['ari_mean'] )
        df_ari_mean = pd.concat( [ df_ari_mean, pd.DataFrame( data = [ ari_mean ] ) ] )
        
        ari_std = {'name' : dataset_name }
        ari_std.update( results['ari_std'] )
        df_ari_std = pd.concat( [ df_ari_std, pd.DataFrame( data = [ ari_std ] ) ] )
        
        ami_mean = {'name' : dataset_name }
        ami_mean.update( results['ami_mean'] )
        df_ami_mean = pd.concat( [ df_ami_mean, pd.DataFrame( data = [ ami_mean ] ) ] )
        
        ami_std = {'name' : dataset_name }
        ami_std.update( results['ami_std'] )
        df_ami_std = pd.concat( [ df_ami_std, pd.DataFrame( data = [ ami_std ] ) ] )


        time_mean = {'name' : dataset_name }
        time_mean.update( results['execution_time_mean'] )
        df_time_mean = pd.concat( [ df_time_mean, pd.DataFrame( data = [ time_mean ] ) ] )
        
        time_std = {'name' : dataset_name }
        time_std.update( results['execution_time_std'] )
        df_time_std = pd.concat( [ df_time_std, pd.DataFrame( data = [ time_std ] ) ] )
        
        

    if saveResults:
        print( 'Saving results...' )
        df_accuracy_mean.to_csv( filename + '_accuracy_mean_nAverage_' + str(nAverage) + '.csv', index = False )
        df_accuracy_std.to_csv( filename + '_accuracy_std_ ' + str(nAverage) + '.csv', index = False )
        df_ami_mean.to_csv( filename + '_AMI_mean_' + str(nAverage) + '.csv', index = False )
        df_ami_std.to_csv( filename + '_AMI_std_' + str(nAverage) + '.csv', index = False )
        df_ari_mean.to_csv( filename + '_ARI_mean_' + str(nAverage) + '.csv', index = False )
        df_ari_std.to_csv( filename + '_ARI_std_' + str(nAverage) + '.csv', index = False )
        df_time_mean.to_csv( filename + '_time_mean_' + str(nAverage) + '.csv', index = False )
        df_time_std.to_csv( filename + '_time_std_' + str(nAverage) + '.csv', index = False )

        df_statistics.to_csv( filename + '_graphStatistics.csv', index = False )
    
    return {
        'statistics' : df_statistics,
        'accuracy_mean' : df_accuracy_mean,
        'accuracy_std' : df_accuracy_std,
        'ami_mean' : df_ami_mean,
        'ami_std' : df_ami_std,
        'ari_mean' : df_ari_mean,
        'ari_std' : df_ari_std,
        'time_mean' : df_time_mean,
        'time_std' : df_time_std,
    }

    


def getGraphStatistics( G, n_clusters, dataset_name = '' ):
    graph_statistics = dict( )
    graph_statistics[ 'name' ] = dataset_name
    graph_statistics[ 'n' ] = nx.number_of_nodes( G )
    graph_statistics[ 'E' ] = nx.number_of_edges( G )
    graph_statistics[ 'k' ] = n_clusters
    graph_statistics[ 'average degree' ] = 2 * nx.number_of_edges( G ) / nx.number_of_nodes( G )
    
    degrees = [ deg[1] for deg in G.degree() ]
    graph_statistics[ 'std-degree' ] = np.std( degrees )

    #graph_statistics[ 'diameter' ].append( nx.diameter( G ) )
    #graph_statistics[ '1-shell' ].append( nx.k_shell(G, k=1).number_of_nodes( ) )
    return pd.DataFrame( data = [ graph_statistics ] )


def getClusteringMetrics( predicted_clusterings, labels_true, dataset_name = 'dataset', metric = 'accuracy' ):
    
    clustering_results = dict( )
    clustering_results[ 'name' ] = dataset_name
    
    for algo in predicted_clusterings.keys( ) :
        clustering_results[ algo ] = utils.computePartitionMetric( labels_true, predicted_clusterings[ algo ], metric = metric )
    
    return clustering_results #pd.DataFrame( data = [ clustering_results ] )


