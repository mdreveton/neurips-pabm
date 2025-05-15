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



gml_datasets = [ 'citeseer', 'cora', 'liveJournal-top2', 'politicalBlogs' ]
gl_datasets = [ 'mnist', 'fashionmnist', 'cifar10' ]


"""
# =============================================================================
# REPRODUCE RESULTS OF TABLE 1
# =============================================================================

#o reproduce, replace 

df_statistics = pd.DataFrame(  )

df_accuracy = pd.DataFrame( )
df_ami = pd.DataFrame( )
df_ari = pd.DataFrame( )
df_cc = pd.DataFrame( )

df_time = pd.DataFrame( )

for dataset_name in gl_datasets:
    
    print( dataset_name )
    A, labels_true = datasets.getRealGraph( dataset_name , n = 10000 )
    
    
    #G = readFromGml( dataset_name )
    #labels_true = get_communities( G, community_label = 'community' )
    n_clusters = len( set( labels_true ) )
    #A = nx.adjacency_matrix( G )

    graph_statistics = getGraphStatistics( nx.from_scipy_sparse_array( A ), n_clusters, dataset_name = dataset_name )
    df_statistics = pd.concat( [ df_statistics, graph_statistics ] )
        
    predicted_clusterings, time_execution = getClusterings( A, n_clusters )
    
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


def getClusterings( A, n_clusters, algorithms = ['bm', 'dcbm', 'pabm-k', 'pabm-2k', 'pabm-ksquared', 'osc', 'sklearn' ] ):
    
    time_execution = dict( )
    clusterings = dict( )

    for algorithm in algorithms:
        print( 'Running algorithm :' , algorithm )
        start_time = time.time( )
        z = clustering.graph_clustering( A , n_clusters , variant = algorithm )
        end_time = time.time( )
        
        clusterings[ algorithm ] = z
        time_execution[ algorithm ] = end_time - start_time
        print( 'Algorithm took ', end_time - start_time , ' seconds.' )
    
    #z_dcbm = clustering.spectralClustering_dcbm( A , n_clusters )
    #z_pabm_k = clustering.spectralClustering_pabm( A, n_clusters, infer_rank = n_clusters )
    #z_pabm_2k = clustering.spectralClustering_pabm( A, n_clusters, infer_rank = 2 * n_clusters )
    #z_pabm_ksquared = clustering.spectralClustering_pabm( A, n_clusters, infer_rank = n_clusters * n_clusters )
    #z_osc = clustering.orthogonalSpectralClustering( A, n_clusters )

    #return { 'bm' : z_bm, 'dcbm' : z_dcbm, 'pabm-k' : z_pabm_k, 'pabm-2k' : z_pabm_2k, 'pabm-ksquared' : z_pabm_ksquared, 'osc' : z_osc }

    return clusterings, time_execution
    


    


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


def getClusteringMetrics( predicted_clusterings, labels_true, dataset_name = '', metric = 'accuracy' ):
    
    clustering_results = dict( )
    clustering_results[ 'name' ] = dataset_name
    
    for algo in predicted_clusterings.keys( ) :
        clustering_results[ algo ] = utils.computePartitionMetric( labels_true, predicted_clusterings[ algo ], metric = metric )
    
    return pd.DataFrame( data = [ clustering_results ] )


