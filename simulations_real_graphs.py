#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 15:47:24 2025
"""

import networkx as nx
import utils as utils
import pandas as pd
import numpy as np
import graphlearning as gl

import datasets as datasets
import clustering as clustering

import time



gml_datasets = [ 'citeseer', 'cora', 'liveJournal-top2', 'politicalBlogs' ]

gl_datasets = [ 'mnist', 'fashionmnist', 'cifar10' ]


"""
df_statistics = pd.DataFrame(  )

df_accuracy = pd.DataFrame( )
df_ami = pd.DataFrame( )
df_ari = pd.DataFrame( )
df_cc = pd.DataFrame( )

df_time = pd.DataFrame( )

df_cv = pd.DataFrame( )

for dataset_name in gml_datasets:
    
    print( dataset_name )
    A, labels_true = datasets.getRealGraph( dataset_name , n = 10000 )
    
    
    #G = readFromGml( dataset_name )
    #labels_true = get_communities( G, community_label = 'community' )
    n_clusters = len( set( labels_true ) )
    #A = nx.adjacency_matrix( G )

    graph_statistics = getGraphStatistics( nx.from_scipy_sparse_array( A ), n_clusters, dataset_name = dataset_name )
    df_statistics = pd.concat( [ df_statistics, graph_statistics ] )
    
    #predicted_clusterings = simulations.getClusterings( A, n_clusters ) 
    
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

    cv_results = cv.crossValidation_knownNumberOfClusters( A, n_clusters = n_clusters, epsilon = 0.15 )
    cv_results[ 'name' ] = dataset_name
    df_cv = pd.concat( [ df_cv, pd.DataFrame( [ cv_results ] ) ] )

    
df_accuracy.to_csv( 'gml_clusteringAccuracy.csv', index = False )
df_ami.to_csv( 'gml_clusteringAMI.csv', index = False )
df_ari.to_csv( 'gml_clusteringARI.csv', index = False )
df_cc.to_csv( 'gml_clusteringCorrelationCoefficient.csv', index = False )

df_statistics.to_csv( 'gml_graphStatistics.csv', index = False )
df_cv.to_csv( 'gml_crossValidation.csv', index = False )

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




"""

#   below is some old non-needed code

clustering_results = dict( )
clustering_results[ 'name' ] = [ ]
clustering_results[ 'SBM' ] = [ ]
clustering_results[ 'DCBM' ] = [ ]
clustering_results[ 'PABM' ] = [ ]
clustering_results[ 'OSC' ] = [ ]

graph_statistics = dict( )
graph_statistics[ 'name' ] = [ ]
graph_statistics[ 'n' ] = [ ]
graph_statistics[ 'E' ] = [ ]
graph_statistics[ 'k' ] = [ ]
graph_statistics[ 'average degree' ] = [ ]
graph_statistics[ 'diameter' ] = [ ]
graph_statistics[ '1-shell' ] = [ ]


facebook_datasets = getFacebook100DatasetNames( )

facebook_datasets = [ 'Caltech36', 'Rice31', 'Simmons81', 'Middlebury45', 'Brandeis99', 'Georgetown15' ]

for dataset_name in facebook_datasets:
    print( dataset_name )
    G = readFromFacebook100( dataset_name )
    G = preprocessingFacebook100( G )
    labels_true = get_communities( G, community_label = 'dormitory' )
    
    n_clusters = len( set( labels_true ) )
    print( n_clusters )
    if n_clusters > 0:
        A = nx.adjacency_matrix( G )
        
        graph_statistics = getGraphStatistics( G, n_clusters, dataset_name = '' )
        graph_statistics[ 'name' ].append( dataset_name )
    
        z_bm = clustering.spectralClustering_bm( A , n_clusters )
        z_dcbm = clustering.spectralClustering_dcbm( A , n_clusters )
        z_pabm = clustering.spectralClustering_pabm( A, n_clusters )
        z_osc = clustering.orthogonalSpectralClustering( A, n_clusters )
        
        clustering_results[ 'name' ].append( dataset_name )
        clustering_results[ 'SBM' ].append( utils.computeAccuracy( labels_true, z_bm ) )
        clustering_results[ 'DCBM' ].append( utils.computeAccuracy( labels_true, z_dcbm ) )
        clustering_results[ 'PABM' ].append( utils.computeAccuracy( labels_true, z_pabm ) )
        clustering_results[ 'OSC' ].append( utils.computeAccuracy( labels_true, z_osc ) )

        cv_results = cv.crossValidation_knownNumberOfClusters( A, n_clusters=n_clusters, epsilon=0.15)
        cross_validation[ 'name' ].append( dataset_name )
        cross_validation[ 'SBM' ].append( cv_results[ 'bm' ] )
        cross_validation[ 'DCBM' ].append( cv_results[ 'dcbm' ] )
        cross_validation[ 'PABM' ].append( cv_results[ 'pabm' ] )

    
#df_statistics = pd.DataFrame( data = graph_statistics )
df_accuracy = pd.DataFrame( data = clustering_results )
df_cv = pd.DataFrame( data = cross_validation )

df_accuracy.to_csv( 'facebook100_clusteringAccuracy.csv', index = False )
df_statistics.to_csv( 'facebook100_graphStatistics.csv', index = False )
df_cv.to_csv( 'facebook100_crossValidation.csv', index = False )


"""