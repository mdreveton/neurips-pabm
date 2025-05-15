#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 19:40:39 2025
"""

import networkx as nx
import utils as utils
import pandas as pd
import numpy as np
import graphlearning as gl

from random import shuffle 

import sklearn as sklearn
from sklearn.decomposition import PCA
import time



gml_datasets = [ 'citeseer', 'cora', 'liveJournal-top2', 'politicalBlogs' ]


def getRealGraph( dataset_name , n = 'all' ):
    
    if dataset_name in gml_datasets:
        G = readFromGml( dataset_name )
        labels_true = get_communities( G, community_label = 'community' )
        return nx.adjacency_matrix( G ), labels_true
    
    
    elif dataset_name in [ 'mnist', 'fashionmnist', 'cifar10' ]:
        return getGraphLearningDatasets( dataset_name, metric = 'auto', n = n )
    
    else:
        raise TypeError( 'This dataset is not implemented' )




def getGraphLearningDatasets( dataset_name, metric = 'vae', n = 'all' ):
    """
    This function retrieve the MNIST, FashionMNIST and Cifar10 datasets
    using the graph-learning package
    """
    
    if dataset_name not in [ 'mnist', 'fashionmnist', 'cifar10' ]:
        raise TypeError( 'This dataset is not implemented in the graph learning library' )
        
    if metric == 'auto':
        if dataset_name == 'cifar10':
            metric = 'simclr'
        else:
            metric = 'vae'
            #metric = 'vae_old' #This is to use the same embedding as in their original ICML paper. Somehow the new embedding raise issues when computing k-nearest neighborhood with annoy 

    
    data, labels = gl.datasets.load( dataset_name, metric = metric )
    if n == 'all' or n > data.shape[0]:
        n = data.shape[0]
    else:
        indices_kept = [ i for i in range( n ) ]
        #indices_kept = [ i for i in range( data.shape[0] ) ]
        #shuffle( indices_kept )
        #indices_kept[:n]
        
        
        labels = labels[ indices_kept ]
        data = data[ indices_kept, : ]
        
    labels = labels + np.ones( n )
    labels = labels.astype( int )


    W = gl.weightmatrix.knn( data, k = 10, kernel = 'distance' )
    A = W.copy()
    A.data[ A.data > 0 ] = 1
    
    #A = sklearn.neighbors.kneighbors_graph( data, 10, mode = 'connectivity' )

    
    return A, labels.tolist( )



def readFromGml( dataset_name ):
    """
    This function returns the dataset that are saved in the datasets folder in a gml format
    This includes: dolphins, karateClub, politicalBlogs, liveJournal-top2, citeseer, cora 
    """
    
    if dataset_name not in gml_datasets:
        raise TypeError( 'The dataset is not in the gml file folder' )
    
    G = nx.read_gml( 'datasets/' + dataset_name + '.gml' , label = 'id' )
    G = nx.to_undirected( G )
    G = nx.Graph( G ) #to delete the multi edges
    G.remove_edges_from(nx.selfloop_edges(G)) #Delete self loops
    
    return G


def get_communities( G , community_label = 'community' ):
    
    communities = list( nx.get_node_attributes(G, community_label ).values() )
    
    #n_clusters = list( set( communities.values( ) ) )
    
    #The following lines will make sure that the communities are indexed from 0 to n_clusters
    new_indexes = dict( )
    start = 1
    for elt in set( communities ):
        new_indexes[ elt ] = start
        start += 1
    
    for i in range( len( communities ) ) :
        old_index = communities[ i ]
        communities[ i ] = new_indexes[ old_index ]
    
    return communities
