#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 19:40:39 2025
"""

import networkx as nx
import utils as utils
import numpy as np
import graphlearning as gl

from random import shuffle 

import sklearn as sklearn
from sklearn.decomposition import PCA
import time
import os




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



"""
from sklearn.datasets import load_digits
from sknetwork.utils import KNeighborsTransformer
def sklearnDatasest( dataset_name ):
    EMB_DIM = 32
    N_NEIGH = 10

    if dataset_name=='digits':
        digits = load_digits()
        pca = PCA(n_components=EMB_DIM)
        embedding = pca.fit_transform(digits.data)
        knn = KNeighborsTransformer(n_neighbors=N_NEIGH, n_jobs=-1, undirected=True)
        A = knn.fit_transform(embedding)
        return A

    else:
        raise NotImplementedError('Dataset not implemented')
"""

def getGraphLearningDatasets( dataset_name, metric = 'vae', n = 'all' ):
    """
    This function retrieve the MNIST, FashionMNIST and Cifar10 datasets
    using the same preprocessing as done in the graph-learning package
    https://pypi.org/project/graphlearning/
    """
    
    if dataset_name not in [ 'mnist', 'fashionmnist', 'cifar10' ]:
        raise TypeError( 'This dataset is not implemented in the graph learning library' )
        
    if metric == 'auto':
        if dataset_name == 'cifar10':
            metric = 'simclr'
        else:
            metric = 'vae'

    #Dataset filename
    dataFile = dataset_name.lower()+"_"+metric.lower()+".npz"
    labelsFile = dataset_name.lower()+"_labels.npz"

    #Full path to file
    data_dir = os.path.abspath(os.path.join(os.getcwd(),'datasets'))
    dataFile_path = os.path.join(data_dir, dataFile)
    labelsFile_path = os.path.join(data_dir, labelsFile)
    
    #Download labels file if needed
    if not os.path.exists(labelsFile_path):
        print('dataset labels does not exists, we will try to download them')
        urlpath = 'https://github.com/jwcalder/GraphLearning/raw/master/Data/'+labelsFile
        gl.utils.download_file(urlpath, labelsFile_path)

    if not os.path.exists(dataFile_path):
        print('dataset does not exists, we will try to download them')
        urlpath = 'http://www-users.math.umn.edu/~jwcalder/Data/'+dataFile
        gl.utils.download_file(urlpath, dataFile_path)

    M = np.load(dataFile_path,allow_pickle=True)
    data = M['data']
    
    M = np.load(labelsFile_path,allow_pickle=True)
    labels = M['labels']

#    data, labels = gl.datasets.load( dataset_name, metric = metric )

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

    A = sklearn.neighbors.kneighbors_graph( data, 12, mode = 'distance' )
    A = A + A.T
    
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
