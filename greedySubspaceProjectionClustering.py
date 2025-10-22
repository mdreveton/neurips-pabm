#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 18:46:20 2025

@author: dreveton


This file provides a code for the algorithm that minimizes the function Q3 in the paper 
arXiv:2505.22459

The algorithm does not have a name, but I named it: 
Greedy Subspace Projection Clustering (GSPC)
"""

import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.cluster import SpectralClustering


def GreedySubspaceProjectionClustering(A, n_clusters, model = 'pabm', max_iter=300, verbose = False ):
    
    A.indices = A.indices.astype(np.int32, casting='same_kind')
    A.indptr = A.indptr.astype(np.int32, casting='same_kind') #For some reason, scikit-learn requires sparse matrix elements to be int32 and not int64
    sc = SpectralClustering( n_clusters = n_clusters, affinity = 'precomputed' )
    init_labels = sc.fit_predict( A )

    labels_pred, Q3 = greedy_optimize_Q3( A, init_labels, n_clusters, model = 'pabm', max_iter = max_iter, verbose = verbose )
    
    return labels_pred + np.ones( A.shape[0] )



def adjacency_spectral_embedding( A, embedding_dimension ):
    vals, vecs = eigsh(A, k = embedding_dimension, which='LM')
    return vecs  # shape (n, embedding_dimension)


def compute_projections( U, labels, n_clusters ):
    
    projections = { }
    
    for k in range( n_clusters ):
        Mk = U[labels == k ]
        Mk = Mk.T
        
        if len(Mk) == 0:
            projections[k] = None
        else:
            Uk, _, Vk = np.linalg.svd( Mk, full_matrices = True )
            leading_singularvectors = Uk[:,:n_clusters]
            projections[k] = leading_singularvectors @ leading_singularvectors.T
            
    return projections


def compute_Q3( U, labels, n_clusters ):
    
    total = 0.0
    projections = compute_projections(  U, labels, n_clusters )
    
    for i in range( U.shape[0]):
        proj = projections[ labels[ i ] ]
        
        if proj is not None:
            Ui = U[ i,: ]
            total += np.linalg.norm( Ui - proj @ Ui )**2
            
    return total


def greedy_optimize_Q3(A, init_labels, n_clusters, model = 'pabm', max_iter=300, verbose = False ):
    
    if model == 'pabm':
        embedding_dimension = n_clusters**2
    
    elif model == 'dcbm':
        embedding_dimension = n_clusters
    
    U = adjacency_spectral_embedding( A, embedding_dimension )
    old_labels = init_labels.copy( )
    
    for it in range(max_iter):
        
        projections = compute_projections( U, old_labels, n_clusters )
        changed = False
        new_labels = old_labels.copy()
        
        for i in range( A.shape[0] ):
            Ui = U[i,:]
            loss = [ ]
            
            for k in range( n_clusters ):
                Pik = projections[k]
                if Pik is None:
                    loss.append( np.inf )
                else:
                    loss.append( np.linalg.norm( Pik @ Ui - Ui ) )
            new_labels[ i ] = np.argmin( loss )
        
        if np.sum( new_labels != old_labels ) >= 1:
            changed = True
        if not changed:
            if verbose:
                print('The algorithm converged in ', it, ' iterations')
            break

        else:
            old_labels = new_labels
            
    return new_labels, compute_Q3( U, new_labels, n_clusters )



