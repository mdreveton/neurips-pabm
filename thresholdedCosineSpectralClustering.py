#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 15:23:34 2025

@author: dreveton
"""

import numpy as np
import scipy as sp
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from random import shuffle

import utils as utils


def RefinedThresholdedCosineSpectralClustering( A, n_clusters, number_eigenvectors = 'k-squared', number_refinements = 'auto', verbose = False ):
    
    if number_refinements == 'auto':
        #Original paper recommands 2 iteration of the refinement.
        #However, I noticed that 2 log (n) appears better (with a stopping criteria in case the algo converged earlier)
        #But my current implementation is not effective and doing log(n) iterations taks a lot of time.
        
        #number_refinements = int( 2 * np.log( A.shape[0] ) )
        number_refinements = 2
            
    
    z_old = ThresholdedCosineSpectralClustering( A, n_clusters, number_eigenvectors = number_eigenvectors )
    
    for t in utils.iteration( range( number_refinements ), verbose = verbose ):
        z_new = refinement_newVersion( A, n_clusters, z_old )
        if np.sum( z_old != z_new ) >= 1:
            z_old = z_new
        else:
            if verbose:
                print('The algorithm converged in ', t, ' iterations')
            break

    if number_refinements == 0:
        z_new = z_old
    
    return z_new
    


def ThresholdedCosineSpectralClustering( A, n_clusters, number_eigenvectors = 'k-squared', verbose = False ):
    """ Perform spectral clustering for a PABM
    Parameters
    ----------
    A : {array-like, sparse matrix} of shape (n, n)
        Adjacency matrix of a graph.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    z : vector of shape (n) with integer entries between 1 and n_clusters
        entries z_i denotes the cluster of vertex i.
    """
    
    n = A.shape[0]    
    
    if number_eigenvectors == 'k-squared':
        number_eigenvectors = n_clusters * n_clusters
    elif not isinstance(number_eigenvectors, int):
        number_eigenvectors = n_clusters * n_clusters
    if verbose:
        print( 'We will use ', number_eigenvectors, ' eigenvectors.' )
        
    vals, vecs = sp.sparse.linalg.eigsh( A.astype(float), k = number_eigenvectors, which = 'LM' )
    
    #tau = np.abs( np.cos( vecs @ vecs.T ) ) #This version is slightly different that in the original paper, yet it also works well in practice (on real datasets it even appears to be better)
    tau = np.abs( cosine_similarity(vecs) ) - np.eye(n) #taking out the diagonal seems to dramatically improve the performance. 
    hist, bin_edges = np.histogram(tau, bins = 'auto', density=True)
    difference = [ hist[t+1] - hist[t] for t in range( len(hist) - 1 ) ]
    
    threshold = bin_edges[ np.argmax( difference ) + 1 ]
    tau_thresholded = np.where(tau >= threshold, 1, 0)
    
    z = KMeans(n_clusters = n_clusters, n_init = 'auto' ).fit_predict( tau_thresholded ) + np.ones( n )
    
    return z.astype(int)


from scipy.sparse import issparse

def refinement_newVersion(A, n_clusters, z_old):
    #Is equivalent to refinement( ... ) but is faster
    
    # Pre-calculate boolean masks for communities
    community_masks = [(z_old == ell + 1) for ell in range(n_clusters)]

    A_per_cluster = [ ]
    for ell in range( n_clusters ):
        A_per_cluster.append( A[ :, community_masks[ell] ] )
        
    Abarre = dict( )
    for k in range( n_clusters ):
        n_k = np.sum( community_masks[ k ] ) #size of community k
        for ell in range( n_clusters ):
            if n_k != 0:
                Abarre[ (k,ell) ] = np.sum( A_per_cluster[ell][ community_masks[k] ], axis = 0) / n_k
            else:
                print( 'Cluster k in empty, this may be a problem')
                Abarre[ (k,ell) ] = 0 
        
    z_new = z_old.copy( )
    
    
    vertices = [ i for i in range( A.shape[ 0 ] )]
    shuffle( vertices )
    for i in vertices:
        proxy = np.zeros( n_clusters )
        
        A_i_ell_parts = [] # List to store sparse row vectors A[i, C_ell_mask]
        A_i_ell_norms = [] # List to store norms of these parts

        for ell in range(n_clusters):
            # Select columns corresponding to community 'ell' from the i-th row of A
            # A_i_current_ell = A_per_cluster[ell].getrow(i) 
            A_i_current_ell = A_per_cluster[ell][i,:] 
            A_i_ell_parts.append(A_i_current_ell)

            # Calculate the L2 norm. .toarray().flatten() ensures it's a dense 1D array for norm calculation
            if issparse(A_i_current_ell):
                A_i_ell_norms.append(np.linalg.norm(A_i_current_ell.toarray().flatten(), ord=2))
            else:
                A_i_ell_norms.append(np.linalg.norm(A_i_current_ell.flatten(), ord=2))
                
        # Iterate over possible new clusters 'k' for vertex 'i'
        for k in range(n_clusters):
            proxy_k_sum = 0.0 # Accumulator for proxy_k

            # Iterate over 'ell' for the sum
            for ell in range(n_clusters):
                # Abarre[(k,ell)] is a dense 1D numpy array
                abarre_k_ell_vec = Abarre[(k, ell)]

                # Calculate L2 norm of Abarre[(k,ell)]
                abarre_k_ell_norm = np.linalg.norm(abarre_k_ell_vec, ord=2)
                dot_product = A_i_ell_parts[ell].dot(abarre_k_ell_vec.T)
                if not isinstance(dot_product, float):
                    dot_product=dot_product[0,0]

                # Calculate product_norm
                product_norm_val = A_i_ell_norms[ell] * abarre_k_ell_norm

                if product_norm_val != 0:
                    proxy_k_sum += dot_product / product_norm_val
            
            proxy[k] = proxy_k_sum

            
        z_new[ i ] = np.argmax( proxy ) + 1 

    return z_new



def refinement( A, n_clusters, z_old ):
    
    A_per_cluster = [ ]
    for ell in range( n_clusters ):
        A_per_cluster.append( A[ :, z_old == ell + 1 ] )
        
    Abarre = dict( )
    for k in range( n_clusters ):
        C_k = z_old == k+1  #A bit unreadable but this gives a boolean array where element i is True iff z_old[i] = k+1
        n_k = np.sum( C_k ) #size of community k
        for ell in range( n_clusters ):
            if n_k != 0:
                Abarre[ (k,ell) ] = np.sum( A_per_cluster[ell][ z_old == k + 1 ], axis = 0) / n_k
            else:
                print( 'Cluster k in empty, this may be a problem')
                Abarre[ (k,ell) ] = 0 
        
    z_new = z_old.copy( )
    
    vertices = [ i for i in range( A.shape[ 0 ] )]
    shuffle( vertices )
    for i in vertices:
        proxy = [ ]
        for k in  range( n_clusters ):
            product_norm = [ np.linalg.norm( A_per_cluster[ell][i,:].todense() , ord = 2 ) * np.linalg.norm( Abarre[ (k,ell) ], ord = 2) for ell in range( n_clusters ) ]
            product_norm = np.asarray( product_norm )
            #product_norm = np.where(product_norm == 0, 1, product_norm )
            
            proxy_k = 0
            for ell in range( n_clusters ):
                if product_norm[ell] != 0:
                    prod =  A_per_cluster[ell][i,:] @ Abarre[ (k,ell) ].T
                    if isinstance(prod, float):
                        proxy_k += prod / product_norm[ell]
                    else:
                        proxy_k += prod[0,0] / product_norm[ell]
            proxy.append( proxy_k )
            
        z_new[ i ] = np.argmax( proxy ) + 1 

    """
    for i in range( A.shape[ 0 ] ):
        proxy = [ ]
        for k in  range( n_clusters ):
            proxy_k = 0 
            for ell in range( n_clusters ):
                product_norm = np.linalg.norm( A_per_cluster[ell][i,:].todense() , ord = 2 ) * np.linalg.norm( Abarre[ (k,ell) ], ord = 2)
                if product_norm != 0:
                    proxy_k += ( A_per_cluster[ell][i,:] @ Abarre[ (k,ell) ].T ) [0,0] / product_norm
                    #proxy_k.append( np.sum( [ A_per_cluster[ell][i,:].T @ Abarre[ (k,ell) ] / ( np.linalg.norm( A_per_cluster[ell][i,:].todense(), ord = 2 ) * np.linalg.norm( Abarre[ (k,ell) ], ord = 2)) ] ) )
            proxy.append( proxy_k )
        z_new[ i ] = np.argmax( proxy ) + 1 
    """
    
    return z_new
