#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 17:31:02 2024
"""

import numpy as np
import scipy as sp
from sklearn.cluster import KMeans, SpectralClustering

import utils as utils
import selfrepresentation as selfrepresentation


from tqdm import tqdm

"""
model = ElasticNetSubspaceClustering(n_clusters=3,algorithm='lasso_lars',gamma=50).fit(X.T)
print(model.labels_)

"""

# =============================================================================
# CLUSTERING: ALGO IMPLEMENTED
# =============================================================================

def graph_clustering( A, n_clusters, variant = 'bm' ):
    
    n = A.shape[ 0 ]
    variant = variant.lower() #To avoid issues with lower/upper-case letters
    
    if variant == 'bm' or variant == 'sbm':
        return spectralClustering_bm( A , n_clusters )
    
    elif variant == 'dcbm':
        if n <= 15000:
            return spectralClustering_dcbm(A, n_clusters, version = 'full')
        else:
            return spectralClustering_dcbm(A, n_clusters, version ='reduced' )
    
    elif variant == 'pabm' or variant == 'pabm-ksquared':
        return spectralClustering_pabm( A, n_clusters, number_eigenvectors = n_clusters * n_clusters )

    elif variant == 'pabm-k':
        return spectralClustering_pabm( A, n_clusters, number_eigenvectors = n_clusters )

    elif variant == 'pabm-2k':
        return spectralClustering_pabm( A, n_clusters, number_eigenvectors = 2 * n_clusters )

        
    elif variant == 'osc':
        return orthogonalSpectralClustering(A, n_clusters)
    
    elif variant == 'scikit-learn' or variant == 'sklearn' or variant == 'normalized laplacian':
        sc = SpectralClustering( n_clusters = n_clusters, affinity = 'precomputed' )
        A.indices = A.indices.astype(np.int32, casting='same_kind')
        A.indptr = A.indptr.astype(np.int32, casting='same_kind')
        z = sc.fit_predict( A ) + np.ones( n )
        return z.astype( int )

    
    return TypeError( 'The algorithm is not implemented' )



# =============================================================================
# SPECTRAL CLUSTERING: VARIOUS FORMS
# =============================================================================


def spectralClustering_bm( A , n_clusters ):
    """ Perform spectral clustering for a SBM
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
    vals, vecs = sp.sparse.linalg.eigsh( A.astype(float), k = n_clusters, which = 'LM' )
    hatP = vecs @ np.diag( vals ) #@ vecs.T #Note: k-means on vecs @ np.diag( vals ) and on vecs @ np.diag( vals ) @ vecs.T is equivalent, but faster using vecs @ np.diag( vals )  (n-by-n_clusters matrix instead of n-by-n)
    z = KMeans( n_clusters = n_clusters, n_init = 'auto', max_iter = max(300, n ) ).fit_predict( hatP ) + np.ones( n ) #Somehow less iterations in k-means sometimes degrade the performance. But n_iter = n is way too much if n is large. 
    
    return z.astype(int) 


def spectralClustering_dcbm( A , n_clusters, version ='full' ):
    """ Perform spectral clustering for a DCBM
    Parameters
    ----------
    A : {array-like, sparse matrix} of shape (n, n)
        Adjacency matrix of a graph.
    n_clusters : int
        Number of clusters.
        
    version : optional
        full: Algorithm 1 in  
            Community detection in degree-corrected block models
            Chao Gao, Zongming Ma, Anderson Y. Zhang, Harrison H. Zhou
            Ann. Statist. 46(5): 2153-2185 (October 2018). DOI: 10.1214/17-AOS1615
        
        reduced: Another version of normalization of the embedding
        
    The full version has O(n^2) space complexity and thus is inadapted to large (say n above 10,000) graphs.


    Returns
    -------
    z : vector of shape (n) with integer entries between 1 and n_clusters
        entries z_i denotes the cluster of vertex i.
    """
    n = A.shape[0]
    vals, vecs = sp.sparse.linalg.eigsh( A.astype(float), k = n_clusters, which = 'LM' )

    if version == 'full':
        hatP = vecs @ np.diag( vals ) @ vecs.T
    elif version == 'reduced':
        hatP = vecs @ np.diag( vals )
    else:
        raise TypeError('The dcbm version is not implemented. I will run the full version.' )
        hatP = vecs @ np.diag( vals ) @ vecs.T
    
    hatP_rowNormalized = hatP
    for i in range( n ):
        if np.linalg.norm( hatP[i,:], ord = 1) != 0:
            hatP_rowNormalized[i,:] = hatP[i,:] / np.linalg.norm( hatP[i,:], ord = 1)
        
    z = KMeans(n_clusters = n_clusters, n_init = 'auto' ).fit_predict( hatP_rowNormalized ) + np.ones( n )        
    
    return z.astype(int) 


def spectralClustering_pabm( A, n_clusters, version = 'subspace', number_eigenvectors = 'k-squared' ):
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
    print( 'We will use ', number_eigenvectors, ' eigenvectors.' )
        
    vals, vecs = sp.sparse.linalg.eigsh( A.astype(float), k = number_eigenvectors, which = 'LM' )
    
    if n > 15000 and version == 'subspace':
        print( 'ElasticNetSubspaceClustering  may take too long for large dataset. We use subspace SparseSubspaceClusteringOMP instead.' )
        version = 'subspace-omp'
    
    if version == 'spherical':
        hatP = vecs @ np.diag( vals ) @ vecs.T
        hatP_rowNormalized = hatP
        for i in range( n ):
            if np.linalg.norm( hatP[i,:], ord = 1) != 0:
                hatP_rowNormalized[i,:] = hatP[i,:] / np.linalg.norm( hatP[i,:], ord = 1 )
            else:
                hatP_rowNormalized[i,:] = 1/n * np.ones( n )
        
        #z = KMeans(n_clusters = n_clusters, n_init = 'auto' ).fit_predict( hatP_rowNormalized ) + np.ones( n )
        model = selfrepresentation.ElasticNetSubspaceClustering( n_clusters = n_clusters ,algorithm = 'lasso_lars',gamma=50 ).fit( hatP_rowNormalized )
        z = model.labels_ + np.ones( n )

    elif version == 'kmeans':
        z = KMeans(n_clusters = n_clusters, n_init = 'auto' ).fit_predict( vecs @ np.diag( vals ) ) + np.ones( n )
    
    elif version == 'subspace':
        #model_ensc = selfrepresentation.ElasticNetSubspaceClustering(n_clusters=10,affinity='nearest_neighbors',algorithm='spams',active_support=True,gamma=200,tau=0.9)
        #model = selfrepresentation.ElasticNetSubspaceClustering( n_clusters = n_clusters ,algorithm = 'lasso_lars',gamma=50 ).fit( vecs @ np.diag( vals ) @ vecs.T )
        model = selfrepresentation.ElasticNetSubspaceClustering( n_clusters = n_clusters ,algorithm = 'lasso_lars',gamma=50 ).fit( vecs @ np.diag( vals ) )
        z = model.labels_ + np.ones( n )

    elif version == 'subspace-omp':
        model = selfrepresentation.SparseSubspaceClusteringOMP( n_clusters = n_clusters, thr=1e-5 ).fit( vecs @ np.diag( vals ) )
        z = model.labels_ + np.ones( n )
    
    return z.astype(int)


def subspaceClustering( A, n_clusters ):
    
    model = selfrepresentation.SparseSubspaceClusteringOMP( n_clusters = n_clusters, n_nonzero = n_clusters*n_clusters,thr=1e-5 ).fit( A )
    z = model.labels_ + np.ones( A.shape[0] )

    return z.astype(int) 



def orthogonalSpectralClustering( A, n_clusters, infer_rank = False ):
    """Perform Orthogonal Spectral Clustering. 
    See Algorithm 1 of : 
    John Koo, Minh Tang, and Michael W. Trosset. "Popularity adjusted block models are generalized random dot product graphs." Journal of Computational and Graphical Statistics 32.1 (2023): 131-144.
    
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

    vals, vecs = sp.sparse.linalg.eigsh( A.astype(float), k = n_clusters * n_clusters, which = 'BE' )
    #hatP = vecs @ np.diag( vals ) @ vecs.T
    
    if infer_rank:
        density = np.sum(A)/(n**2)
        threshold_eigenvalue = np.sqrt( n * density ) * 2 #* np.log( n * density )
        vals_ = vals[ np.abs(vals) > threshold_eigenvalue ]
        #print( len( vals_ ) )
        vecs_ = vecs[ :, np.abs(vals) > threshold_eigenvalue ]
        B = np.sqrt( n ) * vecs_ @ vecs_.T
    else:
        B = np.sqrt( n ) * vecs @ vecs.T
    
    clustering_ = SpectralClustering( n_clusters = n_clusters, affinity='precomputed').fit( np.abs(B) )
    z = clustering_.labels_ + np.ones( n )

    return z.astype(int) 


import math as math

def fast_spectral_cluster( A, n_clusters: int):
    """
    This is a faster spectral clustering, from 
    Peter Macgregor. "Fast and simple spectral clustering in theory and practice." Advances in Neural Information Processing Systems 36 (2023): 34410-34425.
    
    Implementation copy/pasted and adapted from 
    https://github.com/pmacg/fast-spectral-clustering 
    """
    n = A.shape[ 0 ]
    
    l = min( n_clusters, math.ceil( math.log( n_clusters, 2) ) )
    t = 10 * math.ceil( math.log( n / n_clusters, 2 ) )
    
    #M = g.normalised_signless_laplacian()
    Dhalf = utils.degree_matrix( A, power = -1/2 )
    M = sp.sparse.identity( n ) - Dhalf @ A @ Dhalf
    Y = np.random.normal( size = (n,l) )

    # We know the top eigenvector of the normalised laplacian.
    # It doesn't help with clustering, so we will project our power method to
    # be orthogonal to it.
    
    top_eigvec = np.sqrt( utils.degree_matrix(A) @ np.full( (n,), 1) )
    norm = np.linalg.norm(top_eigvec)
    if norm > 0:
        top_eigvec /= norm

    for _ in range(t):
        Y = M @ Y

        # Project Y to be orthogonal to the top eigenvector
        for i in range(l):
            Y[:, i] -= (top_eigvec.transpose() @ Y[:, i]) * top_eigvec

    kmeans = KMeans(n_clusters = n_clusters, n_init='auto')
    kmeans.fit( Y )
    z = kmeans.labels_ + np.ones( n )
    
    return z.astype(int) 



# =============================================================================
# ESTIMATE PARAMETERS OF VARIOUS BLOCK MODELS (FROM A GIVEN CLUSTERING)
# =============================================================================


def estimate_bm( A, n_clusters, z ):
    node_in_each_clusters = utils.obtain_community_lists( z, n_clusters = n_clusters )
    B = estimateRateMatrix_bm( A, n_clusters, node_in_each_clusters )
    Z = utils.oneHotEncoding( z, n_clusters = n_clusters )
    P_bm = Z @ B @ Z.T

    return P_bm

def estimate_dcbm( A, n_clusters, z ):
    node_in_each_clusters = utils.obtain_community_lists( z, n_clusters = n_clusters )
    edge_count = edge_count_between_communities( A, n_clusters, node_in_each_clusters )
    theta_hat = estimate_theta_dcbm( A, z, edge_count )                
    Z = utils.oneHotEncoding( z, n_clusters )
    P_dcbm = np.diag( theta_hat ) @ Z @ edge_count @ Z.T @ np.diag( theta_hat )

    return P_dcbm


def estimate_pabm( A, n_clusters, z, lambda_hat = None ):
    
    if lambda_hat == None:
        lambda_hat = estimate_lambdas_pabm( A, n_clusters, z )
    
    n = A.shape[0]
    P_pabm = np.zeros( (n,n) )
    for i in range( n ):
        for j in range( n ):
            P_pabm[i,j] = lambda_hat[ i, z[j]-1 ] * lambda_hat[ j, z[i]-1 ]

    return P_pabm


# =============================================================================
# CLUSTERING VARIOUS BLOCK MODELS
# =============================================================================


def clustering_bm( A, n_clusters, n_iter = 10 ):
    
    z = spectralClustering_bm( A, n_clusters )
    
    for iteration in tqdm( range(n_iter ) ):
        z, B = likelihoodImprovement_bm( A, n_clusters, z )
    
    if n_iter == 0:
        P_hat = estimate_bm( A, n_clusters, z )
    else: 
        Z = utils.oneHotEncoding( z, n_clusters = n_clusters )
        P_hat = Z @ B @ Z.T
    
    return z.astype(int), P_hat


def clustering_dcbm( A, n_clusters, n_iter = 10 ):
    
    z = spectralClustering_dcbm( A, n_clusters )
    for iteration in tqdm( range( n_iter ) ):
        z, P_hat = likelihoodImprovement_dcbm( A, n_clusters, z )
    
    if n_iter == 0:
        P_hat = estimate_dcbm( A, n_clusters, z )
        
    return z.astype(int), P_hat


def clustering_pabm( A, n_clusters, n_iter = 10 ):
    
    z = spectralClustering_pabm( A, n_clusters )
    #z = spectralClustering_dcbm( A, n_clusters )
    #z = orthogonalSpectralClustering( A, n_clusters )
    n = A.shape[ 0 ]
    
    if n_iter >= 1:
        neighbor_list = [ ]
        for i in range( n ):
            neigh_i = list( A[[i],:].nonzero()[1] )
            neighbor_list.append( neigh_i )

    for iteration in tqdm( range( n_iter ) ):
        z, lambda_hat = likelihoodImprovement_pabm( A, n_clusters, z, neighbor_list = neighbor_list )
        
    if n_iter == 0:
        lambda_hat = estimate_lambdas_pabm( A, n_clusters, z )
    
    n = A.shape[0]
    P_hat = np.zeros( (n,n) )
    for i in range( n ):
        for j in range( n ):
            P_hat[i,j] = lambda_hat[ i, z[j]-1 ] * lambda_hat[ j, z[i]-1 ]

    return z.astype(int), P_hat





# =============================================================================
# BM: LIKELIHOOD IMPROVEMENT AND PARAMETER ESTIMATIONS
# =============================================================================

def estimateRateMatrix_bm( A, n_clusters, node_in_each_clusters ):
    
    B = np.zeros( (n_clusters, n_clusters) )
    n = A.shape[0]
    
    for a in range( n_clusters ):
        for b in range( a + 1 ):
            dummy = A[ node_in_each_clusters[ a ], : ]
            dummy = dummy[ :, node_in_each_clusters[ b ] ]
            if a != b:                
                normalisation = len( node_in_each_clusters[ a ] ) * len( node_in_each_clusters[ b ] )
            else:
                normalisation = len( node_in_each_clusters[ a ] ) * ( len( node_in_each_clusters[ a ] ) - 1 )
            
            if normalisation != 0:
                B[ a, b ] = np.sum( dummy ) / normalisation
            else:
                B[ a, b ] = 1 / n
            
            B[ b, a ] = B[ a, b ]
    
    return B


def number_neighbors_in_each_community( A, n_clusters, z, node_in_each_clusters ):
    n = A.shape[0]
    number_neighbors = np.zeros( ( n, n_clusters ) )
    for a in range(n_clusters):
        dummy = A[:,node_in_each_clusters[a] ]
        for i in range( n ):
            number_neighbors[i,a] = np.sum( dummy[ [i], : ] )

    return number_neighbors


def likelihoodImprovement_bm( A, n_clusters, z_init ):
    
    n = A.shape[0]
    
    z = np.zeros( n )
    
    node_in_each_clusters = utils.obtain_community_lists( z_init, n_clusters = n_clusters )
    B = estimateRateMatrix_bm( A, n_clusters, node_in_each_clusters )
    
    number_neighbors = number_neighbors_in_each_community( A, n_clusters, z, node_in_each_clusters ) 
    
    for i in range( n ):
        Li = np.zeros( n_clusters )
        for a in range( n_clusters ):
            Li[ a ] = np.sum( [ number_neighbors[i,b] * np.log( max( B[a,b], 1/n ) ) for b in range( n_clusters ) ] ) 
            Li[a] += np.sum( [ ( len( node_in_each_clusters[b] ) - number_neighbors[i,b] ) * np.log( max( 1-B[a,b], 1/n ) ) for b in range( n_clusters ) ] ) #Here max to avoid cases where B[a,b] = 1 (should not happen anyway)
        z[ i ] = np.argmax( Li ) + 1
    
    return z.astype(int), B



# =============================================================================
# DC-BM: LIKELIHOOD IMPROVEMENT AND PARAMETER ESTIMATIONS
# =============================================================================

def edge_count_between_communities( A, n_clusters, node_in_each_clusters ):
    edge_count = np.zeros( ( n_clusters, n_clusters ) )
    for a in range( n_clusters ):
        dummy = A[:,node_in_each_clusters[a] ]
        for b in range( a+1 ):
            edge_count[a,b] = dummy[node_in_each_clusters[b],:].sum()
            edge_count[b,a] = edge_count[a,b]
            
    return edge_count


def estimate_theta_dcbm( A, z, edge_count = None ):
    
    if edge_count is None:
        n_clusters = len( set(z) )
        node_in_each_clusters = utils.obtain_community_lists( z, n_clusters = n_clusters )
        edge_count = edge_count_between_communities( A, n_clusters, node_in_each_clusters )

    n = A.shape[ 0 ]
    theta_hat = np.zeros( n )
    for i in range( n ):
        if np.sum( edge_count[ z[i] - 1, : ] ) != 0:
            theta_hat[ i ] = A[[i], : ].sum() / np.sum( edge_count[ z[i] - 1, : ] )
        else:
            theta_hat[ i ] = 1/n
    return theta_hat


def likelihoodImprovement_dcbm( A, n_clusters, z_init, tol = 0.0000000001 ):
    
    n = A.shape[0]
    z = np.zeros( n )
    
    node_in_each_clusters = utils.obtain_community_lists( z_init, n_clusters = n_clusters )

    number_neighbors = number_neighbors_in_each_community( A, n_clusters, z_init, node_in_each_clusters )
    edge_count = edge_count_between_communities( A, n_clusters, node_in_each_clusters )
    theta_hat = estimate_theta_dcbm( A, z_init, edge_count )            
    
    Z = utils.oneHotEncoding( z_init, n_clusters )
    P_hat = np.diag( theta_hat ) @ Z @ edge_count @ Z.T @ np.diag( theta_hat )
    
    for i in range( n ):
        Li = np.zeros( n_clusters )
        for a in range( n_clusters ):
            degree_i = np.sum( number_neighbors[ i, : ] )
            k_a = np.sum( edge_count[ a, : ] ) + 1 / n
            Li[ a ]= np.sum( [ number_neighbors[ i, b ] * np.log( max( edge_count[ a, b ] / k_a, 1/n ) ) - degree_i / k_a * edge_count[ a, b ] for b in range( n_clusters ) ] ) 
            #Li[a] = np.sum( [ A[i,j] * np.log( min( 1, theta_hat[i] * theta_hat[j] * edge_count[ a, z_init[j] - 1 ] + tol ) ) + (1-A[i,j]) * np.log( max( tol, 1 - theta_hat[i] * theta_hat[j] * edge_count[ a, z_init[j]-1 ] ) ) for j in range( n ) ] )
        z[ i ] = np.argmax( Li ) + 1
    
    return z.astype(int), P_hat



# =============================================================================
# PA-BM: LIKELIHOOD IMPROVEMENT AND PARAMETER ESTIMATIONS
# =============================================================================


def estimate_lambdas_pabm( A, n_clusters, z, edge_count = None, number_neighbors = None ):
    
    if edge_count == None or number_neighbors == None:
        node_in_each_clusters = utils.obtain_community_lists( z, n_clusters = n_clusters )
        edge_count = edge_count_between_communities( A, n_clusters, node_in_each_clusters )
        number_neighbors = number_neighbors_in_each_community( A, n_clusters, z, node_in_each_clusters ) 

    n = A.shape[ 0 ]
    lambda_hat = np.zeros( ( n, n_clusters ) )
    for i in range( n ):
        for a in range( n_clusters ):
            if edge_count[ a, z[ i ] - 1 ] != 0:                
                lambda_hat[ i, a ] = min( number_neighbors[ i, a ] / np.sqrt( edge_count[ a, z[ i ] - 1 ] ), 1 )
                #TODO: I ADDED A MIN WITH 1 AS THE VALUE MIGHT EXCEED 1 IN SOME CASES
            else:
                lambda_hat[ i, a ] = 1/n
    return lambda_hat


def likelihoodImprovement_pabm( A, n_clusters, z_old, tol = 0.0000000001, neighbor_list = None ):

    n = A.shape[0]
    z_new = np.zeros( n )
    
    node_in_each_clusters = utils.obtain_community_lists( z_old, n_clusters = n_clusters )
    number_neighbors = number_neighbors_in_each_community( A, n_clusters, z_old, node_in_each_clusters ) 
    edge_count = edge_count_between_communities( A, n_clusters, node_in_each_clusters )

    lambda_hat = np.zeros( ( n, n_clusters ) )
    for i in range( n ):
        for a in range( n_clusters ):
            if edge_count[ a, z_old[ i ] - 1 ] != 0:                
                lambda_hat[ i, a ] = number_neighbors[ i, a ] / np.sqrt( edge_count[ a, z_old[ i ] - 1 ] )
            else:
                lambda_hat[ i, a ] = 1/n
    
    if neighbor_list == None:
        neighbor_list = [ ]
        for i in range( n ):
            neigh_i = list( A[[i],:].nonzero()[1] )
            neighbor_list.append( neigh_i )
    
    for i in range( n ):
        Li = np.zeros( n_clusters )
        for a in range( n_clusters ):
            Li[ a ] = - 1/2 * np.sum( [ number_neighbors[ i, b] * np.log( max( 1/n, edge_count[a,b] ) ) for b in range( n_clusters ) ] )
            Li[ a ] += np.sum( [ np.log( max(1/n, number_neighbors[ j, a ] ) ) for j in neighbor_list[ i ] ] )
            #Li[ a ] = np.sum ( [ number_neighbors[ i, b ] * np.log ( number_neighbors[ i, b ] / np.sqrt( edge_count[ b, a ] ) ) for b in range( n_clusters ) ] )
            #Li[ a ] = Li[ a ] - 1/2 * np.sum( [ number_neighbors[ i, z_old[j]-1 ] * number_neighbors[ j, a ] /  edge_count[ a, z_old[j]-1 ]  for j in range( n ) ] )
            # Li[ a ] = np.sum( [ A[i,j] * np.log( min(1, lambda_hat[ i, z_old[j]-1 ] * lambda_hat[j,a] + tol ) ) + (1-A[i,j]) * np.log( max(tol, 1 - lambda_hat[ i, z_old[j]-1 ] * lambda_hat[ j,a ] ) ) for j in range( n ) ] ) # THIS ONE WORKS BUT SUPER SLOW
            #Li[ a ] = - 1/2 * np.sum( [ number_neighbors[ i, b ] * np.log( max( edge_count[ a, b ], 1/n ) ) for b in range( n_clusters ) ] )
            #Li[ a ] = -1/2 * np.sum( [ lambda_hat[ i, z_old[j]-1 ] * lambda_hat[ j, a ] for j in range( n ) ] )
            # Li[ a ] = np.sum( [ A[i,j] * np.log( max( 1/n, min( 1, lambda_hat[i,z_old[j]-1] * lambda_hat[j,a] ) ) ) - min( 1, lambda_hat[i,z_old[j]-1] * lambda_hat[j,a] ) for j in range( n ) ] ) # THIS ONE WORKS BUT IS SLOW
            #Li[ a ] = np.sum( [ - lambda_hat[ i, z_old[j]-1 ] * lambda_hat[j,a] for j in range( n ) ] )
            #Li[ a ] = Li[ a ] + np.sum( [ np.log( lambda_hat[ j, a ] + tol ) for j in neigh_i ] )
        z_new[ i ] = np.argmax( Li ) + 1
    
    return z_new.astype(int), lambda_hat




# =============================================================================
# POSTERIOR LIKELIHOODS
# =============================================================================


def integratedCompleteLikelihood( A, z , model ):
    n = A.shape[ 0 ]
    n_clusters = len( set(z) )
    icl = 0
    if model == 'pabm':
        lambda_hat = estimate_lambdas_pabm( A, n_clusters, z, edge_count = None, number_neighbors = None )
        for i in range(n):
            for j in range(i):
                if A[i,j] == 1:
                    icl += np.log( max( 1/n, lambda_hat[ i,z[j]-1 ] * lambda_hat[ j,z[i]-1 ] ) )
                else:
                    icl += np.log( max( 1/n, 1 - lambda_hat[ i,z[j]-1 ] * lambda_hat[ j,z[i]-1 ] ) )
    if model == 'dcbm':
        theta_hat = estimate_theta_dcbm( A, z, edge_count = None )
        for i in range(n):
            for j in range(i):
                if A[i,j] == 1:
                    icl += np.log( max( 1/n, theta_hat[ i ] * theta_hat[ j ] ) )
                else:
                    icl += np.log( max( 1/n, 1 - theta_hat[ i ] * theta_hat[ j ] ) )
    if model == 'sbm':
        n_clusters = len( set(z) )
        node_in_each_clusters = utils.obtain_community_lists( z, n_clusters = n_clusters )
        B = estimateRateMatrix_bm( A, n_clusters, node_in_each_clusters )
        for i in range(n):
            for j in range(i):
                if A[i,j] == 1:
                    icl += np.log( max( 1/n, B[ z[i]-1, z[j]-1 ] ) )
                else:
                    icl += np.log( max( 1/n, 1 - B[ z[i]-1, z[j]-1 ] ) )
        
        
    return icl
