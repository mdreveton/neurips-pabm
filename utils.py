#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 12:03:16 2024
"""


import networkx as nx
import numpy as np
from sklearn.metrics import accuracy_score
import scipy as sp
from itertools import combinations

from sklearn.metrics.cluster import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.metrics import matthews_corrcoef



def generateBernoulliAdjacency( P ):
    A = np.zeros( P.shape )
    for i in range( P.shape[0] ):
        for j in range( i ):
            A[i,j] = np.random.binomial( 1, P[i,j] )
            A[j,i] = A[i,j]
    G = nx.from_numpy_array( A )
    return nx.adjacency_matrix( G )


def generateP_of_homogeneousPABM( labels_true, p, q, theta_in, theta_out ):
    n = len( labels_true )
    
    P = np.zeros( (n,n ) )
    for i in range( n ):
        for j in range( i + 1 ):
            if labels_true[i] == labels_true[j]:
                P[i,j] = min( p * theta_in[i] * theta_in[j], 1 )
            else:
                P[i,j] = min( q * theta_out[i] * theta_out[ j ], 1 )
            P[j,i] = P[i,j]
    
    return P


def generateP_inhomogeneousPABM( sizes, Lambdas ):
    n = sum( sizes )
    n_clusters = len( sizes )
        
    labels_true = [ ]
    for community in range( n_clusters ):
        labels_true = labels_true + [ community+1 for i in range( sizes[community] ) ]
    labels_true = np.array( labels_true, dtype = int )

    P = np.zeros( (n,n ) )
    nodeListPerCommunity = obtain_community_lists( labels_true, n_clusters = len( sizes ) )
    
    for a in range( n_clusters ):
        for b in range( n_clusters ):
            dummy = np.array( [ Lambdas[ a ] [ b ] ] ).T #makes an array of size n_a times 1 where n_a is the number of nodes in cluster a
            P[ np.ix_( nodeListPerCommunity[a] , nodeListPerCommunity[b] ) ] = dummy @ dummy.T 
    
    P = np.where( P > 1, 1, P )
    return P


def degree_vector( A ):
    return A.sum( axis = 1 ).flatten()


def degree_matrix( A , power=1 ):
    """
    Parameters
    ----------
    A : TYPE
        DESCRIPTION.
    p : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    d = degree_vector( A )
    with np.errstate(divide='ignore'):
        d = d**power
    d[ np.isinf( d ) ] = 0
    #Construct sparse degree matrix
    n = A.shape[0]  #Number of points
    D = sp.sparse.spdiags( d, 0, n, n)

    return D.tocsr()


# =============================================================================
# FUNCTIONS TO COMPUTE ACCURACY AND RELATED METRICS BETWEEN CLUSTERINGS
# =============================================================================


from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix



def computePartitionMetric( labels_true, labels_pred, metric = 'accuracy' ):
    
    if metric == 'accuracy':
        return computeAccuracy( labels_true, labels_pred )
    elif metric == 'correlation coefficient' or metric == 'cc' :
        return correlation_coefficient( labels_true, labels_pred )
    elif metric.lower() == 'ari':
        return adjusted_rand_score( labels_true, labels_pred )
    elif metric.lower() == 'ami':
        return adjusted_mutual_info_score( labels_true, labels_pred )
    
    else:
        raise TypeError( 'This metric is not implemented' )
        
def correlation_coefficient( labels_true, labels_pred ):
    
    pair_true = [ ]
    pair_false = [ ]
    
    n = len( labels_true )
    for pair in combinations( [ i for i in range( n ) ], 2 ):
        i = pair[ 0 ]
        j = pair[ 1 ]
        if labels_true[ i ] == labels_true[ j ]:
            pair_true.append( 1 )
        else:
            pair_true.append( 0 )
            
        if labels_pred[ i ] == labels_pred[ j ]:
            pair_false.append( 1 )
        else:
            pair_false.append( 0 )
        
    return sp.stats.pearsonr( pair_true, pair_false ).statistic


def _make_cost_m(cm):
    s = np.max(cm)
    return (- cm + s)

def computeAccuracy( labels_true, labels_pred ):
    #Compute accuracy by finding the best permutation using Hungarian algorithm
    #See also https://smorbieu.gitlab.io/accuracy-from-classification-to-clustering-evaluation/
    #But be careful, a small difference in the format of the result of linear_assignment of scipy and sklearn
    cm = confusion_matrix( labels_true, labels_pred )
    indexes = linear_sum_assignment(_make_cost_m(cm))
    #js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    cm2 = cm[:, indexes[1] ]

    return np.trace( cm2 ) / np.sum( cm2 )



def obtain_community_lists( z, n_clusters = 0 ):
    
    if n_clusters == 0:
        n_clusters = len( set( list( z ) ) )
    
    if n_clusters < len( set( list( z ) ) ):
        raise TypeError( 'The number of cluster should be larger than the number of elements in z' )
        
    nodeList = [ [ ] for a in range( n_clusters ) ]
    for i in range( len( z ) ):
        nodeList[ z[i] - 1 ].append( i )
    
    return nodeList
        
    
    
def oneHotEncoding( z , n_clusters ):
    n = len( z )
    
    if len( set( list( z ) ) ) > n_clusters:
        raise TypeError( 'The number of cluster should be larger than the number of elements in z' )
    
    Z = np.zeros( (n, n_clusters ) )
    for i in range( n ):
        Z[ i, z[i]-1 ] = 1
    return Z