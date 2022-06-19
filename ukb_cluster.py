#!/usr/bin/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import networkx as nx
import pickle

# FUNCTIONS
# remove clusters that are ridiculously large - they cause problems
def filter_clusters_by_size(clusters,minClusterSize,maxClusterSize):
    maxSize = maxClusterSize
    clusters3 = [x for x in clusters if len(x)>= minClusterSize ]
    clusters3 = np.array(sorted([sorted(list(x)) for x in clusters3]))
    clusterSizes = np.array([len(x) for x in clusters3])
    toobig = np.where(clusterSizes>maxSize) # four ... let's remove these.
    print('toobig',clusterSizes[toobig])
    print('cluster3 length ',len(clusters3))
    size_filtered_clusters = np.delete(clusters3,toobig)
    print('with big clusters removed: ', len(size_filtered_clusters))
    return size_filtered_clusters

# function to get slice of kinship dataframe where ID1 or ID2 is on list of eids
def kinship_for_ids(kinship_df,eidlist):
    has1 = kinship_df[kinship_df['ID1'].isin(eidlist)].index.values.tolist()
    has2 = kinship_df[kinship_df['ID2'].isin(eidlist)].index.values.tolist()

    hasboth = list(set( has1 + has2 ) )
    return kinship_df[kinship_df.index.isin(hasboth)]

# function to get list of pairs of eids (as tuples) from a kinship slice
def eidPairs_from_kinshipdf(kinship_df):
    eidPairs = []
    for i,r in kinship_df.iterrows():
        eidPairs.append( ( r['ID1'],r['ID2'] ) )
    #print(eidPairs)
    return eidPairs

# function to get cliques and graphs from a list of pairs of eids
# cliques? https://networkx.github.io/documentation/stable/reference/algorithms/clique.html
# more cliques: https://en.wikipedia.org/wiki/Clique_(graph_theory)
def eidPairs_to_cliques(eidPairs):
    G = nx.Graph()
    G.add_edges_from(eidPairs)
    return list(nx.find_cliques(G)), G

# function to draw a graph and print the cliques
def draw_graph(cliques,G):
    plt.subplot(111)
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()
    #print(len(cliques),cliques)

## networkx & graph to find maximal cliques in each cluster
# start with clusters3 = a list of lists
def get_graphed_clusters(size_filtered_clusters,kindf):
    graphed_clusters = [] # going to be a list of lists
    i = 1
    for eidlist in size_filtered_clusters:
    #     print('cluster',i)
    #     i+=1
        # get slice of kinship dataframe where ID1 or ID2 is on list of eids
        kinship_slice = kinship_for_ids(kindf,eidlist)

        # get list of pairs of eids (as tuples) from kinship slice
        eidPairs = eidPairs_from_kinshipdf(kinship_slice)

        # get cliques and graphs from a list of pairs of eids
        cliques, G = eidPairs_to_cliques(eidPairs)

        # add cliques to graphed_clusters
        graphed_clusters.extend(cliques)

    return graphed_clusters

# remove duplicate clusters and small clusters
def remove_duplicate_clusters(clusters):

    # sort each element within the list
    sorted_clusters = [(sorted(x)) for x in clusters]

    # convert each element to a string (needed before 'set')
    sorted_cluster_strings = []
    for c in sorted_clusters:
        c = [str(x) for x in c]
        s = ','.join(c)
        sorted_cluster_strings.append(s)
    print('after string conversion ',len(sorted_cluster_strings))

    # 'set' to get unique clusters
    uniq_clusters = list(set(sorted_cluster_strings))
    print('unique clusters ', len(uniq_clusters))

    return uniq_clusters

# 1. goal: make list of lists, all eids in all clusters
# 1a. function to get overall list of eids in a dataframe
def get_all_eids(kindf):
    id1 = kindf['ID1'].values.tolist()
    id2 = kindf['ID2'].values.tolist()
    return sorted(list(set(id1 + id2)))

# 1b. function to make list for all partners of a particular eid within a kindf
def get_partners_for_eid(eid,kindf):
    id2_partners = kindf[kindf['ID1']==eid]['ID2'].values.tolist()
    id1_partners = kindf[kindf['ID2']==eid]['ID1'].values.tolist()
    return sorted(list(set(id2_partners + id1_partners)))

def get_clusters(kindf):
    # get eids from kindf
    all_eids = get_all_eids(kindf)
    # for each eid
    clusters = [{11111111}] # a list of SETS, need to preload to enable iteration
    done = []
    i = 1
    for eid in all_eids:
        # make list of all partners for eid
        partners = get_partners_for_eid(eid,kindf)
        #print(i,len(testlist),eid,partners)
        i+=1

        # is this eid in one of the existing cluster SETS?
        for cluster in clusters:
            if eid in cluster:
                #print('saw eid')
                # add the partners to the existing cluster ...
                cluster.update(partners)
                #print(clusters)
                # done, can stop checking
                done.append(eid)
                break

        # are any of the partners of this eid in an existing cluster?
        if eid not in done:
            #print('looking at partners')
            for partner in partners:
                for cluster in clusters:
                    if partner in cluster:
                        #print('found a partner')
                        # add the partners to the existing cluster ...
                        cluster.update(partners)
                        #print(clusters)
                        # done, can stop checking
                        done.append(eid)
                        break

        if eid not in done:

                    #print('did not see anything')
                    # make a new SET with eid and all partners, and add it to clusters
                    clusters.append(set([eid] + partners))
                    done.append(eid)
                    #print(clusters)

    clusters = clusters[1:]
    return clusters

# Histogram of cluster sizes
def clusterHistogram(clust):
    clusterSizes = [x.count(',')+1 for x in clust]
    print('Max cluster size: ',np.max(clusterSizes))
    print('Min cluster size: ',np.min(clusterSizes))
    plt.figure(figsize = (10, 5))
    ax = plt.subplot(111)
    plt.hist(clusterSizes, np.arange(min(clusterSizes)-0.5, max(clusterSizes)+1.5))
    plt.ylabel('Number of clusters',fontsize=16)
    plt.xlabel('Size of cluster',fontsize = 16)
    xlab = np.arange(np.min(clusterSizes),np.max(clusterSizes)+1)
    ax.set_xticks(xlab)
    ax.set_xticklabels(xlab,fontsize=12)
    plt.show()
    return clusterSizes

def get_maximum_independent_set(G,numTrials):
    maxLength = 0
    for i in np.arange(numTrials):
            mvs = nx.maximal_independent_set(G)
            if len(mvs) > maxLength:
                max_mvs = mvs
                maxLength = len(mvs)
            #print('this: ', mvs, 'Max: ', max_mvs)
    return max_mvs
