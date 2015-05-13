import math
import operator
import numpy as np
from numpy.random import random
np.set_printoptions(threshold=np.nan)
from scipy.sparse import *
from collections import Counter

# Global count variables to track no. of iterations till convergance
count = 1
successfulCount = 0

# Method to show main menu
def menu():
    method = 0
    while method < 1 or method > 4:
        print "(K = Number of Clusters)"
        print "Please enter one of the following options:"
        print "1. Perform K-Means clustering using cluster means as cluster centre."
        print "2. Perform K-Means clustering using closest insatnce to mean as cluster centre."
        print "3. Automate option 1 for K = 2->20 with 5 iterations on each value of K."
        print "4. Automate option 2 for K = 2->20 with 5 iterations on each value of K."

        # Read input of clusters wanted and method choice
        method = int(raw_input())

    if method == 1 or method == 2:
        K = int(raw_input("\nHow many clusters would you like? "))
        process(K, method)
    elif method == 3 or method == 4:
        store = []
        for K in range(2,21):
            precision, recall, f_score = 0, 0, 0
            for i in range(5):
                temp_p, temp_r, temp_f = process(K, method)
                precision += temp_p
                recall += temp_r
                f_score += temp_f

                # Reset global counters
                global count
                global successfulCount
                count = 1
                successfulCount = 0

            precision = float(precision) / 5
            recall = float(recall) / 5
            f_score = float(f_score) / 5

            # Store scores in an array
            store.append([precision, recall, f_score])

        print "\nCluster, Precision, Recall, F_Score"
        for i in range(19):
            print "%i, %f, %f, %f" % (i+2, store[i][0], store[i][1], store[i][2])


# Main processing method
def process(K, method):
    featspace, feat_vects, f_labels = get_feats("./data/data.txt")
    D = len(featspace)
    print "\nDimensionality of the feature space:", D
    numFV = len(feat_vects.getnnz(axis=1)) # number of feat_vects

    # L2 Normalize feature vectors
    l2_feat_vects = feat_vects.copy()
    for i in range(numFV):
        l2_feat_vects.data[l2_feat_vects.indptr[i]:l2_feat_vects.indptr[i+1]] = 1/np.sqrt(l2_feat_vects[i].sum())

    # Randomly assign centroids
    centroids = rand(K, D, density=0.1, format='csr')
    # Since this generates between 0 and 1, convert all > 0 to 1
    centroids.data[:] = 1

    # L2 Normalize centroids
    for i in range(K):
        centroids.data[centroids.indptr[i]:centroids.indptr[i+1]] = 1/np.sqrt(centroids[i].sum())

    print "OUTPUT FORMAT - Cluster totals: [(Cluster number, Total items in cluster)]\n"
    cluster_totals = -1
    global count
    # Main loop, max iterations 2000 to prevent infinite loop
    while count < 2000:
        # Assign each feature vector to a cluster
        sorted_labels, cluster_totals = assignClusters(numFV, K, l2_feat_vects, centroids, cluster_totals, f_labels)

        # If the cluster totals have converged and remained the same for 5 iterations, break
        if successfulCount > 4:
            break

        if method == 1 or method == 3:
            # Move the cluster centroids to the mean of the feature vectors within them
            centroids = getNewCentroids(K, D, sorted_labels, cluster_totals, l2_feat_vects)
        elif method == 2 or method == 4:
            # Move the cluster centroids to the closest instance to the mean of the feature vectors within them
            centroids = getNewCentroidsClosestInstance(K, D, sorted_labels, cluster_totals, feat_vects, l2_feat_vects)
        count += 1

    print "Clustering complete.\n"

    # Evaluate and calc precision, recall, f-score
    return evaluate(K, sorted_labels, cluster_totals, f_labels)


# Method to determine the feature space from some input files
def get_feats(fname):
    feats = {}
    temp = []
    pointers = [0]
    indices = []
    f_labels = {}
    iteration = 0
    with open(fname) as file:
        for line in file:
            f_labels[iteration] = line.strip().split()[0]
            iteration += 1
            line = line.strip().split()[1:]
            for w in line:
                index = feats.setdefault(w, len(feats))
                indices.append(index)
                temp.append(1)

            pointers.append(len(indices))

    feat_vects = csr_matrix((temp, indices, pointers), dtype=float)

    return feats, feat_vects, f_labels

    f_labels = {}
    for (val, label) in enumerate(featspace):
        feat_index[feat_val] = feat_id
        weights[feat_val] = 0


# Method to assign cluster labels to feature vectors
def assignClusters(numFV, K, l2_feat_vects, centroids, cluster_totals, f_labels):

    global count
    global successfulCount
    oldCT = cluster_totals

    # Assign features to a centroids
    c_labels = {}
    for i in range(numFV):
        # Get distance to each centroid and then label with the minimum
        distances = np.sqrt(((l2_feat_vects[i].toarray() - centroids.toarray())**2).sum(axis=1))
        c_labels.setdefault(i, len(c_labels))
        c_labels[i] = np.argmin(distances)


    # Sort the feat_vects by label and calculate total in each cluster
    sorted_labels = sorted(c_labels.items(), key=operator.itemgetter(1))
    cluster_totals = []
    for i in range(K):
        counter = 0
        for elem in sorted_labels:
            if elem[1] == i:
                counter = counter + 1
        cluster_totals.append((i,counter))

    if oldCT == cluster_totals:
        print "%i. Cluster Totals: %s SAME" % (count, cluster_totals)
        successfulCount += 1
    else:
        print "%i. Cluster Totals: %s" % (count, cluster_totals)
        successfulCount = 0

    return sorted_labels, cluster_totals


# Method to calculate new centroids using cluster means
def getNewCentroids(K, D, sorted_labels, cluster_totals, feat_vects):
    # Calculate mean of each cluster and re-assign centroid
    iteration = 0
    newCentroids = None
    for i in range(K):
        temp = np.zeros(D)
        cluster_total = cluster_totals[i][1]
        for j in range(cluster_total):
            fv_num = sorted_labels[iteration][0]
            temp = np.add(temp, feat_vects[fv_num].toarray())
            iteration = iteration+1

        # If cluster_total is 0, randomly re-init a new centroid
        # Else compute average of cluster and add to newCentroids
        if cluster_total == 0:
            temp = np.random.randint(0,4,D)
            temp[temp!=1]=0
        else:
            temp = np.divide(temp, cluster_total)

        # Stack newCentroids in an array
        if newCentroids is None:
            newCentroids = temp
        else:
            newCentroids = np.vstack((newCentroids,temp))

    newCentroids = csr_matrix(newCentroids)

    return newCentroids

# Method to calculate new centroids using closest instance to cluster means
def getNewCentroidsClosestInstance(K, D, sorted_labels, cluster_totals, feat_vects, l2_feat_vects):
    # Calculate mean of each cluster and re-assign centroid
    iteration = 0
    iteration2 = 0
    newCentroids = None
    for i in range(K):
        temp = np.zeros(D)
        cluster_total = cluster_totals[i][1]
        for j in range(cluster_total):
            fv_num = sorted_labels[iteration][0]
            temp = np.add(temp, feat_vects[fv_num].toarray())
            iteration = iteration+1

        # If cluster_total is 0, randomly re-init a new centroid
        # Else compute average of cluster and add to newCentroids
        if cluster_total == 0:
            temp = np.random.randint(0,4,D)
            temp[temp!=1]=0
        else:
            temp = np.divide(temp, cluster_total)

        # Calculate the closest instance
        distance, shortestDistance, closest = None, None, None
        for j in range(cluster_total):
            # Get distance to the mean for each feature vector
            fv_num = sorted_labels[iteration2][0]
            distance = np.sqrt(((l2_feat_vects[fv_num].toarray() - temp)**2).sum(axis=1))
            iteration2 = iteration2+1

            if distance < shortestDistance or shortestDistance is None:
                shortestDistance = distance
                closest = fv_num

        # Stack newCentroids in an array
        if newCentroids is None:
            newCentroids = l2_feat_vects[fv_num].toarray()
        else:
            newCentroids = np.vstack((newCentroids,l2_feat_vects[fv_num].toarray()))

    newCentroids = csr_matrix(newCentroids)

    return newCentroids


# Method to calculate precision, recall and f_score
def evaluate(K, sorted_labels, cluster_totals, f_labels):

    # Sort feat_vects by cluster and retreive their labels
    # Store this in a dict ready for calculations
    iteration = 0
    clusters = []
    for i in range(K):
        cluster = {}
        cluster_total = cluster_totals[i][1]
        for j in range(cluster_total):
            fv_num = sorted_labels[iteration][0]
            label = f_labels[fv_num]
            cluster[fv_num] = label
            iteration += 1

        clusters.append(cluster)

    # Get majority cluster labels and check for clusters with same label
    labelList = []
    for cluster in clusters:
        labelList.append(Counter(cluster.values()).most_common(1)[0])
    labelClusters = {}
    for num, val in enumerate(labelList):
        if val[0] not in labelClusters:
            labelClusters[val[0]] = [[num,val[1]]]
        else:
            labelClusters[val[0]].append([num,val[1]])

    # Merge clusters with same label
    # Calculate precision and recall
    precision, recall = 0, 0
    for key in labelClusters:
        truePositives, clusterSize = 0, 0
        for cluster in labelClusters[key]:
            # Add values for same labeled clusters within this inner loop
            truePositives += cluster[1]
            clusterSize += len(clusters[cluster[0]])
            print "Cluster %i \t Majority: (%i/%i) \t %s" % (cluster[0], cluster[1], len(clusters[cluster[0]]), key)

        precision += float(truePositives) / clusterSize
        recall += float(truePositives) / 51

    # Final calculations and print
    precision = precision / len(labelClusters)
    recall = recall / len(labelClusters)
    f_score = 2 * ((precision * recall) / (precision + recall))
    print "\nMacro-Averaged Precision:", precision
    print "Macro-Averaged Recall:", recall
    print "Macro-Averaged F-Score:", f_score

    return precision, recall, f_score


if __name__ == "__main__":
    menu()
