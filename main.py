import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0) # Random seed for result verifaction.

def load_data(fname, classLabel):
    features = []
    with open(fname) as F:
        for line in F:
            p = line.strip().split(' ')
            features.append(np.array(p[1:], float))
    data = np.array(features)
    labels = np.full((np.shape(features)[0], 1), classLabel)
    data = np.hstack((labels, data))
    return data

for i in range(4):
    if i == 0:
        dataset = load_data("veggies",i)
    if i == 1:
        dataset = np.vstack((dataset,load_data("fruits",i)))
    if i == 2:
        dataset = np.vstack((dataset,load_data("animals",i)))
    if i == 3:
        dataset = np.vstack((dataset,load_data("countries",i)))

def l2Norm(data):
    for rowIndex, row in enumerate(data):
        norm = np.linalg.norm(row)
        for colIndex, item in enumerate(row):
            data[rowIndex,colIndex] = item/norm
    return data

def euclidean(a,b): # Euclidean sqaured not euclidean
    distance = np.sum((a-b)**2)
    return distance

def manhattan(a,b):
    distance = np.sum(np.abs(a-b))
    return distance

def plot(precision, recall, fscore, kname, l2):
    labels = ['K=1', 'K=2', 'K=3', 'K=4', 'K=5', 'K=6', 'K=7', 'K=8', 'K=9']
    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, precision, width, label='Precision')
    rects2 = ax.bar(x, recall, width, label='Recall')
    rects3 = ax.bar(x + width, fscore, width, label='F-Score')
    ax.set_ylabel('Scores')
    if l2 == 0:
        ax.set_title(f'B-Cubed Scores of {kname} 1-9')
    else:
        ax.set_title(f'B-Cubed Scores of {kname} 1-9 with L2-Regularisation')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)
    fig.tight_layout
    plt.show()

class Clustering():
    
    def __init__(self, method, dataset, K, norm):
        self.K = K
        self.method = method
        self.dataset = dataset
        self.norm = norm
        self.labels = self.dataset[:,:1]
        self.dataset = self.dataset[:,1:]
        if self.norm == 1:
            self.dataset = l2Norm(self.dataset)
        if self.method == "means":
            self.cluster = np.random.uniform(low=-1.0, high=1.0, size=(self.K,300))
        else:
            self.randomIndices = np.random.choice(self.dataset.shape[0], size=self.K, replace=False)
            self.cluster = self.dataset[self.randomIndices, :]
        self.clusterKey = np.arange(1, self.K+1)

    def assign(self):
        self.clusterList = []
        for row in self.dataset:
            distancesLog = []
            for key, cluster in zip(self.clusterKey,self.cluster):
                if self.method == "means":
                    distancesLog.append(euclidean(row, cluster))
                else:
                    distancesLog.append(manhattan(row, cluster))
            self.clusterList.append(distancesLog.index(min(distancesLog)))
        self.clusterList = np.array(([self.clusterList])).transpose()
        return self.clusterList # Return a list of all indexes for the cluster positions

    def groupBy(self, dataStack):
        indices = np.unique(dataStack[:, 0]) 
        size = len(indices)
        ranges = np.zeros((size,), dtype=object)
        for i in range(size):
            ranges[i] = dataStack[dataStack[:, 0] == indices[i]] # Returns a list of grouped lists from the clusterID
        return ranges

    def optimise(self):
        oldCentroid = self.cluster
        newCentroid = []
        dataStack = np.hstack((self.clusterList,self.dataset)) # add list of cluster to the data
        dataSort = self.groupBy(dataStack)
        for array in dataSort:
            array = array[:,1:]
            if self.method == "means":
                newCentroid.append(np.mean(array, axis=0))
            else:
                newCentroid.append(np.median(array, axis=0))
        self.cluster = np.array(newCentroid)
        if np.array_equal(oldCentroid, newCentroid) == True: # Converged
            return True
        else:
            return False

    def clusterCount(self):
        numberMatrix = np.zeros((self.K,4)) # Set the matrix to the right size
        dataStack = np.hstack((self.clusterList,self.labels))
        returnCluster = self.groupBy(dataStack)
        for array in returnCluster:
            array[:, [0, 1]] = array[:, [1, 0]] # Swap Columns so we can reuse the groupBy function.
            nestArray = self.groupBy(array)
            for array in nestArray:
                y = int(array[0,0])
                x = int(array[0,1])
                numberMatrix[x,y] = array.shape[0] # Insert the number of elements for cluster in label for the matrix.
        print(numberMatrix)
        return numberMatrix

    def bCubed(self):
        numberMatrix = self.clusterCount() # Matrix containing all of the clusters with label data
        labelCount = np.sum(numberMatrix, axis=0) # Total Number of items with Label
        clusterCount = np.sum(numberMatrix, axis=1) # number of items in cluster
        precisionArray = []
        recallArray = []
        fscoreArray = []
        for rowIndex, row in enumerate(numberMatrix): # Loop through the matrix array to calculate scores.
            for colIndex, digit in enumerate(row):
                if digit != 0: # Ignore empty entries
                    for i in range(int(digit)):
                        precisionX = digit/clusterCount[rowIndex] # X means for that X object
                        precisionArray.append(precisionX)
                        recallX = digit/labelCount[colIndex]
                        recallArray.append(recallX)
                        fscoreX = ((2 * recallX * precisionX)/(recallX + precisionX))
                        fscoreArray.append(fscoreX)
        precision = np.mean(precisionArray)
        recall = np.mean(recallArray)
        fscore = np.mean(fscoreArray)
        return precision, recall, fscore

    def run(self):
        converge = False
        count = 0
        while converge == False: # Run Assigning & Optimising until convergence
            self.assign()
            converge = self.optimise()
            count += 1
        if self.method == "means":
            method = "K-Means"
        else:
            method = "K-Medians"
        if self.norm == 0:
            self.bCubed()
            print(f"{method} = {self.K}: Converged after: {count} times")
        else: 
            self.bCubed()
            print(f"L2-Norm, {method} = {self.K}: Converged after: {count} times")
        return self.bCubed()

# K-Means 1-9
precision = []
recall = []
fscore = []
for i in range(9):
    kmeans = Clustering("means",dataset,i+1,0)
    results = kmeans.run()
    precision.append(results[0])
    recall.append(results[1])
    fscore.append(results[2])
plot(precision, recall, fscore, "K-Means", 0)

# K-Means 1-9 with L2-Normalisation
precision = []
recall = []
fscore = []
for i in range(9):
    kmeans = Clustering("means",dataset,i+1,1)
    results = kmeans.run()
    precision.append(results[0])
    recall.append(results[1])
    fscore.append(results[2])
plot(precision, recall, fscore, "K-Means", 1)

# K-Medians 1-9
precision = []
recall = []
fscore = []
for i in range(9):
    kmedians = Clustering("medians",dataset,i+1,0)
    results = kmedians.run()
    precision.append(results[0])
    recall.append(results[1])
    fscore.append(results[2])
plot(precision, recall, fscore, "K-Medians", 0)

# K-Medians 1-9 with L2-Normalisation
precision = []
recall = []
fscore = []
for i in range(9):
    kmedians = Clustering("medians",dataset,i+1,1)
    results = kmedians.run()
    precision.append(results[0])
    recall.append(results[1])
    fscore.append(results[2])
plot(precision, recall, fscore, "K-Medians", 1)