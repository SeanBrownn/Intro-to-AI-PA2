import csv
import math
from statistics import mean

import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from itertools import cycle
from mpl_toolkits.mplot3d import Axes3D

df=pd.read_csv('IRIS.csv')

# selects k random points as cluster centroids, returns table of coordinates
def initializeCentroids(k):
    randomList=random.sample(range(len(df)),k)
    # generates k random indexes of points in the dataframe
    cluster=np.zeros(len(df))
    k=0
    centroids=pd.DataFrame()
    for i in randomList:
        cluster[i]=k
        centroids[k] = df.iloc[i, :4] # column k in means is coordinates of mean of cluster k
        k=k+1
    df['cluster']=cluster
    return centroids

# adds df['newCluster], which assigns each data point to cluster w/ closest centroid
def assign(centroids):
    newCluster=np.zeros(len(df))
    for i in range(len(df)):
        coordinates=df.iloc[i,:4]
        minDistance=math.dist(coordinates,centroids[0]) # uses L2 to find nearest mean
        cluster=0
        for column in centroids:
            if math.dist(coordinates,centroids[column])<minDistance:
                minDistance=math.dist(coordinates,centroids[column])
                cluster=column
        newCluster[i]=cluster
    df['newCluster']=newCluster

# returns new centroids for each cluster
def computeCentroids(k):
    centroids=pd.DataFrame()
    for i in range(k):
        thisCluster=df.loc[df['cluster']==i] # finds all data points assigned to cluster i
        thisCluster=thisCluster.iloc[:, :4]
        centroids[i]=thisCluster.mean(axis=0)
        # centroid[i]= centroid for all points assigned to cluster i
    return centroids

# returns value of D given a set of centroids
def objectiveFunction(centroids):
    distortion=0
    for i in range(len(df)):
        coordinates=df.iloc[i, :4]
        cluster=int(df.at[i,'cluster']) # the cluster the data point at row i is in
        distortion=distortion+pow(math.dist(coordinates,centroids[cluster]),2)
        # squares the distance between the coordinates of a given point and the centroid for the
        # cluster the point is in
    return distortion

# plots objective function, learning process, decision boundaries for k means clustering
def plotKMeans(x,y,centers,data,k):
    fig1=plt.figure("figure 1")
    plt.scatter(x, y)
    plt.plot(x, y)
    plt.title("Value of objective function as a function of iterations")
    plt.xlabel("Iteration #")
    plt.ylabel("Objective function")
    plt.show()

    fig2=plt.figure("figure 2")
    plt.scatter(data['petal_length'],data['petal_width'],s=10, c='b') # makes data points blue
    if k==2:
        colors="gr" # cycle of colors to go through when plotting cluster centers
    else: # we can do else since we are only trying k=2 or k=3
        colors="grk"
    colorIndex=0
    for i in range(0,len(centers),2):
        for cluster in range(k): # for each cluster
            if i==0:
                plt.scatter(centers.iat[i, cluster], centers.iat[i + 1, cluster], s=10, marker='D',
                            c=colors[colorIndex]) # makes initial centers diamond
            elif i==len(centers)-2:
                plt.scatter(centers.iat[i, cluster], centers.iat[i + 1, cluster], s=10, marker='s',
                            c=colors[colorIndex]) # makes converged centers square
            else:
                plt.scatter(centers.iat[i,cluster],centers.iat[i+1,cluster], s=10,
                            c=colors[colorIndex])
            colorIndex=(colorIndex+1)%len(colors)
            # plots cluster centers (each in a different color) overlaid on data
    plt.title("Cluster centers overlaid on data points")
    plt.xlabel("petal length")
    plt.ylabel("petal width")
    plt.show()

    finalCenters=(centers.tail(2).T).to_numpy()
    # gets final coordinates of cluster centers in a form where we can make voronoi diagram
    boundary=np.array([[5.0,7.0]]) # "dummy" boundary so that boundaries show up on graph
    points=np.concatenate([finalCenters,boundary])
    vor=Voronoi(points)
    fig3=voronoi_plot_2d(vor)
    plt.title("Decision boundaries for k means clustering")
    plt.xlabel("petal length")
    plt.ylabel("petal width")
    for i in range(len(data)):
        if data.at[i,'species'] == "Iris-setosa":
            color = 'r'
        elif data.at[i,'species'] == "Iris-versicolor":
            color = 'g'
        else:
            color = 'k'
        plt.scatter(data.at[i,'petal_length'], data.at[i,'petal_width'], s=10,c=color)
    plt.show()

# assigns each data point to a cluster and returns the assignments as a list
def kMeans(k):
    centroids=initializeCentroids(k)
    assign(centroids)
    plotCenters = pd.DataFrame()  # centers in terms of (petal length, width); used for plotting
    plotCenters=plotCenters._append(centroids.tail(2)) # adds centers for iteration 1
    iterations = 1  # of iterations
    x = list()
    x.append(iterations) # (x,y) = (iteration #, value of objective function); used for plotting
    y = list()
    y.append(objectiveFunction(centroids))
    while(df['cluster'].equals(df['newCluster']))==False: # continue if we reassigned data points
        df['cluster']=df['newCluster']
        df.drop('newCluster', axis=1, inplace=True)
        # the above 2 lines change cluster assignments to the new ones
        newCentroids = computeCentroids(k)
        assign(newCentroids)
        plotCenters = plotCenters._append(newCentroids.tail(2))
        iterations = iterations + 1
        x.append(iterations)
        y.append(objectiveFunction(newCentroids))
    df['cluster']=df['newCluster']
    assignments = df['cluster']
    df.drop(['newCluster','cluster'], axis=1, inplace=True)
    plotKMeans(x,y,plotCenters,df, k)
    return assignments

# returns classification of each point in dataVectors using linear weights
def classify(dataVectors, weights):
    # weights is a 3x1 matrix, so we need to remove w0 and transpose remaining matrix
    weights=weights.T
    sums=weights.dot(dataVectors) # multiplies matrices
    actualSpecies=np.zeros(len(sums.axes[1])) # len(sums.axes[1]) = # of columns, which is what we need
    y=1/(1+np.exp(-sums))
    for i in range(len(y.axes[1])):
        if y.at[0,i]<0.5: # do [0,i] because y is a dataframe w/ 1 row
            actualSpecies[i]=0 # our neuron assigns iris versicolor
        else:
            actualSpecies[i]=1 # our neuron assigns iris virginica
    return actualSpecies

#returns mean squared errors of classifications
def meanSquaredError(dataVectors, weights,labels):
    assignments=classify(dataVectors,weights)
    squaredDiff = pow((labels - assignments), 2)
    mse = mean(squaredDiff)  # mean of squared errors
    return mse

# returns summed gradient for data vectors
def gradients(dataVectors,weights,labels):
    assignments=classify(dataVectors,weights)
    diff=(labels-assignments) # expected-predicted
    dYHat=pd.DataFrame(-2*diff) # derivative of D w/ respect to predicted values
    weights=weights.T
    sums=weights.dot(dataVectors)
    g=1/(1+np.exp(-sums)) # output of activation function for each data vector
    dataVectors=dataVectors.T
    dw0=(g*(1-g)).T # derivative of y hat w/ respect to w0
    dw1=(dataVectors[1]*g*(1-g)).T # derivative of y hat w/ respect to w1
    dw2=(dataVectors[2]*g*(1-g)).T # derivative of y hat w/ respect to w2
    grad0=sum(dYHat.iat[i,0]*dw0.iat[i,0] for i in range(len(dw0))) # derivative of D w/ respect to w0
    grad1=sum(dYHat.iat[i,0]*dw1.iat[i,0] for i in range(len(dw1))) # derivative of D w/ respect to w1
    grad2=sum(dYHat.iat[i,0]*dw2.iat[i,0] for i in range(len(dw2))) # derivative of D w/ respect to w2
    gradients = [grad0,grad1,grad2]
    gradients=pd.DataFrame(gradients)
    return gradients

# plots decision boundaries for neural network
def plotNN(dataVectors,weights,expectedSpecies):
    vectors=dataVectors.T
    mins=vectors.min()
    maxes=vectors.max()
    min1, max1 = mins[1] - 1, maxes[1] + 1
    min2, max2 = mins[2] - 1, maxes[2] + 1
    x1grid = np.arange(min1, max1, 0.1)
    x2grid = np.arange(min2, max2, 0.1)
    xx, yy = np.meshgrid(x1grid, x2grid)
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    grid = np.hstack((r1, r2))
    newGrid=pd.DataFrame(data=grid)
    ones = np.ones(len(newGrid))
    newGrid.insert(0,'ones', ones)
    newGrid=newGrid.T.reset_index(drop=True)
    yhat=classify(newGrid,weights)
    zz = yhat.reshape(xx.shape)
    plt.contourf(xx, yy, zz, cmap='Paired')
    vectors['expected']=expectedSpecies
    for class_value in range(2):
        select=vectors.loc[vectors['expected']==class_value]
        # create scatter of these samples
        if class_value==0:
            color='g'
        else:
            color='r'
        for i in range(len(select)):
            plt.scatter(select.iat[i, 1], select.iat[i,2], c=color)
    plt.title("decision boundaries for neural network")
    plt.xlabel("petal length")
    plt.ylabel("petal width")
    plt.show()

linearWeights=[20,-2.7,-1.6]
weights=pd.DataFrame(linearWeights) # for mean squared error function
data=df[df['species']!="Iris-setosa"]
data=data.reset_index()
expectedSpecies=np.zeros(len(data)) # pattern classes for mean squared error function
sums=np.zeros(len(data))
for i in range(len(data)):
    if data.at[i,'species']=="Iris-virginica":
        expectedSpecies[i]=1.0
    # if iris-versicolor, leave it as zero
coordinates=data.iloc[:, 3:5] # takes only petal length and width
ones=np.ones(len(coordinates))
coordinates.insert(0,0,ones)
coordinates=coordinates.T # data vectors needed for input into mean squared error function
coordinates=coordinates.reset_index() # resets so that we can multiply the matrices
coordinates.drop('index',axis=1,inplace=True)
print(meanSquaredError(coordinates,weights,expectedSpecies))
#kMeans(2)
plotNN(coordinates,weights,expectedSpecies)
