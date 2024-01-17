import time
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler

from utils import common


def DensityAndBarDistance(X,i ):
    if i == BatchNum - 1:
        subX = X[i * subSize:, :]
    else:
        subX = X[i * subSize:(i + 1) * subSize, :]
    TempRowNum = np.shape(subX)[0]
    ### Part against All distance
    SubtDist = common.euclidian_dist(subX, X)
    subDensity = common.ComputeLocalDensity(SubtDist, dc)
    ###patition is IMPORTANT!!!
    subKNearInds = (np.argpartition(SubtDist, KNearest))[:, 0:KNearest]
    subKNearDist = np.zeros((TempRowNum, KNearest), dtype='float32')
    for row in range(TempRowNum):
        subKNearDist[row, :] = SubtDist[row, subKNearInds[row, :]]
    return subDensity, subKNearInds, subKNearDist

def ClusterIDAssignment(density, ledingNodes, LeadingDistance, K):
    N = len(density)
    Y = np.zeros(N,dtype="int")-1
    potential = density * LeadingDistance
    sortedPotentialInds = np.argsort(-1*potential)
    for i in range(K):
        Y[sortedPotentialInds[i]] = i


    PathNodes = []
    for j in range(N):
        if Y[j] != -1:
            continue
        else:
            PathNodes.append(j)
            pa = ledingNodes[j]
            while Y[pa] == -1:
                pa = ledingNodes[pa]
                PathNodes.append(pa)
                print("pa", pa, "path length:", len(PathNodes))

            label =  Y[pa]

            for node in PathNodes:
                Y[node] = label
            PathNodes =[]


    return Y

def FindLeadingNodeEasy(i, subsize, KNearDist,KNearInds, DensityAll, LeadingNodeInds, LeadingDistance):

    if i == BatchNum - 1:
        IDrange = range(i * subSize, N)

    else:
        IDrange = range(i * subSize, (i + 1) * subSize)
    for nodeID in IDrange:
        ### solution One: sort KnearDist
        distVec = KNearDist[nodeID, :]
        distSortInds = np.argsort(distVec)
        distVecSort = distVec[distSortInds]

        RealDistSortInds = KNearInds[nodeID, distSortInds]
        for j in range(KNearest):
            possibleLeading = RealDistSortInds[j]
            if DensityAll[possibleLeading] > DensityAll[nodeID]:
                LeadingNodeInds[nodeID] = possibleLeading
                LeadingDistance[nodeID] = distVecSort[j]  ### Attention!
                break  ###find then finish!


if __name__ == "__main__":
    lt_num = 8  # number of subtrees
    wine = datasets.load_wine()
    X, Y = common.load_data("data/5Spiral50K.csv", label_index=3)
    X = X[:, 1:]

    #X, _, _ = common.max_min_norm(X)
    N = np.shape(X)[0]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    dc = 0.2

    t1 = time.time()
    BatchNum =500
    subSize = int(N/BatchNum)
    DensityAll = np.zeros(N, dtype='float32')
    ## the width of bar matrix, storing the nearest neighbors' Indics and Distance
    KNearest = 20
    KNearInds = np.zeros((N,KNearest), dtype='int')
    KNearDist = np.zeros((N, KNearest), dtype='float32')

    ### limulate paralell with data independent sequential processing

    #### Step One: compute rho and bar Distance matrices########
    ##############################################################
    print("Computing local density and bar matrix....")
    for i in range(BatchNum):
        if (i % 5 ==0):
            print("Worker No.", i , " of ",BatchNum)
        ## The last work takes all remainder
        subDensity, subKNearInds, subKNearDist = DensityAndBarDistance(X,i)
        ##The last part
        if i == BatchNum-1:
            DensityAll[i * subSize:] = subDensity
            KNearInds[i * subSize:, :] = subKNearInds
            KNearDist[i * subSize:, :] = subKNearDist
        else:
            DensityAll[i * subSize:(i + 1) * subSize] = subDensity
            KNearInds[i * subSize:(i + 1) * subSize, :] = subKNearInds
            KNearDist[i * subSize:(i + 1) * subSize, :] = subKNearDist

    #### Step Two: compute leading nodes and leading distance ########
    ##############################################################

    # DensityAllsortInds = np.argsort(DensityAll, 'descending')
    # DensityAllsort = DensityAll[DensityAllsortInds]
    LeadingNodeInds = np.zeros(N, dtype="int")-1
    LeadingDistance = np.zeros(N, dtype="float32")-1

    print("Computing leading nodes and delta distance...")
    for i in range(BatchNum):
        ## The last work takes all remainder
        FindLeadingNodeEasy(i, subSize, KNearDist, KNearInds, DensityAll, LeadingNodeInds, LeadingDistance)

    #### Step Three: compute leading nodes for those failed in Step Two ########
    ##############################################################
    ###Solution One, step A: sparse minicenters' distance matrix and extracted densities
    NotFoundInds = np.array( [i for i in range(N) if LeadingNodeInds[i]==-1 ])
    mcNum = len(NotFoundInds)

    DensitysortInds = np.argsort(-DensityAll)
    Densitysort = DensityAll[DensitysortInds]

    ### step B: Recomputing the distance between micro centers and the entire dataset
    mCenterX = X[NotFoundInds, :] #### already in the order of density
    mCenterDist = common.euclidian_dist(mCenterX, X) ###Solution Two
    ##mCenterDist = common.euclidian_dist(mCenterX, mCenterX) ###Solution one

    for i in range(mcNum):### pay attention to Index Transfering

        currInd = NotFoundInds[i]
        if currInd == DensitysortInds[0] :
            LeadingDistance[currInd] = max(mCenterDist[i, :])
            continue

        LeadingNodeInds[currInd] = DensitysortInds[0]
        LeadingDistance[currInd] = mCenterDist[i, DensitysortInds[0]]
        StopDensityInd = list(DensitysortInds) .index(currInd)
        ## Search Range
        for j in range (1, StopDensityInd):
            tmpj = DensitysortInds[j]
            if mCenterDist[i,tmpj] < LeadingDistance[currInd]:
                LeadingNodeInds[currInd] = tmpj
                LeadingDistance[currInd] = mCenterDist[i, tmpj]

    Y = ClusterIDAssignment(DensityAll, LeadingNodeInds, LeadingDistance, 5)

    t2= time.time()

    print("Time consumption (s):", t2-t1)

    common.PlotCluster(X, Y)





