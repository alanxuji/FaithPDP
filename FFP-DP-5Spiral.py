##### AlanXu @ SKLPBD, GZU. 2024-1-15
##### For Faithful parallelism of Density Peaks Clustering
#### mpiexec -n 14 python FFP-DP-5Spiral.py
import time
from mpi4py import MPI
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from utils import common


def DensityAndBarDistance(ranX, X, i, subSize ):
    if i == BatchNum - 1:
        subX = ranX[i * subSize:, :]
    else:
        subX = ranX[i * subSize:(i + 1) * subSize, :]
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
                #print("pa", pa, "path length:", len(PathNodes))

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

    X, Y = common.load_data("data/5Spiral50K.csv", label_index=3)
    X = X[:, 1:]

    #X, _, _ = common.max_min_norm(X)
    N = np.shape(X)[0]
    scaler = MinMaxScaler()

    X = scaler.fit_transform(X)
    dc = 0.2
    BatchNum = 20 ### for 500K
    #BatchNum = 20


    ## the width of bar matrix, storing the nearest neighbors' Indics and Distance
    KNearest = 20
    KNearInds = np.zeros((N, KNearest), dtype='int')
    KNearDist = np.zeros((N, KNearest), dtype='float32')
    DensityAll = np.zeros(N, dtype='float32')

    t1 = time.time()

    ###### MPI Parallelism
    comm = MPI.COMM_WORLD
    size = comm.Get_size() #### 10 or 20, Now suppose N is Integer times of size.
    rank = comm.Get_rank()

    rankSize = int(N / size)
    BeginID = int(rankSize * rank )
    print("BeginID:", BeginID)
    EndID = int(rankSize * (rank + 1) )
    if rank == size-1: ####deal with the remaining a few samples when N is not the integer times of size
        EndID = N
    rankX = X[BeginID:EndID, ]

    # comm.send(sum0, dest=size-1)
    print("This is rank %d, on node2" % rank)

    if rank == 0:


        #subSize = int(N/BatchNum)
        subSize = int(rankSize / BatchNum)
        ### limulate paralell with data independent sequential processing

        #### Step One: compute rho and bar Distance matrices########
        ##############################################################
        print("Computing local density and bar matrix....")
        for i in range(BatchNum):
            if (i % 5 ==0):
                print("Batch No.", i , " of ",BatchNum)
            ## The last work takes all remainder
            subDensity, subKNearInds, subKNearDist = DensityAndBarDistance(rankX, X, i, subSize)
            ##The last part
            if i == BatchNum-1:
                DensityAll[rankSize*rank+i * subSize:rankSize*(rank+1)] = subDensity
                KNearInds[rankSize*rank+i * subSize:rankSize*(rank+1), :] = subKNearInds
                KNearDist[rankSize*rank+i * subSize:rankSize*(rank+1), :] = subKNearDist
            else:
                DensityAll[rankSize*rank+i * subSize:rankSize*rank+(i + 1) * subSize] = subDensity
                KNearInds[rankSize*rank+i * subSize:rankSize*rank+(i + 1) * subSize, :] = subKNearInds
                KNearDist[rankSize*rank+i * subSize:rankSize*rank+(i + 1) * subSize, :] = subKNearDist
        #####Collect data to process whose rank==0

        for sendRank in range(1, size):
            # recvDensity = comm.recv(source=sendRank, tag=1)
            # recvKNearInds = comm.recv(source=sendRank, tag=2)
            # recvKNearDist = comm.recv(source=sendRank, tag=3)####还需要再放置！！！
            # DensityAll[rankSize * rank:rankSize * (rank + 1)] = recvDensity
            # KNearInds[rankSize * rank:rankSize * (rank + 1), :] = recvKNearInds
            # KNearDist[rankSize * rank :rankSize * (rank + 1), :] = recvKNearDist
            recvEndID = rankSize * (sendRank + 1)
            if sendRank == size-1:
                recvEndID = N
            DensityAll[rankSize * sendRank:recvEndID]  = comm.recv(source=sendRank, tag=1)
            KNearInds[rankSize * sendRank:recvEndID, :]  = comm.recv(source=sendRank, tag=2)
            KNearDist[rankSize * sendRank :recvEndID, :]= comm.recv(source=sendRank, tag=3)  #

    else:

        #subSize = int(N/BatchNum)
        subSize = int(rankSize / BatchNum)
        ### limulate paralell with data independent sequential processing

        #### Step One: compute rho and bar Distance matrices########
        ##############################################################
        print("Computing local density and bar matrix....")
        for i in range(BatchNum):
            if (i % 5 ==0):
                print("Batch No.", i, " of ", BatchNum)
            ## The last work takes all remainder
            subDensity, subKNearInds, subKNearDist = DensityAndBarDistance(rankX, X, i, subSize)
            ##The last part
            if i == BatchNum-1:
                BatchEndID = rankSize * (rank + 1)
                if rank == size - 1:
                    BatchEndID = N

                DensityAll[rankSize*rank+i * subSize:BatchEndID] = subDensity
                KNearInds[rankSize*rank+i * subSize:BatchEndID, :] = subKNearInds
                KNearDist[rankSize*rank+i * subSize:BatchEndID, :] = subKNearDist
            else:
                DensityAll[rankSize*rank+i * subSize:rankSize*rank+(i + 1) * subSize] = subDensity
                KNearInds[rankSize*rank+i * subSize:rankSize*rank+(i + 1) * subSize, :] = subKNearInds
                KNearDist[rankSize*rank+i * subSize:rankSize*rank+(i + 1) * subSize, :] = subKNearDist

        SendEndID = rankSize * (rank + 1)
        if rank == size - 1:
            SendEndID = N
        comm.send(DensityAll[rankSize * rank :SendEndID] , dest=0, tag = 1)
        comm.send(KNearInds[rankSize * rank :SendEndID, :] , dest=0, tag = 2)
        comm.send(KNearDist[rankSize * rank :SendEndID, :], dest=0, tag = 3)

    comm.barrier()
    print("density and matrices completed!")
    if rank == 0:    ### Step Two: compute leading nodes and leading distance ########
    #############################################################

    # DensityAllsortInds = np.argsort(DensityAll, 'descending')
    # DensityAllsort = DensityAll[DensityAllsortInds]

        #common.write_csv("KNearInds.csv", KNearInds)
        #common.write_csv("DensityAll", DensityAll)

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
        # print("Labels: \n",Y)
        common.write_csv("Labels.csv",np.reshape(Y,(N,1)))
        common.PlotCluster(X, Y)

    ####TESTING CODE#############
    # D = common.euclidian_dist(X, X)
    # lt1 = lt.LeadingTree(X_train=X, dc=0.2, lt_num=lt_num, D=D)  # Constructing the lead tree for the entire dataset
    # lt1.fit()
    # print("leading distance diff:\n", LeadingDistance-lt1.delta)




