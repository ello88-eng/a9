import os
import random
import sys
import time

import cv2
import numpy as np
from numba import njit

from policies.surv.algorithms.ADD.visualization import Visualizer

np.set_printoptions(threshold=sys.maxsize)

random.seed(1)
os.environ["PYTHONHASHSEED"] = str(1)
np.random.seed(1)


@njit(fastmath=True)
def assign(droneNo, rows, cols, GridEnv, MetricMatrix, A):

    ArrayOfElements = np.zeros(droneNo)
    for i in range(rows):
        for j in range(cols):
            if GridEnv[i, j] == -1:
                minV = MetricMatrix[0, i, j]
                indMin = 0
                for r in range(droneNo):
                    if MetricMatrix[r, i, j] < minV:
                        minV = MetricMatrix[r, i, j]
                        indMin = r

                A[i, j] = indMin
                ArrayOfElements[indMin] += 1

            elif GridEnv[i, j] == -2:
                A[i, j] = droneNo
    return A, ArrayOfElements


@njit(fastmath=True)
def inverse_binary_map_as_uint8(BinaryMap):
    # cv2.distanceTransform needs input of dtype unit8 (8bit)
    return np.logical_not(BinaryMap).astype(np.uint8)


@njit(fastmath=True)
def euclidian_distance_points2d(array1: np.array, array2: np.array):
    # this runs much faster than the (numba) np.linalg.norm and is totally enough for our purpose
    return (((array1[0] - array2[0]) ** 2) + ((array1[1] - array2[1]) ** 2)) ** 0.5


@njit(fastmath=True)
def constructBinaryImages(labels_im, robo_start_point, rows, cols):
    BinaryRobot = np.copy(labels_im)
    BinaryNonRobot = np.copy(labels_im)
    for i in range(rows):
        for j in range(cols):
            if labels_im[i, j] == labels_im[robo_start_point]:
                BinaryRobot[i, j] = 1
                BinaryNonRobot[i, j] = 0
            elif labels_im[i, j] != 0:
                BinaryRobot[i, j] = 0
                BinaryNonRobot[i, j] = 1

    return BinaryRobot, BinaryNonRobot


@njit(fastmath=True)
def CalcConnectedMultiplier(rows, cols, dist1, dist2, CCvariation):
    returnM = np.zeros((rows, cols))
    MaxV = 0
    MinV = 2**30

    for i in range(rows):
        for j in range(cols):
            returnM[i, j] = dist1[i, j] - dist2[i, j]
            if MaxV < returnM[i, j]:
                MaxV = returnM[i, j]
            if MinV > returnM[i, j]:
                MinV = returnM[i, j]

    for i in range(rows):
        for j in range(cols):
            returnM[i, j] = (returnM[i, j] - MinV) * ((2 * CCvariation) / (MaxV - MinV)) + (1 - CCvariation)

    return returnM


class DarpOld:
    def __init__(
        self,
        nx,
        ny,
        notEqualPortions,
        given_initial_positions,
        given_portions,
        obstacles_positions,
        visualization,
        MaxIter=80000,
        CCvariation=0.01,
        randomLevel=0.0001,
        dcells=2,
        importance=False,
    ):

        self.num_rows = nx
        self.num_cols = ny
        self.init_positions, self.obstacles_positions, self.portions = self.sanity_check(
            given_initial_positions,
            given_portions,
            obstacles_positions,
            notEqualPortions,
        )

        self.visualization = visualization
        self.MaxIter = MaxIter
        self.CCvariation = CCvariation
        self.randomLevel = randomLevel
        self.dcells = dcells
        self.importance = importance
        self.notEqualPortions = notEqualPortions

        print("\nInitial Conditions Defined:")
        print("Grid Dimensions:", nx, ny)
        print("Number of Robots:", len(self.init_positions))
        print("Initial Robots' positions", self.init_positions)
        print("Portions for each Robot:", self.portions, "\n")

        self.num_uavs = len(self.init_positions)
        self.assg_mat = np.zeros((self.num_rows, self.num_cols))
        self.GridEnv = self.defineGridEnv()

        self.connectivity = np.zeros((self.num_uavs, self.num_rows, self.num_cols), dtype=np.uint8)
        self.binary_uav_regions = np.zeros((self.num_uavs, self.num_rows, self.num_cols), dtype=bool)

        (
            self.MetricMatrix,
            self.termThr,
            self.Notiles,
            self.DesireableAssign,
            self.TilesImportance,
            self.MinimumImportance,
            self.MaximumImportance,
        ) = self.construct_Assignment_Matrix()
        self.ArrayOfElements = np.zeros(self.num_uavs)
        self.colors = []

        for r in range(self.num_uavs):
            # np.random.seed(r)
            self.colors.append(list(np.random.choice(range(256), size=3)))

        # np.random.seed(1)
        if self.visualization:
            self.assignment_matrix_visualization = Visualizer(
                self.assg_mat, self.num_uavs, self.colors, self.init_positions
            )

    def sanity_check(self, given_initial_positions, given_portions, obs_pos, notEqualPortions):
        initial_positions = []
        for position in given_initial_positions:
            if position < 0 or position >= self.num_rows * self.num_cols:
                print("Initial positions should be inside the Grid.")
                sys.exit(1)
            initial_positions.append((position // self.num_cols, position % self.num_cols))

        obstacles_positions = []
        for obstacle in obs_pos:
            if obstacle < 0 or obstacle >= self.num_rows * self.num_cols:
                print("Obstacles should be inside the Grid.")
                sys.exit(2)
            obstacles_positions.append((obstacle // self.num_cols, obstacle % self.num_cols))

        portions = []
        if notEqualPortions:
            portions = given_portions
        else:
            for drone in range(len(initial_positions)):
                portions.append(1 / len(initial_positions))

        if len(initial_positions) != len(portions):
            print("Portions should be defined for each drone")
            sys.exit(3)

        s = sum(portions)
        if abs(s - 1) >= 0.0001:
            print("Sum of portions should be equal to 1.")
            sys.exit(4)

        for position in initial_positions:
            for obstacle in obstacles_positions:
                if position[0] == obstacle[0] and position[1] == obstacle[1]:
                    print("Initial positions should not be on obstacles")
                    sys.exit(5)

        return initial_positions, obstacles_positions, portions

    def defineGridEnv(self):
        GridEnv = np.full(shape=(self.num_rows, self.num_cols), fill_value=-1)  # create non obstacle map with value -1

        # obstacle tiles value is -2
        for idx, obstacle_pos in enumerate(self.obstacles_positions):
            GridEnv[obstacle_pos[0], obstacle_pos[1]] = -2

        connectivity = np.zeros((self.num_rows, self.num_cols))

        mask = np.where(GridEnv == -1)
        connectivity[mask[0], mask[1]] = 255
        image = np.uint8(connectivity)
        num_labels, labels_im = cv2.connectedComponents(image, connectivity=4)

        if num_labels > 2:
            print("The environment grid MUST not have unreachable and/or closed shape regions")
            sys.exit(6)

        # initial robot tiles will have their array.index as value
        for idx, robot in enumerate(self.init_positions):
            GridEnv[robot] = idx
            self.assg_mat[robot] = idx

        return GridEnv

    def divide_regions(self):
        success = False
        cancelled = False
        criterionMatrix = np.zeros((self.num_rows, self.num_cols))
        iteration = 0

        while self.termThr <= self.dcells and not success and not cancelled:
            downThres = (self.Notiles - self.termThr * (self.num_uavs - 1)) / (self.Notiles * self.num_uavs)
            upperThres = (self.Notiles + self.termThr) / (self.Notiles * self.num_uavs)
            success = True

            # Main optimization loop
            iteration = 0
            while iteration <= self.MaxIter and not cancelled:
                self.assg_mat, self.ArrayOfElements = assign(
                    self.num_uavs,
                    self.num_rows,
                    self.num_cols,
                    self.GridEnv,
                    self.MetricMatrix,
                    self.assg_mat,
                )

                ConnectedMultiplierList = np.ones((self.num_uavs, self.num_rows, self.num_cols))
                ConnectedRobotRegions = np.zeros(self.num_uavs)
                plainErrors = np.zeros((self.num_uavs))
                divFairError = np.zeros((self.num_uavs))

                self.update_connectivity()
                for r in range(self.num_uavs):
                    ConnectedMultiplier = np.ones((self.num_rows, self.num_cols))
                    ConnectedRobotRegions[r] = True
                    num_labels, labels_im = cv2.connectedComponents(self.connectivity[r, :, :], connectivity=4)
                    if num_labels > 2:
                        ConnectedRobotRegions[r] = False
                        BinaryRobot, BinaryNonRobot = constructBinaryImages(
                            labels_im,
                            self.init_positions[r],
                            self.num_rows,
                            self.num_cols,
                        )
                        ConnectedMultiplier = CalcConnectedMultiplier(
                            self.num_rows,
                            self.num_cols,
                            self.NormalizedEuclideanDistanceBinary(True, BinaryRobot),
                            self.NormalizedEuclideanDistanceBinary(False, BinaryNonRobot),
                            self.CCvariation,
                        )
                    ConnectedMultiplierList[r, :, :] = ConnectedMultiplier
                    plainErrors[r] = self.ArrayOfElements[r] / (self.DesireableAssign[r] * self.num_uavs)
                    if plainErrors[r] < downThres:
                        divFairError[r] = downThres - plainErrors[r]
                    elif plainErrors[r] > upperThres:
                        divFairError[r] = upperThres - plainErrors[r]

                if self.IsThisAGoalState(self.termThr, ConnectedRobotRegions):
                    break

                TotalNegPerc = 0
                totalNegPlainErrors = 0
                correctionMult = np.zeros(self.num_uavs)

                for r in range(self.num_uavs):
                    if divFairError[r] < 0:
                        TotalNegPerc += np.absolute(divFairError[r])
                        totalNegPlainErrors += plainErrors[r]
                    correctionMult[r] = 1

                for r in range(self.num_uavs):
                    if totalNegPlainErrors != 0:
                        if divFairError[r] < 0:
                            correctionMult[r] = 1 + (plainErrors[r] / totalNegPlainErrors) * (TotalNegPerc / 2)
                        else:
                            correctionMult[r] = 1 - (plainErrors[r] / totalNegPlainErrors) * (TotalNegPerc / 2)

                        criterionMatrix = self.calculateCriterionMatrix(
                            self.TilesImportance[r],
                            self.MinimumImportance[r],
                            self.MaximumImportance[r],
                            correctionMult[r],
                            divFairError[r] < 0,
                        )

                    self.MetricMatrix[r] = self.FinalUpdateOnMetricMatrix(
                        criterionMatrix,
                        self.generateRandomMatrix(),
                        self.MetricMatrix[r],
                        ConnectedMultiplierList[r, :, :],
                    )

                iteration += 1
                if self.visualization:
                    self.assignment_matrix_visualization.placeCells(self.assg_mat, iteration_number=iteration)
                    time.sleep(0.2)

            if iteration >= self.MaxIter:
                self.MaxIter = self.MaxIter / 2
                success = False
                self.termThr += 1

        self.getBinaryRobotRegions()
        return success, iteration

    def getBinaryRobotRegions(self):
        ind = np.where(self.assg_mat < self.num_uavs)
        temp = (self.assg_mat[ind].astype(int),) + ind
        self.binary_uav_regions[temp] = True

    def generateRandomMatrix(self):
        RandomMatrix = np.zeros((self.num_rows, self.num_cols))
        RandomMatrix = 2 * self.randomLevel * np.random.uniform(0, 1, size=RandomMatrix.shape) + (1 - self.randomLevel)
        return RandomMatrix

    def FinalUpdateOnMetricMatrix(self, CM, RM, currentOne, CC):
        MMnew = np.zeros((self.num_rows, self.num_cols))
        MMnew = currentOne * CM * RM * CC

        return MMnew

    def IsThisAGoalState(self, thresh, connectedRobotRegions):
        for r in range(self.num_uavs):
            if (
                np.absolute(self.DesireableAssign[r] - self.ArrayOfElements[r]) > thresh
                or not connectedRobotRegions[r]
            ):
                return False
        return True

    def update_connectivity(self):
        self.connectivity = np.zeros((self.num_uavs, self.num_rows, self.num_cols), dtype=np.uint8)
        for i in range(self.num_uavs):
            mask = np.where(self.assg_mat == i)
            self.connectivity[i, mask[0], mask[1]] = 255

    # Construct Assignment Matrix
    def construct_Assignment_Matrix(self):
        Notiles = self.num_rows * self.num_cols
        fair_division = 1 / self.num_uavs
        effectiveSize = Notiles - self.num_uavs - len(self.obstacles_positions)
        termThr = 0

        if effectiveSize % self.num_uavs != 0:
            termThr = 1

        DesireableAssign = np.zeros(self.num_uavs)
        MaximunDist = np.zeros(self.num_uavs)
        MaximumImportance = np.zeros(self.num_uavs)
        MinimumImportance = np.zeros(self.num_uavs)

        for i in range(self.num_uavs):
            DesireableAssign[i] = effectiveSize * self.portions[i]
            MinimumImportance[i] = sys.float_info.max
            if DesireableAssign[i] != int(DesireableAssign[i]) and termThr != 1:
                termThr = 1

        AllDistances = np.zeros((self.num_uavs, self.num_rows, self.num_cols))
        TilesImportance = np.zeros((self.num_uavs, self.num_rows, self.num_cols))

        for x in range(self.num_rows):
            for y in range(self.num_cols):
                tempSum = 0
                for r in range(self.num_uavs):
                    AllDistances[r, x, y] = euclidian_distance_points2d(
                        np.array(self.init_positions[r]), np.array((x, y))
                    )  # E!
                    if AllDistances[r, x, y] > MaximunDist[r]:
                        MaximunDist[r] = AllDistances[r, x, y]
                    tempSum += AllDistances[r, x, y]

                for r in range(self.num_uavs):
                    if tempSum - AllDistances[r, x, y] != 0:
                        TilesImportance[r, x, y] = 1 / (tempSum - AllDistances[r, x, y])
                    else:
                        TilesImportance[r, x, y] = 1
                    # Todo FixMe!
                    if TilesImportance[r, x, y] > MaximumImportance[r]:
                        MaximumImportance[r] = TilesImportance[r, x, y]

                    if TilesImportance[r, x, y] < MinimumImportance[r]:
                        MinimumImportance[r] = TilesImportance[r, x, y]

        return (
            AllDistances,
            termThr,
            Notiles,
            DesireableAssign,
            TilesImportance,
            MinimumImportance,
            MaximumImportance,
        )

    def calculateCriterionMatrix(
        self,
        TilesImportance,
        MinimumImportance,
        MaximumImportance,
        correctionMult,
        smallerthan_zero,
    ):
        returnCrit = np.zeros((self.num_rows, self.num_cols))
        if self.importance:
            if smallerthan_zero:
                returnCrit = (TilesImportance - MinimumImportance) * (
                    (correctionMult - 1) / (MaximumImportance - MinimumImportance)
                ) + 1
            else:
                returnCrit = (TilesImportance - MinimumImportance) * (
                    (1 - correctionMult) / (MaximumImportance - MinimumImportance)
                ) + correctionMult
        else:
            returnCrit[:, :] = correctionMult

        return returnCrit

    def NormalizedEuclideanDistanceBinary(self, RobotR, BinaryMap):
        distRobot = cv2.distanceTransform(
            inverse_binary_map_as_uint8(BinaryMap),
            distanceType=2,
            maskSize=0,
            dstType=5,
        )
        MaxV = np.max(distRobot)
        MinV = np.min(distRobot)

        # Normalization
        if RobotR:
            distRobot = (distRobot - MinV) * (1 / (MaxV - MinV)) + 1
        else:
            distRobot = (distRobot - MinV) * (1 / (MaxV - MinV))

        return distRobot
