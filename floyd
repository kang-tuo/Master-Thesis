import math
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall

inf = 9999999
reso = 10
rr = 10


def calc_grid_position(index, minp):
    pos = index * reso + minp
    return pos


def calc_index(x, y):
    xwidth = 84
    minx = 112
    miny = -873
    col = (x - minx) / reso
    row = (y - miny) / reso

    index = round(row * xwidth + col)
    if index >= 6048:
        index = 6047
    return index


def calc_distance_mat(ox, oy):
    minx = round(min(ox))
    miny = round(min(oy))
    maxx = round(max(ox) + 1)
    maxy = round(max(oy) + 1)

    xwidth = round((maxx - minx) / reso) + 1
    ywidth = round((maxy - miny) / reso) + 1

    # obstacle map generation
    obmap = [[False for i in range(ywidth)]
             for i in range(xwidth)]
    for ix in range(xwidth):
        x = calc_grid_position(ix, minx)
        for iy in range(ywidth):
            y = calc_grid_position(iy, miny)
            for iox, ioy in zip(ox, oy):
                d = math.hypot(iox - x, ioy - y)
                if d <= rr:
                    obmap[ix][iy] = True
                    break

    v_num = xwidth * ywidth
    distance_mat = np.full(shape=(v_num, v_num), fill_value=inf)

    for row in range(ywidth):
        for col in range(xwidth):
            index = row * xwidth + col
            distance_mat[index][index] = 0

            if obmap[col][row] is False:
                if col + 1 < xwidth and obmap[col + 1][row] is False:
                    index_right = row * xwidth + col + 1
                    distance_mat[index][index_right] = 10
                if col - 1 >= 0 and obmap[col - 1][row] is False:
                    index_left = row * xwidth + col - 1
                    distance_mat[index][index_left] = 10
                if row - 1 >= 0 and obmap[col][row - 1] is False:
                    index_upper = (row - 1) * xwidth + col
                    distance_mat[index][index_upper] = 10
                if row + 1 < ywidth and obmap[col][row + 1] is False:
                    index_lower = (row + 1) * xwidth + col
                    distance_mat[index][index_lower] = 10

    return distance_mat


def Floyd(Graph, ShortestPath, PathCount):
    NodeNum = len(Graph)

    lastShortestDistance = [[0 for i in range(NodeNum)] for j in range(NodeNum)]
    lastPathCount = [[0 for i in range(NodeNum)] for j in range(NodeNum)]

    currentShortestDistance = [[0 for i in range(NodeNum)] for j in range(NodeNum)]
    currentPathCount = [[0 for i in range(NodeNum)] for j in range(NodeNum)]

    for i in range(NodeNum):
        for j in range(NodeNum):
            lastShortestDistance[i][j] = Graph[i][j]
            if (Graph[i][j] > 0) and (Graph[i][j] < inf):
                lastPathCount[i][j] = 1
            else:
                lastPathCount[i][j] = 0
            currentShortestDistance[i][j] = lastShortestDistance[i][j]
            currentPathCount[i][j] = lastPathCount[i][j]

    for k in range(NodeNum):
        for i in range(NodeNum):
            if (i == k):
                continue
            for j in range(NodeNum):
                if (j == k):
                    continue
                if (lastShortestDistance[i][j] < lastShortestDistance[i][k] + lastShortestDistance[k][j]):
                    continue
                if (lastShortestDistance[i][j] == lastShortestDistance[i][k] + lastShortestDistance[k][j]):
                    currentShortestDistance[i][j] = lastShortestDistance[i][j]
                    currentPathCount[i][j] = lastPathCount[i][j] + lastPathCount[i][k] * lastPathCount[k][j]
                if (lastShortestDistance[i][j] > lastShortestDistance[i][k] + lastShortestDistance[k][j]):
                    currentShortestDistance[i][j] = lastShortestDistance[i][k] + lastShortestDistance[k][j]
                    currentPathCount[i][j] = lastPathCount[i][k] * lastPathCount[k][j]

        lastShortestDistance = currentShortestDistance
        lastPathCount = currentPathCount

    for i in range(NodeNum):
        for j in range(NodeNum):
            ShortestPath[i][j] = currentShortestDistance[i][j]
            PathCount[i][j] = currentPathCount[i][j]

    return None


if __name__ == '__main__':
    print("start time: " + time.asctime(time.localtime(time.time())))
    data_path = "E://Thesis//obstacles_list.txt"
    file = open(data_path, "r", encoding='utf-8')
    line = file.readline()
    obstacles_list = []

    while line:
        split_data = line.split(",")
        if len(split_data) == 3:
            obstacle_x = float(split_data[0])
            obstacle_y = float(split_data[1])
            obstacles_coordinate = [obstacle_x, obstacle_y]
            if obstacles_coordinate:
                obstacles_list.append(obstacles_coordinate)
        line = file.readline()
    file.close()

    ox, oy = [], []
    length = len(obstacles_list)
    for i in range(length):
        ox.append(obstacles_list[i][0])
        oy.append(obstacles_list[i][1])

    distance_mat = calc_distance_mat(ox, oy)
    graph = csr_matrix(distance_mat)
    print("done getting graph: " + time.asctime(time.localtime(time.time())))

    # graph = distance_mat
    # nodeNum = len(distance_mat[0])
    # shortestPath = [[0 for i in range(nodeNum)] for j in range(nodeNum)]
    # pathCount = [[0 for i in range(nodeNum)] for j in range(nodeNum)]
    #
    # Floyd(graph, shortestPath, pathCount)
    # print(shortestPath)
    # print(pathCount)
    dist_matrix, predecessors = floyd_warshall(csgraph=graph, directed=False, return_predecessors=True)
    print(dist_matrix)
    print("finish time: " + time.asctime(time.localtime(time.time())))
