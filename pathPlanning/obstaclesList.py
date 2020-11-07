import time

import numpy as np
import math

from pathPlanning.floyd import Floyd

inf = 9999999
reso = 10
rr = 10


def calc_grid_position(index, minp):
    pos = index * reso + minp
    return pos


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
    print("done getting graph: " + time.asctime(time.localtime(time.time())))
    graph = distance_mat
    nodeNum = 72 * 84
    shortestPath = [[0 for i in range(nodeNum)] for j in range(nodeNum)]
    pathCount = [[0 for i in range(nodeNum)] for j in range(nodeNum)]

    Floyd(graph, shortestPath, pathCount)
    print(shortestPath)
    print(pathCount)
    print("finish: " + time.asctime(time.localtime(time.time())))
