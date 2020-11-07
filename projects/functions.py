import numpy as np
from numpy.linalg import solve
from scipy.sparse.csgraph._shortest_path import floyd_warshall

from pathPlanning.aStar import AStarPlanner
from pathPlanning.floyd import calc_distance_mat
from scipy.sparse import csr_matrix


def get_coefficients_from_certain_points():
    point1_demo = [1359, 520]
    point1_qgis = [922, -233]
    point2_demo = [1040, -2376]
    point2_qgis = [867, -818]

    # c_demo * a + b = c_qgis
    coeffients_matrix = np.mat([[point1_demo[0], 1], [point2_demo[0], 1]])  # 系数矩阵
    constant_matrix = np.mat([point1_qgis[0], point2_qgis[0]]).T  # 常数项列矩阵
    solutionX = solve(coeffients_matrix, constant_matrix)  # 方程组的解
    ax = float(solutionX[0])
    bx = float(solutionX[1])

    coeffients_matrix = np.mat([[point1_demo[1], 1], [point2_demo[1], 1]])  # 系数矩阵
    constant_matrix = np.mat([point1_qgis[1], point2_qgis[1]]).T  # 常数项列矩阵
    solutionY = solve(coeffients_matrix, constant_matrix)  # 方程组的解
    ay = float(solutionY[0])
    by = float(solutionY[1])
    return ax, ay, bx, by


def get_manhattan_distance(pos1, pos2):
    x1 = pos1[0]
    x2 = pos2[0]
    y1 = pos1[1]
    y2 = pos1[1]

    dist = abs(x1 - x2) + abs(y1 - y2)

    return dist


def get_transformed_coordinates(raw_obstacle_x, raw_obstacle_y, ax, ay, bx, by):
    obstacle_x = raw_obstacle_x * ax + bx
    obstacle_y = raw_obstacle_y * ay + by
    return obstacle_x, obstacle_y


def get_reachable_list(data_path):
    ax, ay, bx, by = get_coefficients_from_certain_points()
    file = open(data_path, "r", encoding='utf-8')
    line = file.readline()
    reachable_list = []
    while line:
        split_data = line.split(", ")
        if len(split_data) == 2:
            raw_reachable_x = float(split_data[0])
            raw_reachable_y = float(split_data[1])
            reachable_x, reachable_y = get_transformed_coordinates(raw_reachable_x, raw_reachable_y, ax, ay, bx, by)
            reachable_coordinate = [reachable_x, reachable_y]
            if reachable_coordinate:
                reachable_list.append(reachable_coordinate)
        line = file.readline()
    file.close()
    return reachable_list


def get_opposite_side_players(alive_players, player):
    opposite_side_players = []
    for alive_player in alive_players:
        if alive_player['Name'] != player['Name']:
            if alive_player['Side'] != player['Side']:
                opposite_side_players.append(alive_player)

    return opposite_side_players


def get_obstacles_list(data_path):
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
    return obstacles_list


def get_path_planning(start_position, goal_position, ox, oy, a_star, ax, ay, bx, by):
    sx = start_position[0]
    sy = start_position[1]
    gx = goal_position[0]
    gy = goal_position[1]
    # sx, sy = get_transformed_coordinates(raw_sx, raw_sy, ax, ay, bx, by)
    # gx, gy = get_transformed_coordinates(raw_gx, raw_gy, ax, ay, bx, by)

    if abs(start_position[0] - goal_position[0]) > 1.0 and abs(start_position[1] - goal_position[1]) > 1.0:
        path_cost, rx, ry = a_star.planning(sx, sy, gx, gy)

        if path_cost == 0:
            path_cost = get_manhattan_distance(start_position, goal_position)
        return path_cost

    else:
        return 0


def get_FW(obstacles_list):
    file = open(obstacles_list, "r", encoding='utf-8')
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
    dist_matrix, predecessors = floyd_warshall(csgraph=graph, directed=False, return_predecessors=True)
    return dist_matrix


def get_a_star(obstacles_list):
    grid_size = 10
    robot_radius = 10

    ox, oy = [], []
    length = len(obstacles_list)
    for i in range(length):
        ox.append(obstacles_list[i][0])
        oy.append(obstacles_list[i][1])

    a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
    return a_star, ox, oy


def get_post_plant_total_kill(alive_players):
    total_kill = 0
    for i in range(len(alive_players)):
        kill = alive_players[i]['Kills']
        total_kill = total_kill + kill
    return total_kill


def get_post_plant_avg_kill(alive_players):
    total_kill = get_post_plant_total_kill(alive_players)
    if len(alive_players) != 0:
        avg_kill = total_kill / len(alive_players)
    else:
        avg_kill = 0
    return avg_kill


def get_empty_dict():
    PPDWithMoreKill = {'T': 0, 'CT': 0}
    PPDWithMoreWeapon = {'T': {303: 0, 309: 0, 304: 0, 4: 0, 2: 0, 305: 0},
                         'CT': {303: 0, 309: 0, 304: 0, 4: 0, 2: 0, 305: 0}}
    BPDWithMoreWeapon = {'T': {303: 0, 309: 0, 304: 0, 4: 0, 2: 0, 305: 0},
                         'CT': {303: 0, 309: 0, 304: 0, 4: 0, 2: 0, 305: 0}}

    return PPDWithMoreKill, PPDWithMoreWeapon, BPDWithMoreWeapon
