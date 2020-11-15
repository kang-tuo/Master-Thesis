
import json
import pandas as pd
from Modelling.main.const import *
from Modelling.main.functions import *


class ConvertJsonToDataframe():

    def __init__(self):
        self.round_player_alive_num = round_player_alive_num
        self.dataDict = dataDict

    def main(self, data_path, obstacles_file_path):
        f = open(data_path, "r", encoding='utf-8')
        json_data = f.read()
        data = json.loads(json_data)

        ax, ay, bx, by = get_coefficients_from_certain_points()
        obstacles_list = get_obstacles_list(obstacles_file_path)
        a_star, ox, oy = get_a_star(obstacles_list)

        print(time.asctime(time.localtime(time.time())))
        for rounds in data:
            for round in rounds:
                self.get_round_info(round, ax, ay, bx, by)
                for post_plant in self.post_plant_status:
                    self.get_post_plant_info(post_plant)
                    self.round_general_append(post_plant)
                    self.reset_player_num()
                    self.reset_player_further_num()

                    for alive_player in self.alive_players:
                        self.get_alive_player_info(alive_player, ax, ay, bx, by)
                        if self.player_num[self.player_side] > 5:
                            break
                        self.player_bomb_distance = get_path_planning(self.player_position, self.bomb_position,
                                                                      ox, oy, a_star, ax, ay, bx, by)
                        if self.player_bomb_distance > max_bomb_player_dist:
                            self.player_further_num[self.player_side] += 1

                        self.op_side_num = 0
                        opposite_side_players = get_opposite_side_players(self.alive_players, alive_player)

                        for op_player in opposite_side_players:
                            self.op_side_num += 1
                            pos_x, pos_y = get_transformed_coordinates(op_player['Position'][0],
                                                                       op_player['Position'][1], ax, ay, bx, by)
                            self.player_player_distance = get_path_planning(self.player_position, [pos_x, pos_y],
                                                                       ox, oy, a_star, ax, ay, bx, by)
                            if self.op_side_num > 5:
                                break
                            self.player_dist_append()
                        self.player_append(alive_player)
                    self.append_player_further()
                    self.get_null()

        print(time.asctime(time.localtime(time.time())))
        data_df = pd.DataFrame.from_dict(self.dataDict)

        print(data_df)
        pickle_path = "E://Thesis//data8.pkl"
        data_df.to_pickle(pickle_path, protocol=3)
        return data_df

    def get_null(self):
        data_length = len(self.dataDict['winnerSide'])
        for self.col in self.dataDict:
            actual_data_length = len(self.dataDict[self.col])
            if actual_data_length < data_length:
                self.dataDict[self.col].append(0)

    def reset_player_num(self):
        self.player_num = {'T': 0, 'CT': 0}

    def reset_player_further_num(self):
        self.player_further_num = {'T': 0, 'CT': 0}

    def get_alive_player_info(self, alive_player, ax, ay, bx, by):
        self.player_name = alive_player['Name']
        self.player_side = alive_player['Side']
        self.player_num[self.player_side] += 1
        pos_x, pos_y = get_transformed_coordinates(alive_player['Position'][0],
                                                             alive_player['Position'][1], ax, ay, bx, by)
        velocity_x, velocity_y = get_transformed_coordinates(alive_player['Velocity'][0],
                                                             alive_player['Velocity'][1], ax, ay, bx, by)
        self.player_velocity = [velocity_x, velocity_y]
        self.player_position = [pos_x, pos_y]

        if self.player_side == 'T':
            self.op_side = "CT"
        if self.player_side == 'CT':
            self.op_side = "T"


    def get_feature_col_name(self, feature):
        feature_col_name = self.player_side + str(self.player_num[self.player_side]) + "_" + feature
        return feature_col_name

    def get_player_dist_col_name(self):
        player_dist_col_name = self.player_side + str(self.player_num[self.player_side]) + "_" + \
                               self.op_side + str(self.op_side_num) + "_distance"
        return player_dist_col_name

    def get_bomb_dist_col_name(self):
        bomb_dist_col_name = self.player_side + str(self.player_num[self.player_side]) + "_B_distance"

        return bomb_dist_col_name

    def player_append(self, alive_player):
        for feature in features:
            feature_col = self.get_feature_col_name(feature)
            feature_val = alive_player[feature]
            if feature == 'Velocity':
                feature_val = self.player_velocity
            if feature == 'Position':
                feature_val = self.player_position
            self.dataDict[feature_col].append(feature_val)

        bomb_dist_col_name = self.get_bomb_dist_col_name()
        self.dataDict[bomb_dist_col_name].append(self.player_bomb_distance)

    def append_player_further(self):
        self.dataDict["player_further_num_CT"].append(self.player_further_num["CT"])
        self.dataDict["player_further_num_T"].append(self.player_further_num["T"])

    def player_dist_append(self):
        player_dist_col_name = self.get_player_dist_col_name()
        self.dataDict[player_dist_col_name].append(self.player_player_distance)

    def get_post_plant_info(self, post_plant):
        second = post_plant['second']
        second_string = str(second)
        self.second = int(second_string[0:2])
        if self.second == 50:
            self.second = 5
        self.round_player_alive_num['T'] = post_plant['TsideAlive']
        self.round_player_alive_num['CT'] = post_plant['CTsideAlive']
        self.alive_players = post_plant['alivePlayers']

    def get_round_info(self, round, ax, ay, bx, by):
        self.startParseTime = round['ParseStartTime']
        self.map_name = round['mapName']
        self.round_num = round['num']
        self.winner_side = round['winnerSide']
        self.post_plant_status = round['postPlantStatus']
        raw_bomb_position = round['bombPosition']
        x, y = get_transformed_coordinates(raw_bomb_position[0], raw_bomb_position[1], ax, ay, bx, by)
        self.bomb_position = [x, y]

    def round_general_append(self, post_plant):
        self.dataDict['startParseTime'].append(self.startParseTime)
        self.dataDict['winnerSide'].append(self.winner_side)
        self.dataDict['secondAfterBomb'].append(self.second)
        self.dataDict['bomb_position'].append(self.bomb_position)

        CTNum = self.round_player_alive_num['CT']
        TNum = self.round_player_alive_num['T']
        self.dataDict['alive_num_CT'].append(CTNum)
        self.dataDict['alive_num_T'].append(TNum)
        self.dataDict['flash_CT'].append(post_plant['CTFlashBangs'])
        self.dataDict['flash_T'].append(post_plant['TFlashBangs'])
        self.dataDict['smoke_T'].append(post_plant['TSmokes'])
        self.dataDict['smoke_CT'].append(post_plant['CTSmokes'])
        self.dataDict['molo_CT'].append(post_plant['CTMolo'])
        self.dataDict['molo_T'].append(post_plant['TMolo'])


if __name__ == '__main__':
    data_path = r"C://Users//admin//Desktop//.json"
    obstacles_txt = "E://Thesis//obstacles_list.txt"

    TDF = ConvertJsonToDataframe()
    df = TDF.main(data_path, obstacles_txt)
