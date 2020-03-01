import json
import math
import pandas as pd


dataDict = {
    'winnerSide': [], 'secondAfterBomb': [], 'aliveTNum': [], 'aliveCTNum': [],
    'TP1dist': [], 'TP2dist': [], 'TP3dist': [], 'TP4dist': [], 'TP5dist': [],
    'CTP1dist': [], 'CTP2dist': [], 'CTP3dist': [], 'CTP4dist': [], 'CTP5dist': [],
    'TP1weapon': [], 'TP2weapon': [], 'TP3weapon': [], 'TP4weapon': [], 'TP5weapon': [],
    'CTP1weapon': [], 'CTP2weapon': [], 'CTP3weapon': [], 'CTP4weapon': [], 'CTP5weapon': [],
} # 'map': [], 'round': [],

round_player_alive_num = {'T': 0, 'CT': 0}


class ConvertJsonToDataframe():

    def __init__(self):
        self.round_player_alive_num = round_player_alive_num
        self.dataDict = dataDict

    def main(self, data_path):
        f = open(data_path, "r", encoding='utf-8')
        json_data = f.read()
        data = json.loads(json_data)

        for rounds in data:
            for round in rounds:
                self.get_round_info(round)
                for post_plant in self.post_plant_status:
                    self.get_post_plant_info(post_plant)
                    self.round_general_append()
                    self.reset_player_num()
                    for alive_player in self.alive_players:
                        self.get_alive_player_info(alive_player)
                        if self.player_num[self.player_side] > 5:
                            break
                        self.get_col_names()
                        self.player_bomb_distance = self.get_euclidean_distance(self.player_position, self.bomb_position)
                        self.player_append()
                    self.get_null()

        data_df = pd.DataFrame.from_dict(self.dataDict)
        #print(data_df)
        return data_df

    def get_null(self):
        data_length = len(self.dataDict['winnerSide'])
        for self.col in self.dataDict:
            actual_data_length = len(self.dataDict[self.col])
            if actual_data_length < data_length:
                if "dist" in self.col:
                    self.dataDict[self.col].append(9999999999999999999)
                if "weapon" in self.col:
                    self.dataDict[self.col].append(0)


    def get_col_names(self):
        self.dist_col_name = self.player_side + "P" + str(self.player_num[self.player_side]) + "dist"
        self.weapon_col_name = self.player_side + "P" + str(self.player_num[self.player_side]) + "weapon"

    def reset_player_num(self):
        self.player_num = {'T': 0, 'CT': 0}

    def get_alive_player_info(self, alive_player):
        self.player_name = alive_player['Name']
        self.player_side = alive_player['Side']
        self.player_num[self.player_side] += 1
        self.player_position = alive_player['Position']
        self.player_weapon = alive_player['Weapon']
        self.player_weapon_value = alive_player['WeaponValue']
        self.player_kills = alive_player['Kills']
        self.player_money = alive_player['Money']
        self.player_velocity = alive_player['Velocity']


    def get_post_plant_info(self, post_plant):
        second = post_plant['second']
        second_string = str(second)
        self.second = int(second_string[0:2])
        if self.second == 50:
            self.second = 5
        self.round_player_alive_num['T'] = post_plant['TsideAlive']
        self.round_player_alive_num['CT'] = post_plant['CTsideAlive']
        self.alive_players = post_plant['alivePlayers']


    def get_round_info(self, round):
        self.map_name = round['mapName']
        self.round_num = round['num']
        winner_side = round['winnerSide']
        if winner_side=='T':
            self.winner_side = 1
        if winner_side == 'CT':
            self.winner_side = -1
        self.bomb_position = round['bombPosition']
        self.post_plant_status = round['postPlantStatus']

    def player_append(self):
        self.dataDict[self.dist_col_name].append(self.player_bomb_distance)
        self.dataDict[self.weapon_col_name].append(self.player_weapon)

    def round_general_append(self):
        # self.dataDict['map'].append(self.map_name)
        # self.dataDict['round'].append(self.round_num)
        self.dataDict['winnerSide'].append(self.winner_side)
        self.dataDict['secondAfterBomb'].append(self.second)
        self.dataDict['aliveTNum'].append(self.round_player_alive_num['T'])
        self.dataDict['aliveCTNum'].append(self.round_player_alive_num['CT'])


    def get_euclidean_distance(self,pos1, pos2):
        x1 = pos1[0]
        x2 = pos2[0]
        y1 = pos1[1]
        y2 = pos1[1]

        x_pow = pow((x1 - x2), 2)
        y_pow = pow((y1 - y2), 2)

        dist = math.sqrt(x_pow+y_pow)

        return dist


data_path = r"C://Users//admin//Desktop//Output2.json"
TDF = ConvertJsonToDataframe()
dataset = TDF.main(data_path)
