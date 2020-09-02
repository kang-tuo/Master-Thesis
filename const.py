# dataDict = {
#     'winnerSide': [], 'secondAfterBomb': [], 'alive_num_CT_T': [],  # 'avgDistT':[],'avgDistCT':[],
#     'TP1dist': [], 'TP2dist': [], 'TP3dist': [], 'TP4dist': [], 'TP5dist': [],
#     'CTP1dist': [], 'CTP2dist': [], 'CTP3dist': [], 'CTP4dist': [], 'CTP5dist': [],
#     'TP1weapon': [], 'TP2weapon': [], 'TP3weapon': [], 'TP4weapon': [], 'TP5weapon': [],
#     'CTP1weapon': [], 'CTP2weapon': [], 'CTP3weapon': [], 'CTP4weapon': [], 'CTP5weapon': [],
# }
import time

# dataDict = { 'winnerSide': [], 'secondAfterBomb': [], 'alive_num_CT_T': [], 'BPDWithAK47T': [], 'BPDWithAWPT': [], 
# 'BPDWithM4A4T': [], 'BPDWithDeagleT': [], 'BPDWithGlockT': [], 'BPDWithM4A1T': [], 'PPDWithAK47T': [], 
# 'PPDWithAWPT': [], 'PPDWithM4A4T': [], 'PPDWithDeagleT': [], 'PPDWithGlockT': [], 'PPDWithM4A1T': [], 
# 'BPDWithAK47CT': [], 'BPDWithAWPCT': [], 'BPDWithM4A4CT': [], 'BPDWithDeagleCT': [], 'BPDWithGlockCT': [], 
# 'BPDWithM4A1CT': [], 'PPDWithAK47CT': [], 'PPDWithAWPCT': [], 'PPDWithM4A4CT': [], 'PPDWithDeagleCT': [], 
# 'PPDWithGlockCT': [], 'PPDWithM4A1CT': [], 'PPDWithMoreKillT': [], 'PPDWithMoreKillCT': [], } 

dataDict = {'startParseTime': [],
            'winnerSide': [], 'secondAfterBomb': [], 'bomb_position': [], 'alive_num_CT': [], 'alive_num_T': [],
            'flash_CT': [], 'flash_T': [], 'smoke_CT': [], 'smoke_T': [], 'molo_CT': [], 'molo_T': [],
            'T1_Position': [], 'T2_Position': [], 'T3_Position': [], 'T4_Position': [], 'T5_Position': [],
            'CT1_Position': [], 'CT2_Position': [], 'CT3_Position': [], 'CT4_Position': [], 'CT5_Position': [],
            'T1_Kills': [], 'T2_Kills': [], 'T3_Kills': [], 'T4_Kills': [], 'T5_Kills': [],
            'CT1_Kills': [], 'CT2_Kills': [], 'CT3_Kills': [], 'CT4_Kills': [], 'CT5_Kills': [],
            'T1_WeaponValue': [], 'T2_WeaponValue': [], 'T3_WeaponValue': [], 'T4_WeaponValue': [],
            'T5_WeaponValue': [],
            'CT1_WeaponValue': [], 'CT2_WeaponValue': [], 'CT3_WeaponValue': [], 'CT4_WeaponValue': [],
            'CT5_WeaponValue': [],
            'T1_Velocity': [], 'T2_Velocity': [], 'T3_Velocity': [], 'T4_Velocity': [], 'T5_Velocity': [],
            'CT1_Velocity': [], 'CT2_Velocity': [], 'CT3_Velocity': [], 'CT4_Velocity': [], 'CT5_Velocity': [],
            'T1_Money': [], 'T2_Money': [], 'T3_Money': [], 'T4_Money': [], 'T5_Money': [],
            'CT1_Money': [], 'CT2_Money': [], 'CT3_Money': [], 'CT4_Money': [], 'CT5_Money': [],
            'T1_B_distance': [], 'T2_B_distance': [], 'T3_B_distance': [], 'T4_B_distance': [], 'T5_B_distance': [],
            'CT1_B_distance': [], 'CT2_B_distance': [], 'CT3_B_distance': [], 'CT4_B_distance': [],
            'CT5_B_distance': [],
            'T1_CT1_distance': [], 'T2_CT1_distance': [], 'T3_CT1_distance': [], 'T4_CT1_distance': [],
            'T5_CT1_distance': [],
            'T1_CT2_distance': [], 'T2_CT2_distance': [], 'T3_CT2_distance': [], 'T4_CT2_distance': [],
            'T5_CT2_distance': [],
            'T1_CT3_distance': [], 'T2_CT3_distance': [], 'T3_CT3_distance': [], 'T4_CT3_distance': [],
            'T5_CT3_distance': [],
            'T1_CT4_distance': [], 'T2_CT4_distance': [], 'T3_CT4_distance': [], 'T4_CT4_distance': [],
            'T5_CT4_distance': [],
            'T1_CT5_distance': [], 'T2_CT5_distance': [], 'T3_CT5_distance': [], 'T4_CT5_distance': [],
            'T5_CT5_distance': [],
            'CT1_T1_distance': [], 'CT2_T1_distance': [], 'CT3_T1_distance': [], 'CT4_T1_distance': [],
            'CT5_T1_distance': [],
            'CT1_T2_distance': [], 'CT2_T2_distance': [], 'CT3_T2_distance': [], 'CT4_T2_distance': [],
            'CT5_T2_distance': [],
            'CT1_T3_distance': [], 'CT2_T3_distance': [], 'CT3_T3_distance': [], 'CT4_T3_distance': [],
            'CT5_T3_distance': [],
            'CT1_T4_distance': [], 'CT2_T4_distance': [], 'CT3_T4_distance': [], 'CT4_T4_distance': [],
            'CT5_T4_distance': [],
            'CT1_T5_distance': [], 'CT2_T5_distance': [], 'CT3_T5_distance': [], 'CT4_T5_distance': [],
            'CT5_T5_distance': [],
            'player_further_num_T': [], 'player_further_num_CT': []

            }

features = ['Position', 'Kills', 'WeaponValue', 'Velocity', 'Money']
round_player_alive_num = {'T': 0, 'CT': 0}

alive_num_CT_T = {
    0: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
    1: {0: 10, 1: 11, 2: 12, 3: 13, 4: 14, 5: 15},
    2: {0: 20, 1: 21, 2: 22, 3: 23, 4: 24, 5: 25},
    3: {0: 30, 1: 31, 2: 32, 3: 33, 4: 34, 5: 35},
    4: {0: 40, 1: 41, 2: 42, 3: 43, 4: 44, 5: 45},
    5: {0: 50, 1: 51, 2: 52, 3: 53, 4: 54, 5: 55},
}

weapons = [303, 309, 304, 4, 2]
weapons_mapping = {303: 'AK47', 309: 'AWP', 304: 'M4A4', 4: 'Deagle', 2: 'Glock', 305: 'M4A1'}

max_bomb_player_dist = 200
max_player_player_dist = 200
