# dataDict = {
#     'winnerSide': [], 'secondAfterBomb': [], 'alive_num_CT_T': [],  # 'avgDistT':[],'avgDistCT':[],
#     'TP1dist': [], 'TP2dist': [], 'TP3dist': [], 'TP4dist': [], 'TP5dist': [],
#     'CTP1dist': [], 'CTP2dist': [], 'CTP3dist': [], 'CTP4dist': [], 'CTP5dist': [],
#     'TP1weapon': [], 'TP2weapon': [], 'TP3weapon': [], 'TP4weapon': [], 'TP5weapon': [],
#     'CTP1weapon': [], 'CTP2weapon': [], 'CTP3weapon': [], 'CTP4weapon': [], 'CTP5weapon': [],
# }
import time

dataDict = {
    'winnerSide': [], 'secondAfterBomb': [], 'alive_num_CT_T': [],
    'BPDWithAK47T': [], 'BPDWithAWPT': [], 'BPDWithM4A4T': [], 'BPDWithDeagleT': [], 'BPDWithGlockT': [], 'BPDWithM4A1T': [],
    'PPDWithAK47T': [], 'PPDWithAWPT': [], 'PPDWithM4A4T': [], 'PPDWithDeagleT': [], 'PPDWithGlockT': [], 'PPDWithM4A1T': [],
    'BPDWithAK47CT': [], 'BPDWithAWPCT': [], 'BPDWithM4A4CT': [], 'BPDWithDeagleCT': [], 'BPDWithGlockCT': [], 'BPDWithM4A1CT': [],
    'PPDWithAK47CT': [], 'PPDWithAWPCT': [], 'PPDWithM4A4CT': [], 'PPDWithDeagleCT': [], 'PPDWithGlockCT': [], 'PPDWithM4A1CT': [],
    'PPDWithMoreKillT': [], 'PPDWithMoreKillCT': [],
}

round_player_alive_num = {'T': 0, 'CT': 0}

alive_num_CT_T = {
    0: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
    1: {0: 10, 1: 11, 2: 12, 3: 13, 4: 14, 5: 15},
    2: {0: 20, 1: 21, 2: 22, 3: 23, 4: 24, 5: 25},
    3: {0: 30, 1: 31, 2: 32, 3: 33, 4: 34, 5: 35},
    4: {0: 40, 1: 41, 2: 42, 3: 43, 4: 44, 5: 45},
    5: {0: 50, 1: 51, 2: 52, 3: 53, 4: 54, 5: 55},
}

weapons =[303, 309, 304, 4, 2]
weapons_mapping = {303: 'AK47', 309: 'AWP', 304: 'M4A4', 4: 'Deagle', 2: 'Glock', 305: 'M4A1'}

max_bomb_player_dist_for_weapon = 200
max_player_player_dist_for_weapon = 200
max_player_player_dist_for_kill = 200



