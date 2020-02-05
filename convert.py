import json


def get_data(data_path):
    f = open(data_path,"r")
    json_data = f.read()
    data = json.loads(json_data)

    for rounds in data:
        for round in rounds:
            for snap in round:
                if snap=='mapName':
                    mapName=round[snap]
                elif snap == 'num':
                    roundNum = round[snap]
                elif snap=='winnerSide':
                    winnerSide=round[snap]
                elif snap=='bombPosition':
                    bombPosition=round[snap]
                else:
                    postPlantStatus=round[snap]
                    for pps in postPlantStatus.items():
                       if pps=='second':
                           second=postPlantStatus[pps]
                       elif pps=='TsideAlive':
                            TsideAlive=postPlantStatus[pps]
                       elif pps == 'CTsideAlive':
                            CTsideAlive = postPlantStatus[pps]
                       else:
                           alivePlayers = postPlantStatus[pps]
                           for alivePlayer in alivePlayers:
                               s=2









if __name__ == '__main__':
    data_path = r"C://Users//info//Downloads//Output.json"
    get_data(data_path)