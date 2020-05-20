
data_text = "E://Thesis//list.txt"
file = open(data_text, "r", encoding='utf-8')
line = file.readline()
obstacles_list = []
while line:
    split_data = line.split(",")
    if len(split_data) == 3:
        obstacle_x = int(round(float(split_data[0]), 0))
        obstacle_y = int(round(float(split_data[1]), 0))
        radius = 5
        obstacles_coordinate = (obstacle_x, obstacle_y, radius)
        if obstacles_coordinate:
            obstacles_list.append(obstacles_coordinate)
    line = file.readline()
file.close()
print(obstacles_list)






