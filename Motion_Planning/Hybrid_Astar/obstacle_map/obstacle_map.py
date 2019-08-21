"""
Script for Obstacle map
19.08.16. Seong
"""

def map_maze(ox, oy):

    for i in range(60):
        ox.append(i)
        oy.append(0.0)
    for i in range(60):
        ox.append(60.0)
        oy.append(i)
    for i in range(61):
        ox.append(i)
        oy.append(60.0)
    for i in range(61):
        ox.append(0.0)
        oy.append(i)

    for i in range(40):
        ox.append(20.0)
        oy.append(i)
    for i in range(40):
        ox.append(40.0)
        oy.append(60.0 - i)

    for i in range(5, 16):
        ox.append(i)
        oy.append(30)
    for i in range(15, 26):
        ox.append(i)
        oy.append(40)
        
    for i in range(15, 26):
        ox.append(i)
        oy.append(45)

    for i in range(45, 61):
        ox.append(20)
        oy.append(i)

    for i in range(30, 61):
        ox.append(32)
        oy.append(i)
    for i in range(50, 61):
        ox.append(i)
        oy.append(40)

    for i in range(10, 31):
        ox.append(12)
        oy.append(i)
    for i in range(5, 21):
        ox.append(i)
        oy.append(30)

    return ox, oy

def map_road(ox, oy, pos_cars):
    """
    Make Obstacle map w.r.t ROI and surrounding cars
    """
    for i in range(-10, 11):
        ox.append(i)
        oy.append(50.0)
    for i in range(-10, 11):
        ox.append(i)
        oy.append(-30.0)
    for i in range(-30, 51):
        ox.append(-10.0)
        oy.append(i)
    for i in range(-30, 51):
        ox.append(10.0)
        oy.append(i)

    for pos_car in pos_cars:
        for i in range(pos_car[1]-2, pos_car[1]+2):
            ox.append(pos_car[0]-1)
            oy.append(i)
        for i in range(pos_car[1]-2, pos_car[1]+2):
            ox.append(pos_car[0]+1)
            oy.append(i)

    # # Road
    # for i in range(0, 51):
    #     ox.append(-10)
    #     oy.append(i)
    # for i in range(0, 51):
    #     ox.append(10)
    #     oy.append(i)

    # # Vehicles
    # for i in range(-5, -2):
    #     for j in range(0, 4):
    #         ox.append(i)
    #         oy.append(j)

    return ox, oy