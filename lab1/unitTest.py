import numpy as np
import operator
import minatourMaze as mmz

def coordinateAddition(coord_1, coord_2):
    """ given coord_1, and coord2  two tuples, add componentwise, return a tuple
    """
    return tuple(map(operator.add, coord_1, coord_2))

# maze = np.array([[1,0],[1,1]])

# minaMaze = mmz.Maze(maze)

coord_1 = (1,2)
coord_2 = (3,4)

coord_3 = coordinateAddition(coord_1, coord_2)

print(coord_3)
