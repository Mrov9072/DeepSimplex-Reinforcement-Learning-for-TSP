import numpy as np
N_CITIES = 100

def generate_city_coords(env_data):
    """
    generate N_CITIES cities with coordinates being integer values
    representing the horizontal and vertical cells within the map grid
    with height and width equal to MAP_SIZE
    """
    c = np.random.randint(100, size=(N_CITIES, 2))
    env_data["x_coords"] = c[:,0]
    env_data["y_coords"] = c[:,1]