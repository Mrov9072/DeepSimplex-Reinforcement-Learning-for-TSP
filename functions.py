import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pulp as pl

def create_graph(N: int) -> (np.ndarray, np.ndarray):
    """
    Generates coordinates and distance matrix of a graph with N nodes.

    Parameters:
    N (int): The number of nodes in the graph.

    Returns:
    (np.ndarray, np.ndarray): A tuple containing two numpy arrays:
    - coords: The coordinates of the nodes generated randomly using np.random.rand.
    - dists: The distance matrix representing the pairwise distances between nodes.

    The function first generates N pairs of random coordinates for the nodes in a 2D space.
    Then, it calculates the pairwise Euclidean distances between the coordinates to create the
    distance matrix.
    
    Example:
    >>> coords, dists = create_graph(5)
    This will generate the coordinates and distance matrix for a graph with 5 nodes.
    """

    coords = np.random.rand(N, 2)
    dists = np.sqrt((coords[:, [0]] - coords[:, 0])**2 + (coords[:, [1]] - coords[:, 1])**2)
    return coords, dists

def plot_graph(coords: np.ndarray):
    """
    Plots a graph based on the given coordinates of nodes.

    Parameters:
    coords (np.ndarray): A numpy array containing the coordinates of nodes in a 2D space.

    This function takes the provided coordinates and creates a scatter plot of the nodes on a 2D space.

    Example:
    >>> plot_graph(coords)
    This will generate a plot of the graph based on the provided coordinates.
    """
     
    for i, (x, y) in enumerate(coords):
        plt.text(x+0.005, y+0.005, str(i), fontdict={'fontsize':16})
    plt.scatter(x=coords[:, 0], y=coords[:, 1])
    plt.show()

def LP_TSP_solve(coords: np.ndarray, dists: np.ndarray, show_result: bool = True) -> list:
    '''
    This function takes the coordinates and distances of a set of points 
    and uses a linear programming approach to solve the Traveling Salesman Problem (TSP) 
    to find the shortest path cycle.

    Args:
    - coords: A numpy array containing the coordinates of each point.
    - dists: A numpy array containing the distances between each pair of points.
    - show_result: A boolean indicating whether to display the graph as a result. Defaults to True.

    Returns:
    - A list representing the minimum path of the cycle.

    '''
    n = len(dists)

    points = coords.copy()
    init_graph = nx.from_numpy_array(dists)

    model = pl.LpProblem(name="tsp", sense=pl.LpMinimize)
    solver = pl.PULP_CBC_CMD(msg=False)

    x = [pl.LpVariable(name=f'x_{i:03}_{j:03}', cat='Binary') for i in range(n) for j in range(n)]
  
    for i in range(n):
        model += pl.lpSum([x[i * n + j if i < j else j * n + i] for j in range(n) if i != j]) == 2

    model += pl.lpSum([init_graph[i][j]['weight'] * x[i * n + j if i < j else j * n + i] for i in range(n) for j in range(n) if i < j])

    step = 0
    while True:
        
        status = model.solve(solver)   
        step += 1

        graph_result = nx.Graph() 
        a = 0
        for i, v in enumerate(model.variables()):
            
            int_val = round(v.value())
            if int_val > 0:                    
                temp_name = v.name.split('_')
                ii, jj = int(temp_name[1]), int(temp_name[2])
                graph_result.add_edge(ii, jj)  
         
        result_sets = list(nx.connected_components(graph_result))
        qty_sets = len(result_sets)
        
        if qty_sets == 1:
            break
            
        if qty_sets == 2:
            model += pl.lpSum([x[ii * n + jj if ii<jj else jj * n + ii] for i, vi in enumerate(result_sets) for j, vj in enumerate(result_sets) if i < j for ii in vi for jj in vj]) >= 2
        else:
            for i, vi in enumerate(result_sets):
                model += pl.lpSum([x[ii * n + jj if ii<jj else jj * n + ii] for j, vj in enumerate(result_sets) if i != j for ii in vi for jj in vj]) >= 2            

    min_cycle = nx.find_cycle(graph_result, 0)
    min_path = [i[0] for i in min_cycle]

    if show_result:
        plt.axis ("equal")
        nx.draw(init_graph, points, width=0.5, edge_color="#C0C0C0", with_labels=True, node_size=400, font_size=14)
        nx.draw(graph_result, points, width=2, edge_color="red", style="-", with_labels=False, node_size=0, alpha=1)
        plt.show()

    return min_path