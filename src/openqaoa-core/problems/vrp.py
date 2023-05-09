import numpy as np
import itertools 
from matplotlib import colormaps 
import copy 
import networkx as nx
import matplotlib.pyplot as plt
from docplex.mp.model import Model
from typing import Union
from .problem import Problem
from .converters import FromDocplex2IsingModel
from .qubo import QUBO


class VRPMatrix(Problem):
    
    """
    Creates an instance of the Vehicle Routing Problem (VRP) using a distance matrix, graph, or coordinates.
    
    Parameters
    ----------
    n_vehicles: int
        The number of vehicles used in the solution 
    depot: int, optional, default=0
        The node where all the vehicles leave for and return after.
    distance_matrix: list, optional
        Distance matrix representing the problem.
    G: nx.Graph, optional
        Graph representing the problem.
    pos: list, optional
        Positions of the nodes used to calculate the distance matrix.
    subtours: list, optional, default=-1
        If -1: All the possible subtours are added to the constraints. Avoid it for large instances.
        If there are subtours that want to be avoided in the solution, e.g., an 8 nodes
        VRP with an optimal solution showing subtour between nodes 4, 5, and 8 can be
        avoided introducing the constraint subtours=[[4,5,8]]. For additional information
        about subtours refer to https://de.wikipedia.org/wiki/Datei:TSP_short_cycles.png
    method: str, optional, default="slack"
        Two available methods for the inequality constraints ["slack", "unbalanced"]
        For 'unblanced', see https://arxiv.org/abs/2211.13914
    penalty: Union[int, float, list], optional, default=4
        Penalty for the constraints. If the method is 'unbalanced', three values are needed,
        one for the equality constraints and two for the inequality constraints.

    Returns
    -------
        An instance of the VRP problem.
    """
    __name__ = "vehicle_routing_from_matrix"
    
    def __init__(
        self,
        distance_matrix: list = None,
        n_vehicles: int = 1,
        depot: int = 0,
        subtours: list = -1,
        method: str = "slack",
        penalty: Union[int, float, list] = 4,
    ):  
        self.distance_matrix = distance_matrix
        
        self.n_vehicles = n_vehicles
        self.depot = depot 
        self.subtours = subtours

        
        
        self.method = method
        if method == "unbalanced" and len(penalty) != 3:
            raise ValueError(
                "The penalty must have 3 parameters [lambda_0, lambda_1, lambda_2]"
            )
        else:
            self.penalty = penalty 

 

    @staticmethod
    def random_instance(  **kwargs):
        """
    Generates a random instance of the VRP problem.

    Parameters
    ----------
    **kwargs:
        Required keyword arguments are:
        n_nodes : int
            The number of nodes in the distance matrix.
        n_vehicles : int
            The number of vehicles used in the solution.
        depot : int, optional
            The node where all the vehicles leave for and return after. Default is 0.
        max_distance : int, optional
            The maximum distance between nodes. Default is 100.
        seed : int, optional
            Seed for the random number generator. Default is None.

    Returns
    -------
    VRPMatrix
        A randomly generated VRPMatrix instance.
    """
        n_nodes = kwargs.get("n_nodes", 6)
        n_vehicles = kwargs.get("n_vehicles", 2)
        depot = kwargs.get("depot", 0)
        max_distance = kwargs.get("max_distance", 100)
        seed = kwargs.get("seed", None) 
        method = kwargs.get("method", "slack")
        if method == "slack":
            penalty = kwargs.get("penalty", 4)
        elif method == "unbalanced":
            penalty = kwargs.get("penalty", [4, 1, 1])
        else:
            raise ValueError(f"The method '{method}' is not valid.")
        subtours = kwargs.get("subtours", -1)

        if seed:
            np.random.seed(seed)

        distance_matrix = np.random.randint(1, max_distance, size=(n_nodes, n_nodes))
        distance_matrix = (distance_matrix + distance_matrix.T) / 2 
        np.fill_diagonal(distance_matrix, 0) 
        distance_matrix = distance_matrix.tolist()
        
        return VRPMatrix( distance_matrix = distance_matrix, n_vehicles       = n_vehicles , depot = depot,   subtours= subtours, method =method, penalty =penalty ) 


    @property
    def docplex_model(self):
        
        """
        Creates a docplex model for the Vehicle Routing Problem (VRP) using the distance matrix and other parameters.

        Returns
        -------
        mdl: docplex.mp.model.Model
            The docplex model representing the VRP.
        """
        mdl = Model("VRP")
        num_nodes = len(self.distance_matrix) 
        # Variables: the edges between nodes for a symmetric problem.
        x = {
            (i, j): mdl.binary_var(name=f"x_{i}_{j}")
            for i in range(num_nodes - 1)
            for j in range(i + 1, num_nodes)
        }

        # Minimize distance traveled
        mdl.minimize(
            mdl.sum(
                self.distance_matrix[i][j] * x[(i, j)]
                for i in range(num_nodes - 1)
                for j in range(i + 1, num_nodes)
            )
        )

        # Constraints for 2 edges per node
        for i in range(num_nodes):
            if i != self.depot:
                mdl.add_constraint(
                    mdl.sum(
                        [x[tuple(sorted([i, j]))] for j in range(num_nodes) if i != j]
                    )
                    == 2
                )
            else:
                mdl.add_constraint(
                    mdl.sum(
                        [x[tuple(sorted([i, j]))] for j in range(num_nodes) if i != j]
                    )
                    == 2 * self.n_vehicles
                )

        # Subtour constraints
        if self.subtours == -1:
            list_subtours = [[i for i in range(num_nodes) if i != self.depot]]
            for nodes in list_subtours:
                for i in range(3, num_nodes - 2 * self.n_vehicles):
                    for subtour in itertools.combinations(nodes, i):
                        tour = sorted(subtour)
                        n_subtour = len(subtour)
                        mdl.add_constraint(
                            mdl.sum(
                                [
                                    x[(tour[i], tour[j])]
                                    for i in range(n_subtour - 1)
                                    for j in range(i + 1, n_subtour)
                                ]
                            )
                            <= n_subtour - 1
                        )
        elif isinstance(self.subtours, list):
            list_subtours = self.subtours
            for subtour in list_subtours:
                tour = sorted(subtour)
                n_subtour = len(subtour)
                if n_subtour != 0:
                    mdl.add_constraint(
                        mdl.sum(
                            [
                                x[(tour[i], tour[j])]
                                for i in range(n_subtour)
                                for j in range(i + 1, n_subtour)
                            ]
                        )
                        <= n_subtour - 1
                    )
        else:
            raise ValueError(f"{type(self.subtour)} is not a valid format for the subtours.")
        
        return mdl 
    
    
    @property 
    def qubo(self):
        """
    Returns the QUBO encoding of this problem.
    Returns
    -------
        The QUBO encoding of this problem.
    """
        cplex_model = self.docplex_model
        if self.method == "slack":
            qubo_docplex = FromDocplex2IsingModel(
                cplex_model          ,multipliers=self.penalty        ).ising_model
        elif self.method == "unbalanced":
            qubo_docplex = FromDocplex2IsingModel(
                cplex_model,
                multipliers=self.penalty[0],
                unbalanced_const=True,
                strength_ineq=self.penalty[1:],
            ).ising_model
        return QUBO(
            qubo_docplex.n,
            qubo_docplex.terms + [[]],
            qubo_docplex.weights + [qubo_docplex.constant],
            self.problem_instance,
        )



    def classical_solution(self, string: bool = False):
        """
        Return the classical solution of the vehicle routing problem
        Parameters
        ----------
        string : bool, optional
            If the solution is returned as a string. The default is False.
        Raises
        ------
        ValueError
            A flag if docplex does not find a valid solution.
        Returns
        -------
        solution : Union[str, dict]
            The classical solution of the specific problem as a string or a dict.
        """
        cplex_model = self.docplex_model
        docplex_sol = cplex_model.solve()

        if docplex_sol is None:
            raise ValueError(f"Solution not found: {cplex_model.solve_details.status}.")

        if string:
            solution = ""
        else:
            solution = {}
        for var in cplex_model.iter_binary_vars():
            if string:
                solution += str(round(docplex_sol.get_value(var)))
            else:
                solution[var.name] = round(docplex_sol.get_value(var))
        return solution 
    def plot_solution(self, solution: Union[dict, str], ax=None, edge_width=4, colors=None): 

        """
    A visualization method for the vehicle routing problem solution.
    Parameters
    ----------
    solution : Union[dict, str]
        The solution of the specific vehicle routing problem as a string or dictionary.
    ax : matplotlib axes, optional
        The default is None.
    Returns
    -------
    fig : matplotlib.pyplot.Figure()
        The graph visualization of the solution.
    """

        colors = colormaps["jet"] if colors is None else colors
        if type(colors) is list and len(colors) != self.n_vehicles:
            raise ValueError(f"The length of colors {len(colors)} and the number of vehicles {self.n_vehicles} do not match")

        if isinstance(solution, str):
            sol = {}
            for n, var in enumerate(self.docplex_model.iter_binary_vars()):
                sol[var.name] = int(solution[n])
            solution = sol
        paths_and_subtours = self.paths_subtours(solution)
        paths = paths_and_subtours["paths"]
        subtours = paths_and_subtours["subtours"]
        tours_color = {}
        for vehicle in range(self.n_vehicles):
            for i, j in paths[vehicle]:
                if type(colors) is list:
                    tours_color[f"x_{i}_{j}"] = colors[vehicle]
                else:
                    tours_color[f"x_{i}_{j}"] = colors((vehicle + 1) / self.n_vehicles)
        for subtour in subtours.keys():
            for i, j in subtours[subtour]:
                tours_color[f"x_{i}_{j}"] = "black"
        color_node = "#5EB1EB"
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None
        #pos = self.pos
        num_vertices = len(self.distance_matrix)
        G = nx.Graph()
        G.add_nodes_from(range(num_vertices))
        edge_color = []
        for i in range(num_vertices - 1):
            for j in range( i + 1, num_vertices):
                if i != j and int(solution[f"x_{i}_{j}"]):
                    edge_color.append(tours_color[f"x_{i}_{j}"])
                    G.add_edge(i, j)
        nx.draw(
            G,
            width=edge_width,
            edge_color=edge_color,
            node_color=color_node,
            alpha=0.8,
            labels={i: str(i) for i in range(num_vertices)},
            ax=ax,
            edgecolors="black",
        )
        return fig

    def paths_subtours(   self, sol):
        n_nodes = len(self.distance_matrix)
        vars_list = []
        for i in range(n_nodes -1):
            for j in range( i + 1, n_nodes):
                if round(sol[f"x_{i}_{j}"]):
                    vars_list.append([i, j])

    # ----------------  vehicle routing problem solutions -----------------
        paths = {}
        for i in range(self.n_vehicles):
            paths[i] = []
            nodes = [self.depot]
            count_depot = 0
            max_edges = n_nodes * (n_nodes - 1)
            while count_depot < 2:
                for edge in vars_list:
                    if nodes[-1] in edge:
                        if self.depot in edge:
                            count_depot += 1
                        paths[i].append(edge)
                        vars_list.remove(edge)
                        nodes.append(edge[0] if edge[0] != nodes[-1] else edge[1])
                        break
                max_edges -= 1
                if max_edges < 0:
                    raise ValueError(
                        "Solution provided does not fulfill all the path conditions."
                    )

        # ----------------            subtours                -----------------
        subtours = {}
        i = 0
        max_edges = n_nodes * (n_nodes - 1)          /            2
        while len(vars_list) > 0:
            subtours[i] = [vars_list.pop(0)]
            nodes = copy.copy(subtours[i][0])
            count = 1
            while count < 2:
                for edge in vars_list:
                    if nodes[-1] in edge:
                        if nodes[0] in edge:
                            count += 1
                        subtours[i].append(edge)
                        vars_list.remove(edge)
                        nodes.append(edge[0] if edge[0] != nodes[-1] else edge[1])
                        break
                max_edges -= 1
                if max_edges < 0:
                    raise ValueError("The subtours in the solution are broken.")
            i += 1
        return {"paths": paths, "subtours": subtours}
