# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP3. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
from multiprocessing import parent_process
import queue as que
import sys
from copy import deepcopy
from sysconfig import get_path
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)


# Feel free to use the code below as you wish
# Initialize it with a list/tuple of objectives
# Call compute_mst_weight to get the weight of the MST with those objectives
# TODO: hint, you probably want to cache the MST value for sets of objectives you've already computed...
# Note that if you want to test one of your search methods, please make sure to return a blank list
#  for the other search methods otherwise the grader will not crash.
class MST:
    def __init__(self, objectives):
        self.elements = {key: None for key in objectives}

        # TODO: implement some distance between two objectives
        # ... either compute the shortest path between them, or just use the manhattan distance between the objectives
        self.distances   = {
                (i, j): calc_manhattan_distance(i, j)
                for i, j in self.cross(objectives)
            }

    # Prim's algorithm adds edges to the MST in sorted order as long as they don't create a cycle
    def compute_mst_weight(self):
        weight      = 0
        for distance, i, j in sorted((self.distances[(i, j)], i, j) for (i, j) in self.distances):
            if self.unify(i, j):
                weight += distance
        return weight

    # helper checks the root of a node, in the process flatten the path to the root
    def resolve(self, key):
        path = []
        root = key
        while self.elements[root] is not None:
            path.append(root)
            root = self.elements[root]
        for key in path:
            self.elements[key] = root
        return root

    # helper checks if the two elements have the same root they are part of the same tree
    # otherwise set the root of one to the other, connecting the trees
    def unify(self, a, b):
        ra = self.resolve(a)
        rb = self.resolve(b)
        if ra == rb:
            return False
        else:
            self.elements[rb] = ra
            return True

    # helper that gets all pairs i,j for a list of keys
    def cross(self, keys):
        return (x for y in (((i, j) for j in keys if i < j) for i in keys) for x in y)

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    result = []
    start = maze.start #get starting point
    set_start = start

    if start == (maze.waypoints)[0] :
    # check if start is the goal
        result.append(start)
        return result
    
    queue = []
    parent = {}
    visited = {}

    queue.append(start)
    visited[start] = True
    parent[start] = None

    #traverse
    while queue :
        start = queue.pop(0)
        if start == (maze.waypoints)[0] :#start is the goal
            result = [start]

            while result[-1] != set_start :#maze.start
                #parent of the most recent element
                parent_path = parent[result[-1]]
                result.append(parent_path)
            
            result.reverse()

            return result

        neighbors = maze.neighbors(start[0], start[1])

        for neighbor in neighbors :
            if (neighbor not in queue) and (neighbor not in visited) :
                queue.append(neighbor)
                visited[neighbor] = True
                parent[neighbor] = start
    

def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    start = maze.start
    goal = maze.waypoints[0]

    # lists
    parents = {}
    priority = [start]
    visited = []

    # f(n) and g(n)
    start_to_curr = {}
    total_cost = {}
    
    # initialize
    start_to_curr[start] = 0
    total_cost[start] = calc_manhattan_distance(start, goal)

    # go through priority queue until it's empty
    while priority :
        curr = None
        tmp = 0

        for node in priority :
            if total_cost[node] < tmp or curr is None  :
                tmp = total_cost[node]
                curr = node
        
        if curr == goal :
            path = [curr]

            while curr in parents :
                curr = parents[curr]
                path.append(curr)

            path.reverse()
            return path

        visited.append(curr)
        priority.remove(curr)

        neighbors = maze.neighbors(curr[0], curr[1])
        for neighbor in neighbors :
            if maze.navigable(neighbor[0], neighbor[1]) :

                # don't look if already visited
                if neighbor in visited :
                    continue

                possible = start_to_curr[curr] + calc_manhattan_distance(curr, neighbor)

                if neighbor not in priority :
                    priority.append(neighbor)
                
                parents[neighbor] = curr
                start_to_curr[neighbor] = possible
                heuristic = calc_manhattan_distance(neighbor, goal)

                total_cost[neighbor] = start_to_curr[neighbor] + heuristic

    



def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    return []
    

def fast(maze):
    """
    Runs suboptimal search algorithm for extra credit/part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    return []

# Helpers

# def heuristic(position, goals) :
#     heur = 99999999999

#     for goal in goals :
#         manhattan = calc_manhattan_distance(position, goal)

#         #might be heur = (heur, manhattan)[manhattan < heur]
#         heur = (manhattan, heur)[manhattan < heur]
    
#     return heur

def calc_manhattan_distance(start, end) :
    x = abs(start[0] - end[0])
    y = abs(start[1] - end[1])

    return x + y
