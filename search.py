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
    # result = []
    # start = maze.start

    # g_score={cell:float('inf') for cell in maze.indices}
    # g_score[start]=0
    # f_score={cell:float('inf') for cell in maze.indices}
    # f_score[start]=heuristic(start,(1,1))

    # priority = que.PriorityQueue()
    # priority.put((heuristic(start, (1,1)), heuristic(start, (1,1)), start))

    # path = {}

    # while not priority.empty() :
    #     curr = priority.get()[2]

    #     if curr == (1,1) :
    #         break

    #     for 
    start = maze.start

    priority = que.PriorityQueue()
    visited = {}
    parents = {}
    visited[start] = True
    goals = maze.waypoints

    start_pos = (heuristic(start, goals), start)
    priority.put(start_pos)

    while priority :
        set_start = priority.get()
        
        if set_start[1] in goals:
            goal = set_start[1]
            return find_path(start, goal, parents)
        
        curr = set_start[1]
        neighbors = maze.neighbors(curr[0], curr[1])

        for neighbor in neighbors:
            parents[neighbor] = curr
            heur = heuristic(neighbor, goals)
            start_to_node = len(find_path(start, curr, parents))

            if (neighbor not in visited) and (maze.navigable(neighbor[0], neighbor[1])):
                # parents[neighbor] = curr
                # heur = heuristic(neighbor, goals)
                # start_to_node = len(find_path(start, curr, parents))

                # trying to figure outt he order of the priority queue
                new_node = ((heur + start_to_node), neighbor)
                priority.put(new_node)
                visited[neighbor] = True

    # return find_path(start, goals, parents)


    # result = []
    
    # start = maze.start #get starting point
    # #set_start = start
    # priority = que.PriorityQueue()
    # goals = maze.waypoints

    # parent = {}
    # visited = {}
    # visited[start] = True
    
    # start_position = (heuristic(start, goals), start)
    # priority.put(start_position)

    # #search
    # while priority :
    #     curr = priority.get()
    #     position = curr[1]
        
    #     #curr[1] = position
    #     if position in goals :
    #         result = [position]

    #         while result[-1] != start :
    #             adult = parent[result[-1]]
    #             result.append(adult)
    #         return result.reverse()
        
    #     neighbors = maze.neighbors(position[0], position[1])

    #     for neighbor in neighbors :
    #         if neighbor not in visited :
    #             parent[neighbor] = position
    #             path = find_path(start, position, parent)
    #             manhat = heuristic(neighbor, goals)
    #             length = len(path)

    #             # while path[-1] != start :
    #             #     adult = parent[path[-1]]
    #             #     path.append(adult)
                
    #             # path.reverse()
                
    #             node = ((manhat + length), neighbor)

    #             priority.put(node)
    #             visited[neighbor] = True



def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    return []
    # ways = []
    # priority = que.PriorityQueue()
    # start = maze.start
    # goals = maze.waypoints
    # lengths = {}

    # for x in range(len(goals)):
    #     for y in range(x+1, len(goals)):
    #         ways.append((x, y))

    
    # for direction in ways:
    #     copy_of_maze = deepcopy(maze)

    #     single_goal_x = goals[direction[0]]
    #     # somehow set a new start here
    #     copy_of_maze.start = single_goal_x

    #     single_goal_y = goals[direction[1]]
    #     # somehow set a new goals here
    #     copy_of_maze.waypoints = [single_goal_y]

    #     astar = astar_single(copy_of_maze)

    #     distance = len(astar)
    #     lengths[direction] = distance - 1

    # # parents
    # parent_node = (start, tuple(goals))
    # parents = {parent_node : None}

    # # the priority queue gets (f, distance to current node(g), (current node, remaing goals))
    # heur = spanning_tree(start, tuple(goals), lengths, goals)
    # curr = (heur, 0, parent_node)
    # priority.put(curr)

    # priority_distances = {curr[2]:0}

    # while priority:
    #     priority_curr = priority.get()
    #     priority_position = priority_curr[2][0]

    #     if (len(priority_curr[2][1]) == 0) :
    #         return find_path(priority_curr[2], parents)

    #     neighbors = maze.neighbors(priority_position[0], priority_position[1])
    #     for neighbor in neighbors :
    #         goals_left = goals_left(neighbor, priority_curr[2][1])
    #         goals_at_curr = tuple(goals_left)
    #         dist = (neighbor, goals_at_curr)

    #         if dist in priority_distances :
    #             additional_dist = priority_distances[priority_curr[2]] + 1
    #             if additional_dist >= priority_distances[dist]:
    #                 continue

    #         #update distance
    #         priority_distances[dist] = priority_distances[priority_curr[2]]+1

    #         #update parent
    #         parents[dist] = priority_curr[2]

    #         #update priority
    #         prev = priority_curr[0]
    #         heurist = spanning_tree(neighbor, goals_at_curr, lengths, goals)
    #         next = priority_distances[dist] + heurist
    #         next = max(prev, next)

    #         new_node = (next, priority_distances[dist], dist)
    #         priority.put(new_node)

def fast(maze):
    """
    Runs suboptimal search algorithm for extra credit/part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    return []

# Helpers
def find_path(start, end, parent) :
    # result = []
    # result.append(end)
    # after = end

    # while after != start : 
    #     result.append(parent[after])
    #     after = result[after]

    # return result[::-1]
    path = [end]
    # path.append(end)

    while start != path[-1] :
        adult = parent[path[-1]]
        path.append(adult)

    path.reverse

    return path

def heuristic(position, goals) :
    heur = 99999999999

    for goal in goals :
        manhattan = calc_manhattan_distance(position, goal)

        #might be heur = (heur, manhattan)[manhattan < heur]
        heur = (manhattan, heur)[manhattan < heur]
    
    return heur

def calc_manhattan_distance(start, end) :
    x = abs(start[0] - end[0])
    y = abs(start[1] - end[1])

    return x + y

# change these
# def spanning_tree(node, goals, map_, goal):
#     if len(goals) == 0:
#         return 0

    
#     curr = [goal.index(goals[0])]
#     points = []
#     array = []
#     result = 0

#     for i in range(1, len(goals)):
#         g = goal.index(goals[i])
#         points.append(g)

#     while len(goals) != len(curr):
#         mins = []

#         for c in curr:
#             minimum = sys.maxsize
#             minimum_1 = None

#             for end in points:
#                 if c > end:
#                     edge = (end, c)
#                 else:
#                     edge = (c, end)

#                 if minimum > map_[edge]:
#                     minimum = map_[edge]
#                     minimum_1 = end

#             mins.append((minimum, minimum_1))

#         smallest_point = min(mins)
#         points.remove(smallest_point[1])

#         result += smallest_point[0]
#         curr.append(smallest_point[1])

#     for indiv_goal in goals:
#         array.append(heuristic(node, indiv_goal))
    
#     res = result + min(array)
#     return res

# def goals_left(node, goals):
#     result = []

#     for goal in goals:
#         if node != goal:
#             result.append(goal)
            
#     return result