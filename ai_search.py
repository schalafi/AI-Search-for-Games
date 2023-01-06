import sys
import time
import queue as Q
from typing import Hashable 
from puzzle import Puzzle
import heapq

def get_platform():
    platforms = {
        'linux1' : 'Linux',
        'linux2' : 'Linux',
        'darwin' : 'OS X',
        'win32' : 'Windows'
    }
    if sys.platform not in platforms:
        return sys.platform
    
    return platforms[sys.platform]

IS_LINUX= get_platform().startswith('linux')

if IS_LINUX:
    import resource

#Add and retrieve 
class Frontier:
    def __init__(self):
        self.items = {}
        self.order = []
   
    def __contains__(self, id):
        return id in self.items
        
    def __getitem__(self, id):
        return self.items[id]
    
    def __setitem__(self, id, element):
        self.items[id] = element
        self.order.append(id)
        
    def pop(self, index :int = None):
        """
        index:
            index of the element to retrieve
            By default it returns the last element .pop()
            making it a Stack
            Pass index =0 to implement a Queue or could use another index
            it retrieves the first element in the list (the oldest)

        Return the SearchNode
        """
        if index is None:
            id = self.order.pop()
        else:
            id = self.order.pop(index)
        #Get and delete element from frontier  
        return self.items.pop(id)

    def __len__(self):
        return len(self.order)

# Return objects from min to max priority
class PriorityFrontier(Frontier):    
        
    def update(self, id, element ):
        self.items[id] = element
        priority = element.get_f()
        for i, (p, item_id) in enumerate(self.order):
            if item_id == id:
                self.order[i] = (priority, id)
                heapq.heapify(self.order)
    
    def __setitem__(self, id, element):
        self.items[id] = element
        #priority is computed with get_f method (distance + heuristic)
        heapq.heappush(self.order, (element.get_f(), id))
        
    def pop(self):
        priority, id = heapq.heappop(self.order)
        return self.items.pop(id)
    
def get_info(node,tree,n_nodes_expanded,height):
    """
    Print several pieces of information
    the path to the goal
    the cost of the path
    the number of nodes expanded in the search
    the max search depth
    """
    path = node.moves_to_solution()
    print("Path to goal: {}".format(path))
    print("Cost of path: {}".format(node.g))
    print("Number of nodes expanded:",n_nodes_expanded)
    print("Search depth:",node.depth)
    print("Max search depth:", height)
    return path 

class SearchNode:
    def __init__(self,state:object):
        """
        state:
            an instance of a game board
        """

        self.state = state
        #reference to a SearchNode, which generated this node
        self.parent = None 
        #the action taken by the parent 
        #depends on the game
        #self.parents_action = None
        self.children = []
        
        #Accumulated cost to reach this node from
        #initial state
        self.g = 0 
        #Indicates if the node is in
        #the solution path
        self.solution = False
        #Indicates if node is root
        self._is_root = False
        #Depth of the node in search tree
        self.depth = 0 

    def set_root(self):
        """
        Change the state 
        of the node to root.
        """
        self._is_root  = True

    def is_root(self)-> bool:
        """
        Check if node is root node.
        """
        return self._is_root
    
    def moves_to_solution(self)->list:
        """
        Return a list
            list with the moves to the solution
            or path.
        """
        if self.parent is None:
            return []
        
        p = self.parent
        path = [self.get_action()]

        while (p is not None):
            action = p.get_action()
            if action  is not None:
                path = [action] + path
            p = p.parent
        
        return path

    def get_f(self)->float:
        """
        Return the f-score
            the cost of the solution 
            (cost of moves from root to this node)
            + the heuristic from this node to goal.
        """
        return self.g + self.state.h

    def successors(self):
        """
        Return a list of node (SerachNode) 
        successors
        """
        #A list of succesor states for the game.
        neighbors = self.state.get_neighbors()
        #List of search node successors.
        succs = []

        for n in neighbors:
            #create a new SearcNode
            sn = SearchNode(n)
            #add the cost to reach this new state
            #TODO adapt this cost depending of game
            #in this case we have cost of one movement if 1 
            sn.g = self.g + 1
            sn.parent = self 
            sn.depth = self.depth + 1
            #TODO use a standard method to get
            #this action
            #the position of the 0 for self
            #the positon of the new 0 in successor
            #sn.parents_action = [self.state.zero,sn.state.zero]
            #add succesor node
            succs.extend([sn])
        self.children = succs
    
    def get_action(self):
        """
        Return the action of the parent fo this state
        parent -> action -> state
        """

        return self.state.parents_action

    def __eq__(self,other):
        """
        ==
        Two nodes are equal if 
        the string state 
        are equal.
        """

        if isinstance(other, self.__class__):
            return self.state.string_state == other.state.string_state
        else:
            print(f"Comparing: {self.__class__} and {other.__class__} ")
            return False
    
    def __ne__(self,other):
        """
        !=
        """
        result = self.__eq__(other)

        if result is NotImplemented:
            return result 
        return not result

    def __lt__(self,other):
        """
        <
        A SearchNode a is less than another b
        if the f-score  of a is less than the f-score
        of b
        """
        return self.get_f() < other.get_f()


class Search:

    def __init__(self,initial:object):
        """
        search_method: one of 'dfs' 'bfs' ast'
        initial:
            initial state of the game
        """
        self.initial = initial
        self.search_method  = None
        self.game_class = initial.__class__
        self.act_node = None 

        self.root = SearchNode(initial)
        self.root.set_root()

        self.frontier = None
        self.expanded = None
        self.max_search_depth = 0 
        self.n_nodes_expanded = 0
        self.height = 0 
        self.GOAL_NODE = None 

        self.t0  = None 
        self.stats = {
            "nodes_expanded": self.n_nodes_expanded,
            "max_search_depth":  self.height,
            'time':0,
            'n_actions':0 }
       
    def compute_height(self)-> int:
        """
        Compute tree's heigth 
        #TODO use list instead of Queue
        """
        q = Q.Queue()
        q.put(self.root)
        sizel=[1]
        nodes_level =0
        i=0
        while(q.qsize() !=0):
            n= q.get()
            sizel[i] += -1
            for child in n.children:
                q.put(child)
            nodes_level+= len(child.children)

            if sizel[i] ==0:
                sizel.append(0)
                sizel[i+1] += nodes_level
                i+=1
                nodes_level =0

        return i+1

    def solution(self,solution_node:"SearchNode"):
        """
        solution_node:
            Node where the game state is 
            the game's goal.
        Mark the nodes in the solution
            with node.solution = True
        """

        state = solution_node.state
        while(state != None):
            #mark the nodes in the solution
            state.solution = True 
            state = state.parent
    def compute_stats(self, node:SearchNode,n_actions:  int):
        """
        node:
            Goal node
        """
        self.height = self.compute_height()
        self.stats = {
            "nodes_expanded": self.n_nodes_expanded,
            "search_depth":node.depth,
            "max_search_depth":  self.height,
            'time':time.time() - self.t0,
            'n_actions':n_actions ,
            'ram_usage':None}
       
    def dfs(self,
        goal_test: callable = Puzzle.is_goal, 
        neighbors:callable =SearchNode.successors):
        """
        goal_test: class method
            The goal test for the game
        neighbors: class method
            The method in game to produce 
            neighbors or successors 
        """
        print("Running DFS ... \n")
        height = 0 
        #INITIALIZE FRONTIER
        #LIFO
        #Stack
        frontier = Frontier()#id (string): SearchNode

        root_id = self.root.state.string_state
        frontier[root_id] = self.root
        self.frontier = frontier
        
        #Expanded nodes
        #use dict for fast retrival 
        self.expanded = {}

        while not len(self.frontier) == 0:
            #Stack 
            #LIFO
            node= self.frontier.pop()
            
            #add node to expanded nodes
            node_id  = node.state.string_state
            #TODO define a method for string state
            #SearchNode.get_state_id
            self.expanded[node_id]=True
            
            if goal_test(node.state):
                #If node is goal node return the path
                path = get_info(node,self, self.n_nodes_expanded,height)
                self.GOAL_NODE= node
                return path
            
            #Compute chindren of the current node
            neighbors(node)
           
            for n in  node.children:
                #Node expanded and not in frontier
                #Check if node has been expanded (expanded)
                #Avoid loops
                state_id = n.state.string_state
                if (state_id not in self.expanded
                and state_id not in  self.frontier):
                    self.frontier[state_id] = n 
                
                height = max(height,n.depth)
            self.n_nodes_expanded+= 1
        
        return []

    def bfs(self,
        goal_test: callable = Puzzle.is_goal, 
        neighbors:callable =SearchNode.successors):
        """
        goal_test: class method
            The goal test for the game
        neighbors: class method
            The method in game to produce 
            neighbors or successors 
        """
        print("Runnig BFS ... \n")

        height = 0 
        #FIFO
        #Queue 
        frontier = Frontier() #id (string): SearchNode

        #TODO define method to get the string_state (id)
        node_id = self.root.state.string_state
        frontier[node_id] = self.root  #id (string): SearchNode

        #Node whos children have been explored 
        self.expanded = {}

        #while frontier not empty
        while len(frontier)!= 0:
            #Queue  FIFO
            #Get the first in the list
            node = frontier.pop(0)
            node_id = node.state.string_state
            #add nodo to expanded/explored
            self.expanded[node_id] =True 

            #check if we have reached the goal
            if goal_test(node.state):
                #If node is goal 
                #return the path to the start node
                path = get_info(node,self, self.n_nodes_expanded,height)
                self.GOAL_NODE = node
                return path

            #Generate and  add neighbors
            #self.neighbors = generated neighbors
            neighbors(node)

            for n in node.children:
                n_id = n.state.string_state
                if (self.expanded.get(n_id, None) == None 
                    and n_id not in frontier):
                    #add  child to frontier
                    frontier[n_id] = n 

            self.n_nodes_expanded+= 1
        return []

    def a_star(self,
        goal_test: callable = Puzzle.is_goal, 
        neighbors:callable =SearchNode.successors):
        """
        goal_test: class method
            The goal test for the game
        neighbors: class method
            The method in game to produce 
            neighbors or successors 
        """
        print("Runnig A* ... \n")

        height = 0 
        #FIFO
        #Queue 
        frontier = PriorityFrontier() #id (string): SearchNode

        #TODO define method to get the string_state (id)
        node_id = self.root.state.string_state
        frontier[node_id] = self.root  #id (string): SearchNode

        #expanded/explored nodes
        expanded = {}

        #Frontier not empty
        while len(frontier) != 0:
            self.n_nodes_expanded+= 1
            node = frontier.pop()
            node_id = node.state.string_state

            #add to expanded nodes
            expanded[node_id] = True

            #Check for goal
            if goal_test(node.state):
                #If is goal 
                #return the path from start  to goal. 
                path = get_info(node,self, self.n_nodes_expanded,height)
                #set goal node
                self.GOAL_NODE = node
                return path

            #Compute children (next states) from node
            #node.children = next states
            neighbors(node)

            for n in node.children:
                #compute height
                height = max(height,n.depth)
                n_id = n.state.string_state

                #check if we have found a new best route 
                #lower cost
                if n_id in frontier:
                    current_distance  = n.g
                    #TODO generalize to get the cost from the graph or game
                    #compute new distance 
                    #with the cost of 1 step equal 1
                    #node -->n costs 1 
                    new_distance = node.g + 1
                    
                    if new_distance < current_distance:
                        n.g = new_distance
                        #set new parent
                        n.parent  = node 
                        #update the value of n in frontier
                        frontier.update(n_id,n)
                #n is not  in frontier
                # add it
                else:
                    n.parent = node 
                    #update the node in frontier
                    frontier[n_id]= n
        return []

    def search(self,method:str,**kwargs):
        """
        Call the search method

        """
        result = None 
        self.t0 = time.time()

        methods = {'dfs':self.dfs,'bfs':self.bfs,'a_star':self.a_star}

        if method not in methods:
            raise Exception(f"Method: {method} not defined")
        method = methods[method]
        result = method(
                goal_test=self.game_class.is_goal,
                neighbors = SearchNode.successors )

        if self.GOAL_NODE is None:
            print("No solution found")                
        else:
            n_actions = len(result)
            self.compute_stats(self.GOAL_NODE,n_actions)
            print("STATS:")
            print(self.stats)

        return result

            
def run_test_case(matrix, method):
    """
        matrix matrix of the Puzzle
        zero (i,j) position of 0 in mattrix
        method is search method
    """
    t0 = time.time()
    p = Puzzle(matrix)
    S = Search(initial = p)
    path = S.search(method = method)
    if path != None:
        print("Len solution path ",len(path))
    else:
        print("Search Failed")
    
    print("Game final state: ")
    print(S.GOAL_NODE.state)
    
    content =''
    running_time = time.time() -t0
    content+="running_time: {}".format(running_time) +"\n"
    print("Stats: ")
    print(content)
    if IS_LINUX:
        
        ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        content+="max_ram_usage: {}".format(ram) +"\n"

        with open("output.txt","w") as f:
            f.write(content)

    return path 


class ObjectPriority:

    def __init__(self,  name, priority):
        self.name = name
        self.priority = priority
    def get_f(self):

        return self.priority
    def __str__(self) -> str:
        return self.name + ',' +str(self.priority)

def test_priority_frontier():
    f  = PriorityFrontier()
    #id, priority, element
    f['aa']= ObjectPriority('first element',100)
    f['bb']= ObjectPriority( 'second element',-1)
    f['cc']= ObjectPriority( 'third element',-0.1)

    print("priority frontier")
    print(f)
    while len(f) != 0:
        print(f.pop())

if __name__ == "__main__":
    m=[[1,2,5],
    [3,4,0],
    [6,7,8]]
    zero=(1,2)
    run_test_case(matrix = m,  method = 'dfs')

    test_priority_frontier()
            

