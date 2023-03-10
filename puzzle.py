import random 

STRING_TO_ACTION ={'U':(-1,0),'D':(1,0),'L':(0,-1),'R':(0,1)}
ACTION_TO_STRING = {v:k for k,v  in STRING_TO_ACTION.items() }

#Where the tile will move
#Moves are determined by a vector (x,y)
#with matrix-like movement
#(1,0) means row 1 and colum 0 which is DOWN (in code we move the opposite tile in this direction)
#we invert the names, the empty tile is the tile 
#which will move to the direction given by (x,y)
NAME_TO_ACTION ={'Down':(-1,0),'Up':(1,0),'Right':(0,-1),'Left':(0,1)}
ACTION_TO_NAME = {v:k for k,v  in NAME_TO_ACTION.items() }

Index = tuple[int,int]
class Puzzle:
    """
        Class representing the information of a state of a N-Puzzle
    """
    def __init__(self,matrix, generate_goal_matrix = False):
        self.matrix = matrix
        self.N = len(matrix)

        #Set up goal state, as a matrix and string
        #Don't compute at initialization by default
        if generate_goal_matrix:
            self.goal_matrix = self.generate_goal()
            self.string_goal = self.matrix_to_string(self.goal_matrix)
        else:
            self.goal_matrix = None #self.generate_goal()
            self.string_goal =None # self.matrix_to_string(self.goal_matrix)
        
        #print("Goal Matrix: ", self.goal_matrix, sep  ="\n")
        #print("Goal String: ", self.string_goal)


        #Tuple (i,j). Position of empty space represented by 0
        self.zero = self.get_index_of_value(matrix)

        #TODO use str representation
        self.string_state =self.get_state()
        self.parents_action =None
        self.g = 0
        #Don't compute at initialization
        #self.h =self.heuristic()
        self.parent = None

        i,j = self.zero[0],self.zero[1]
        value = self.matrix[i][j]
        assert  value == 0, "Matrix must have an 0 in position ({},{}) \n Found: {}".format(i,j,value)
    def get_h(self):
        return self.heuristic()

    def generate_goal(self):
        """
        Produce the goal matrix of the game
        depending of the size of the board.
        Return the goal matrix
        Example end position in 4x4 puzzle
        [[0, 1, 2, 3]
         [4 ,5, 6, 7], 
         [8, 9, 10, 11], 
         [12, 13, 14, 15]
        ]
        """
        k= 0
        goal_matrix = [] 
        for i in range(self.N):
            row = []
            for j in range(self.N):
                row.append(k)
                k+= 1
            goal_matrix.append(row)
            
        return goal_matrix
    def matrix_to_string(self, matrix):
        
        return ''.join( ''.join( str(y) for y in x) for x in matrix)

    def heuristic(self):
        """
        Manhatan distance heuristic.
        the idea is to compute the index of the value in matrix
        using the next procedure
        k,l = value//N, value%N
        example N= 10
            value = 11, k,l = 1,1
            value 12, k,l = 1,2
        example N =  3
            value = 4 k,l = 1,1
            value = 7, k,l = 2,1
        """

        cost = 0 
        #Tile 0 is not considere for the heuristic
        for i in range(self.N):
            for j in range(self.N):
                value =self.matrix[i][j]
                if value != 0:
                    (k,l) = value//self.N, value%self.N  
                    #add distance between correct (k,l)
                    #and current postion (i,j)
                    cost += abs(i-k) +abs(j-l)

        return cost

    
    def get_number_inversions(self) ->  int:
        """
        Number of p>q where p  = matrix[i]
        q = matrix[j] and i<j example [5,2,1] -> 3 inversions
        (5,2) (5,1) (2,1)

        Return number of invertions
        """

        count= 0

        puzzle_list = [number for row in self.matrix for number in row if number != 0]

        for i in range(len(puzzle_list)):
            for j in range(i + 1, len(puzzle_list)):
                if puzzle_list[i] > puzzle_list[j]:
                    count+= 1

        return count

    def get_blank_space_row_counting_from_bottom(self)->int: 
        """
        Return row position of empty/zero tile
            Row position counting from bottom to top
            instead of using matrix indexing

        """

        zero_row,zero_column = self.get_index_of_value(0)

        return self.N - zero_row


    def is_solvable(self)->bool:
        """
        1. If N is odd, then puzzle instance is solvable if number of inversions is even in the input state.
        2. If N is even, puzzle instance is solvable if
            - the blank is on an even row counting from the bottom (second-last, fourth-last, etc.)
              and number of inversions is odd.
            - the blank is on an odd row counting from the bottom (last, third-last, fifth-last, etc.)
            and number of inversions is even.
        3. For all other cases, the puzzle instance is not solvable.

        Return  if the puzzle is solvable.
        """

        number_inversions = self.get_number_inversions()
        blank_position  = self.get_blank_space_row_counting_from_bottom()

        if (self.N % 2) == 1 and (number_inversions % 2)== 0:
            return True 
        if (self.N % 2) == 0 and  (blank_position % 2)== 0 and (number_inversions % 2) == 1:
            return True

        if (self.N % 2)==0 and (blank_position % 2 )== 1 and (number_inversions % 2)== 0:
            return True 

        return False  


    def is_goal(self):
        """
        Check if this board is the solution for the N Puzzle
        Goal for the board  [[0,1,2]
                           [3,4,5]
                           [6,7,8]]
        is 012345678
        """
        return  self.string_state == self.string_goal

    def get_state(self):
        """for i in range(len(self.matrix)):
            for j in range(len(self.matrix)):
                state += str(self.matrix[i][j])
        """
        state = self.matrix_to_string(self.matrix)
        return state

    def clone_matrix(self):
        p=[]
        for i in range(self.N):
            row = self.matrix[i][:]
            p.append(row)
        return p
    def action_to_string(self,action):
        """
        action: tuple
            action
        Return the string 
        representation of each action

        """
        return ACTION_TO_STRING.get(action,None)
    def step(self,action):
        """
        action: tuple
            an action
            'Up':(-1,0),'Down':(1,0),'Left':(0,-1),'Right':(0,1)
        Apply and action to the game
        """
        zero  = list(self.zero)

        position =  (zero[0]+action[0],zero[1]+action[1])
        #Check if is a valid action
        if (position[0] >= 0 and position[0] <=self.N 
            and position[1]>=0  and position[1] <= self.N ) :
                #Tile to move
                value = self.matrix[position[0]][position[1]] 

                #Create new matrix
                new_matrix =self.clone_matrix()

                #swap positions bewtween zero (empty tile) and adjacent tile
                new_matrix[position[0]][position[1]] = 0 
                new_matrix[zero[0]][zero[1]]= value

                #update zero location
                self.zero = position

                self.matrix = new_matrix
                self.parents_action = action

                #Update zero location
                self.zero = position
                
        else:
            print("zero: ",self.zero)
            raise Exception("Invalid action: " + str(action))     

        
    def get_neighbors(self, is_dfs=False):
        zero  = list(self.zero)
        n = len(self.matrix)-1
        #In this assignment, where an arbitrary choice
        #must be made, we always visit child nodes in the
        #"UDLR" order; that is, [upper, down, left, right]
        #in that exact order. Specifically
        actions =[(-1,0),(1,0),(0,-1),(0,1)]
        #actions_dfs = [(0,1),(0,-1),(1,0),(-1,0)]        

        #if is_dfs ==1:
        #    actions= actions_dfs
    
        #new puzzles 
        puzzles =[]
        for  action in actions:
            #the new zero location
            #and also is the loc of the tile to be moved
            position =  (zero[0]+action[0],zero[1]+action[1])
            #Check if is a valid action
            if (position[0] >= 0 and position[0] <=n 
                and  position[1]>=0  and position[1] <=n) :
                #Tile to move
                #print("POSITION: ", position)
                value = self.matrix[position[0]][position[1]] 
                #Create new matrix
                new_matrix =self.clone_matrix()
                #swap positions bewtween zero (empty tile) and adjacent tile
                new_matrix[position[0]][position[1]] = 0 
                new_matrix[zero[0]][zero[1]]= value
                new_puzzle = Puzzle(new_matrix, generate_goal_matrix = False)
                #copy from parent goal matrix 
                new_puzzle.goal_matrix = self.goal_matrix
                new_puzzle.string_goal= self.string_goal

                new_puzzle.parents_action = action
                new_puzzle.parent= self
                new_puzzle.zero = position
                puzzles.append(new_puzzle)
        return puzzles
    def __str__(self):
        res = ''
        for row in range(self.N):
            res += ' '.join(map(str, self.matrix[row]))
            res += '\r\n'
        res = res[:-2]
        return res

    def get_index_of_value(self,value: float)->Index:
        """
        value:
            value to search for in game's matrix
        Get the index (i,j)
        for the value in matrix game
        """
        zi,zj  = 0,0
        found_zero = False 
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix[i])):
                if self.matrix[i][j] == 0:
                    zi,zj = i,j 
                    found_zero = True 
        if not found_zero:
            raise Exception(f"Not zero found in matrix: {self.matrix}")

        return zi,zj 

def test_neighbors():
    matrix = [[1,2,5],
            [3,4,0],
            [6,7,8]]

    print("initial state: ", *matrix,sep='\n')
    
    puzzle_ = Puzzle(matrix )

    children = puzzle_.get_neighbors()
    for x in children:
        print(str(x))
        print()

def test_step():
    matrix = [[1,2,5],
            [3,4,0],
            [6,7,8]]

    print("initial state: ", *matrix,sep='\n')
        

    puzzle_ = Puzzle(matrix )

    actions = [ 'U','L','L']
    
    print("initial state: ")
    print(puzzle_)
    for a in actions:
        action = STRING_TO_ACTION[a]
        print("action: ", a)
        puzzle_.step(action)
        print(puzzle_)
        print("is goal? ", puzzle_.is_goal())
def test_is_goal():
    matrix = [[0,1,2],
            [3,4,5],
            [6,7,8]]
    p = Puzzle(matrix)
    print("initial state: ", *matrix,sep='\n')
    print("is goal? ", p.is_goal())
    print()

    matrix = [[3,1,2],
            [0,4,5],
            [6,7,8]]
    p = Puzzle(matrix)
    print("initial state: ", *matrix,sep='\n')
    print("is goal? ", p.is_goal())
    print()


def test_steps():
    print("TESTING . . . ",test_steps.__name__)
    matrix = [
            [1,0,2],
            [3,4,5],
            [6,7,8]]
    
    puzzle_ = Puzzle(matrix)
    #{'U':(-1,0),'D':(1,0),'L':(0,-1),'R':(0,1)}
    for action in [(0,1),(1,0),(1,0),(0,-1),(0,-1)]:
        puzzle_.step(action)
        print(puzzle_)
        print("zero: ",puzzle_.zero)
        
def test_is_solvable():
    print("TESTING . . . ",test_is_solvable.__name__)
    matrix = [
            [1,0,2],
            [3,4,5],
            [6,7,8]]
    puzzle_ = Puzzle(matrix)
    print("Is solvable:" , puzzle_.is_solvable())
    print()

    matrix =[[3,1,2],
            [0,4,5],
            [6,7,8]]
    puzzle_ = Puzzle(matrix)
    print("Is solvable:" , puzzle_.is_solvable())
    print()

    matrix =[[1,2,5],
            [3,4,0],
            [6,7,8]]
    puzzle_ = Puzzle(matrix)
    print("Is solvable:" , puzzle_.is_solvable())
    print()

    matrix =[[6,1,8],
    [4,0,2],
    [7,3,5]]
    puzzle_ = Puzzle(matrix)
    print("Is solvable:" , puzzle_.is_solvable())
    print()

    matrix =[[8,6,4],
    [2,1,3],
    [5,7,0]]
    puzzle_ = Puzzle(matrix)
    print("Is solvable:" , puzzle_.is_solvable())
    print()

    matrix =[[8,1,2],
    [3,4,5],
    [6,7,0]]
    puzzle_ = Puzzle(matrix)
    print("Is solvable:" , puzzle_.is_solvable())
    print()

    matrix =[[4,1,2],
    [3,8,5],
    [6,7,0]]
    puzzle_ = Puzzle(matrix)
    print("Is solvable:" , puzzle_.is_solvable())
    print()

    matrix =[[2,1,4],
            [6,8,5],
            [3,7,0]]
    puzzle_ = Puzzle(matrix)
    print("Is solvable:" , puzzle_.is_solvable())
    print()

    matrix =[[6,1,4],
            [2,5,8],
            [3,7,0]]
    puzzle_ = Puzzle(matrix)
    print("Is solvable:" , puzzle_.is_solvable())
    print()

    #FALSE
    matrix =[[6,7,4],
            [2,5,8],
            [3,1,0]]
    puzzle_ = Puzzle(matrix)
    print("Is solvable:" , puzzle_.is_solvable())
    print()

    #FALSE
    matrix =[[6,7,4],
            [0,5,8],
            [3,1,2]]
    puzzle_ = Puzzle(matrix)
    print("Is solvable:" , puzzle_.is_solvable())
    print()

    matrix =[[6,4,7],
            [5,0,8],
            [3,1,2]]
    puzzle_ = Puzzle(matrix)
    print("Is solvable:" , puzzle_.is_solvable())
    print()

def generate_random_matrix(n:int = 3):
    numbers = [x for x in range(n*n)]
    random.shuffle(numbers)
    matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            index = n*i+j
            row.append(numbers[index])
        matrix.append(row)
            
    return matrix 

def test_generate_random_matrix():
    for i in range(11):
        m = generate_random_matrix(n= i)
        print("n = ", i )
        print("matrix:",*m,sep ="\n" )
        print()


def get_solvable_10():
    matrices = []

    for i in range(20):
        m = generate_random_matrix(n= 10)
        #print("MATRIX", m)
        puzzle = Puzzle(m)
        solvable = puzzle.is_solvable()
        if solvable:
            matrices.append(m)
    return matrices 

def tuple_to_matrix(matrix_tuple, n = None):
    """
    transform tuple to matrix
    ex:
    input: (0,1,2,3,4,5,6,7,8)
    output:
    [[0,1,2],
     [3,4,5],
     [6,7,8]]
    """
    if n is None:
        n = len(matrix_tuple)**(1/2)
    
    matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(matrix_tuple[n*i+j])
        matrix.append(row)
    return matrix
import math 
def get_easy(n, heuristic_thresold,number_matrices ):
    """
    n:
    heuristic_thresold:
    number_matrices: 

    Return a list of easy games nxn
    meaning games with low heuristic value
    <= 100

    """
    matrices = []
    perms = Permutations(n*n)
    perms_iter = iter(perms)
    n_permutations = math.factorial(n*n)
    counter = 0 
    while (len(matrices) <= number_matrices 
        and counter < n_permutations):
        
        t_matrix = next(perms_iter)
        m = tuple_to_matrix(t_matrix, n)
        puzzle = Puzzle(m)
        solvable = puzzle.is_solvable()
        h = puzzle.heuristic()
        counter+= 1
        print("Heuristic: ", h)
        if solvable and h <= heuristic_thresold:
            matrices.append(m)
    return matrices 

import itertools

class Permutations:
    def __init__(self, n):
        self.numbers = range(0, n)

    def __iter__(self):
        return itertools.permutations(self.numbers)

    def __reverse__(self):
        return itertools.permutations(reversed(self.numbers))


def test_permutations():
    p = Permutations(10)
    p_iter = iter(p)
    for i in range(10):
        print(next(p_iter))

def test_get_neighbors():
    matrix = [[2,3],
                  [1,0]] 
    p = Puzzle(matrix)

    print(p )
    print("state id: ", p.get_state())

    for n in p.get_neighbors():
        print("Neighbor")
        print(n)
        print("state id: ", n.get_state())
    matrix = [[0]
                 ] 
    p = Puzzle(matrix)

    print(p )
    print("state id: ", p.get_state())

    for n in p.get_neighbors():
        print("Neighbor")
        print(n)
        print("state id: ", n.get_state())
if __name__ == "__main__":
    test_neighbors()
    test_step()
    test_is_goal()
    test_steps()
    
    test_is_solvable()
    test_generate_random_matrix()

    solvable_10 = get_solvable_10()

    print("Solvable 10")

    print(*solvable_10, sep='\n')

    print("Easy Games 5x5:")
    m_easy = get_easy(n = 5, heuristic_thresold=300, number_matrices= 20)
    print(*m_easy,sep='\n')

    #test_permutations()
    print("Test get neighbors:")
    test_get_neighbors()

    print("All 2x2:")
    m_easy = get_easy(n = 2, heuristic_thresold=300, number_matrices= 4*3*2 )
    print(*m_easy,sep='\n')

    