import os 
import sys
import time 
import numpy as np 
from manimlib import * 
sys.path.append('.')

from puzzle import Puzzle 
from puzzle import NAME_TO_ACTION, ACTION_TO_NAME
from ai_search import Search

print(os.getcwd())



class Tile(Group):
    CONFIG = {
        "square_resolution": (3, 3),
        "height": 1,
        "depth": 0.5,
        "color":BLUE_D, 
        "include_label": True,
        #"numeric_labels": True,
        'prism_config':{
            "color": BLUE,
            "opacity": 1,
            "gloss": 0.5,
            "square_resolution": (4, 4),
        }
    }

    def __init__(self,number_label, **kwargs):
        """
        number_label: int or float
            label in the tile 

        """
        super().__init__(**kwargs)
        self.number_label = number_label

        
        
        self.prism =Prism(width= 1.0, height= 1.0, depth= 1.0,**self.prism_config) # Prism(dimensions= [1, 1, 1]) 
        self.add(self.prism)
        self.rotate(90 * DEGREES, OUT)
        self.set_color(self.color)
        
        config ={ 
        "fill_color": WHITE,
        "fill_opacity": 1,
        "stroke_width": 2,
        "stroke_color": BLACK,
        "stroke_opacity": 1.0}

        if self.include_label:
            label_text = Tex(
                str(self.number_label),                
                font= 'Luckiest Guy') 
            label_text.set_style(**config)
            label_text.shift(1.02 * OUT)
            label_text.set_height(self.prism.get_height() *0.5)#0.8

            label_text.apply_depth_test()
            #label_text.set_stroke(width=0)
            self.label_text = label_text
            self.add(self.label_text)
            

        self.set_height(self.height)
        self.set_depth(self.depth, stretch=True)

    
    def flip(self, axis=RIGHT):
        super().flip(axis)
        return self


class PuzzleBoard(SGroup):
    CONFIG = {
        #'tile_size': 2,
        'height':7,
        'depth':0.25,
        'gloss':0.3,
        'colors':[BLUE_C,WHITE ],
        'square_resolution':(3,3),#(3,3),
        'top_square_resolution':(5,5),
        'tile_config':
            {
                "square_resolution": (12, 12),
                "depth": 0.5,
                "color":BLUE_D, 
                "tails_color": RED,
                "include_label": True,
                "stroke_width": 2,
                "stroke_color": WHITE,
                'prism_config':{
                        "color": BLUE,
                        "opacity": 1,
                        "gloss": 0.5,
                        "square_resolution": (4, 4),
                    }
            }
    }

    def __init__(self, state,**kwargs):
        """
        state: Puzzle 
            initial state 
        """
        super().__init__(**kwargs)
        self.state = state 
        # NXN board 
        self.N = len(self.state.matrix)
        #List with relative position
        self.index = []
        self.shape = (self.N, self.N)
        nr, nc = self.shape
        cube = Cube(square_resolution=self.square_resolution)
        # Replace top square with something slightly higher res
        """top_square = Square3D(resolution=self.top_square_resolution)
        top_square.replace(cube[0])
        cube.replace_submobject(0, top_square)
        """

        # individiual squares
        self.squares = [cube.copy() for x in range(nc * nr)]
        
        self.add(*self.squares)

        self.arrange_in_grid(nr, nc, buff=0)
        self.set_height(self.height)
        self.set_depth(self.depth, stretch=True)
        for i, j in it.product(range(nr), range(nc)):
            color = self.colors[(i + j) % 2]
            self[i * nc + j].set_color(color)
            self[i * nc + j].set_opacity(0.2)

        self.center()
        self.set_gloss(self.gloss)
        self.set_opacity(0.1)

        self.add_tiles()

    def add_tiles(self):
        self.tiles = []
        matrix =np.array(self.state.matrix)
        k =0 
        for i in range(self.N):
            for j in range(self.N):

                if matrix[i][j] == 0 :
                    tile = Tile(number_label =" ",
                        **self.tile_config,
                        height= self.squares[0].get_height() )
                    tile.set_opacity(0)
                    tile.set_gloss(0)
                   
                else:
                    new_tile_config = self.tile_config.copy()
                    #gloss = random.random()
                    color,gloss = None,None 
                    if k%2 ==0:
                        color = BLUE_B
                        gloss = 0.2
                    else:
                        color = BLUE_D
                        gloss  = 0.6
                    new_tile_config['gloss'] = gloss
                    new_tile_config['color']  = color

                    tile = Tile(number_label = matrix[i][j],
                        **new_tile_config,
                        height= self.squares[0].get_height() )

                board_square = self.squares[self.get_index(i,j)]

                new_position = board_square.get_center()
                tile.move_to(new_position+ Z_AXIS*self.tile_config["depth"])
                self.add(tile)
                self.tiles.append(tile)
                #tile.make_smooth()
                k+= 1
    def get_index(self,i,j):
        """
        i: int
            row
        j: int
            column

        Return index of list given matrix coordinates
        """
        return self.N*i + j 

    def run_solution(self, solution,scene):
        """
        solution: list 
            [ [zero,action], [zero, action]]
        scene: Scene  
        """
        
        executing = VGroup(Text(r"Solving"), Tex(r'\dots')).arrange(RIGHT)
        executing.to_edge(UP + LEFT)
        executing.shift(LEFT*3)
        executing[1].next_to(executing[0], RIGHT)
        executing[1].shift(DOWN*0.1)

        scene.play(Write(executing[0]))
        scene.add(executing[1])

        dots_anim =turn_animation_into_updater(
            ShowCreation(executing[1],run_time = 2),
            cycle = True) 

        offset = 5
        action_scale = 2.5

        arrow_tip_config =dict(
            fill_opacity= 1.0,
            fill_color= RED,
            stroke_width=2.0,
            length = DEFAULT_ARROW_TIP_LENGTH*1.5,
            tip_style= 0,  # triangle=0, inner_smooth=1, dot=2
        )
        up_arrow =ArrowTip(
            angle = 90*DEGREES,
            **arrow_tip_config
        )

        down_arrow =ArrowTip(
            angle = 270*DEGREES,
            **arrow_tip_config)

        left_arrow =ArrowTip(
            angle = 0*DEGREES,
            **arrow_tip_config)
        
        right_arrow =ArrowTip(
            angle = 180*DEGREES,
            **arrow_tip_config)
    

        action_to_tex = {
            (1,0):up_arrow.scale(action_scale).shift(offset*RIGHT), #Tex(r'\uparrow',color = YELLOW).scale(action_scale).shift(offset*RIGHT),
            (-1,0):down_arrow.scale(action_scale).shift(offset*RIGHT), #Tex(r'\downarrow',color = YELLOW).scale(action_scale).shift(offset*RIGHT),
            (0,1):right_arrow.scale(action_scale).shift(offset*RIGHT), #Tex(r'\leftarrow',color = YELLOW).scale(action_scale).shift(offset*RIGHT),
            (0,-1):left_arrow.scale(action_scale).shift(offset*RIGHT) #Tex(r'\rightarrow',color = YELLOW).scale(action_scale).shift(offset*RIGHT),
        }

        sounds =["whoosh.wav",'whoosh2.wav','whoosh3.wav','whoosh5.wav']
        sounds_dir = "sounds"

        n_moves = 0 
        n_moves_text =VGroup(
            Text(r'Number of moves (actions):'),
            Tex(str(n_moves))
            ).arrange(DOWN)
        n_moves_text.next_to(executing,DOWN)
        n_moves_text.add_updater(
            lambda m,dt: m[1].become(Text(str(n_moves)).move_to(m[1]) ))
        self.n_moves_text = n_moves_text

        scene.play(Write(n_moves_text))
        
        #IDEAS: use zero location and swap 
        #the tile indicated by the action
        for i in range(0,len(solution)):
            action = solution[i]
            #Get zero location from initial state
            zero_loc =self.state.get_index_of_value(0) 

            #update the zero location
            #print("Zero location:", zero_loc)
            #get zero tile in VGroup 
            ind_zero =self.get_index(*zero_loc)
            zero_tile = self.tiles[ind_zero]

            zero_center = zero_tile.get_center()
            zero_center_no_z = np.copy(zero_center)
            zero_center_no_z[2]  = 0

            #get non-zero tile 
            tile_loc =tuple(np.array(zero_loc) + np.array(action) )
            #print("tile location:",tile_loc)
            
            ind_tile = self.get_index(*tile_loc) 
            tile = self.tiles[ind_tile]
            tile_center = tile.get_center()
            tile_center_no_z = np.copy(tile_center)
            tile_center_no_z[2] = zero_center[2]
            zero_center_no_z[2]  = tile_center[2]
            
            # Swap positions in Group to preserve index
            self.tiles[ind_zero], self.tiles[ind_tile] = self.tiles[ind_tile], self.tiles[ind_zero]

            #Apply action to game
            self.state.step(action)

            #Show action 
            action_tex = action_to_tex[action]
            action_label =Text(
                ACTION_TO_NAME[action],
                font = 'Libertatus Duas',
                stroke_color = YELLOW,
                stroke_width = 2,
                stroke_opacity= 1)
            action_label.next_to(action_tex,UP)

            #scene.add_sound('sounds/target.wav',time_offset = 0.1)
            scene.play(
                GrowFromCenter(action_tex,run_time = 0.2),
                ShowCreation(action_label,run_time = 0.2)
                )

            #Animate tiles 
            zero_tile.move_to(tile_center_no_z)
            zero_tile.save_state()
            zero_tile.shift(np.array([0,0,-3]))
            
            #sound_path = os.path.join(sounds_dir,np.random.choice(sounds))
            #scene.add_sound(sound_path, time_offset = 0.60)

            scene.play(
                tile.animate.move_to(zero_center_no_z),
                run_time = 0.30)
            zero_tile.restore()
            scene.remove(action_tex,action_label)
            n_moves+= 1
            #scene.wait(1)
        
        executing[1].suspend_updating()
            
class Complexity(Scene):
    def construct(self):
        title = Tex(r'A^{*}').scale(2.2)
        title.to_edge(UP+LEFT)
        self.play(ShowCreation(title))


        color_map = {r'b': BLUE, r'd':RED}
        str_list =[r'\mathcal{O}' ,r'(',r'b',r'^{',r'd',r'})']
        comp = Tex(*str_list,tex_to_color_map = color_map ).scale(3)

        self.play(
            Write(comp, run_time = 3)
        )

        self.wait(5)



class TestBoard(Scene):
    CONFIG = {
        "camera_class": ThreeDCamera,
        "matrix":  [[1,2,5],
            [3,4,0],
            [6,7,8]],
        "search_method":'dfs'
    }

    def construct(self):
        # Setup
        frame = self.camera.frame
        matrix = self.matrix
        digest_config(self, self.CONFIG)
        
        self.solve(matrix = matrix,
            method =  self.search_method)
    def solve(self,matrix, method):

        #Puzzle logic to use with Search
        puzzle_ = Puzzle(matrix)
        #Puzzle logic for the animation
        #puzzle = Puzzle(matrix)
        t0 = time.time()
        S = Search(initial = puzzle_)
        path = S.search(method = method)
        stats  = S.stats 
        print("Number of actions in solution: ", len(path))
        if len(path)>1000:
            path = path[:10]
            print("Truncated solution path: ",path)
        else:
            print("Solution path: ",path)

        print("Initial state puzzle: ", puzzle_)

        board =PuzzleBoard(puzzle_)
        #board.move_to(ORIGIN, OUT)

        self.play(
            ShowCreation(board,run_time = 2))
        self.add(board)

        self.wait(1)
        self.play(
            self.camera.frame.animate.rotate(angle = PI/5, axis= Y_AXIS+ OUT*0.1),
            run_time = 1)
        self.wait(1)

        board.run_solution(solution = path,scene = self)
        nodes_expanded = stats.get('nodes_expanded',0)
        max_search_depth = stats.get('max_search_depth',0)
        run_time =  stats.get('time',0)

        stats_text =VGroup(
            Text(r"""Expanded nodes: """ +  str(nodes_expanded )),
            Text(r"Max search depth: " +  str(max_search_depth )),
            Text(r"Time used: " +  str(round(run_time,3)) + " seconds")
        ).arrange(DOWN)
        
        stats_text.next_to(board.n_moves_text,DOWN)

        self.play(
            Write(stats_text)
        )
        self.wait(3)


class TestGetNeighbors(Scene):
    CONFIG = {
        "camera_class": Camera,
        "camera_config":
            {"samples": 32,
            "anti_alias_width": 0.01},
        "matrix": [
            [1,0,2],
            [3,4,5],
            [6,7,8]]
    }

    def construct(self):
        digest_config(self, self.CONFIG)
        # Setup
        frame = self.camera.frame
        
        matrix = self.matrix
    
        puzzle_ = Puzzle(matrix)
        #puzzle = Puzzle(matrix)
        board =PuzzleBoard(puzzle_)

        self.play(
            ShowCreation(board,run_time = 1))
        self.add(board)
        
        self.play(
            self.camera.frame.animate.rotate(angle = PI/5, axis= Y_AXIS+ OUT*0.1),
            run_time =1)
        
        board.run_solution(solution = [(0,1),(1,0),(1,0),(0,-1),(0,-1)],scene = self)
        self.wait(2)

        
class TestArrowTip(Scene):

    def construct(self):
        arr = ArrowTip(
            angle =  90*DEGREES,
            fill_opacity= 1.0,
            fill_color= RED,
            stroke_width=2.0,
            length = DEFAULT_ARROW_TIP_LENGTH*1.5,
            tip_style= 0,  # triangle=0, inner_smooth=1, dot=2
        ).scale(4)
        
        self.add(arr)
        self.wait(2)

#Game matrices
MATRICES = [ [
            [1,0,2],
            [3,4,5],
            [6,7,8]]]
MATRICES.append([[3,1,2],
                [0,4,5],
                [6,7,8]])
MATRICES.append([[4,1,2],
                    [3,8,5],
                    [6,7,0]])
MATRICES.append([[2,1,4],
            [6,8,5],
            [3,7,0]])

MATRICES.append([[6,1,4],
            [2,5,8],
            [3,7,0]])

MATRICES.append([[6,4,7],
            [5,0,8],
            [3,1,2]])


class TestBoard2(TestBoard):
    CONFIG = {
        "camera_class": ThreeDCamera,
        "matrix": [
            [1,0,2],
            [3,4,5],
            [6,7,8]]
    }



class TestBoard2(TestBoard):
    CONFIG = {
        "camera_class": ThreeDCamera,
        "matrix": [
            [1,0,2],
            [3,4,5],
            [6,7,8]]
    }


class TestBoard3(TestBoard):
    CONFIG = {
        "camera_class": ThreeDCamera,
        "matrix": [[3,1,2],
                [0,4,5],
                [6,7,8]]
    }

class TestBoard4(TestBoard):
    CONFIG = {
        "camera_class": ThreeDCamera,
        "matrix": [[4,1,2],
                    [3,8,5],
                    [6,7,0]]
    }



class TestBoard5(TestBoard):
    CONFIG = {
        "camera_class": ThreeDCamera,
        "matrix":[[2,1,4],
            [6,8,5],
            [3,7,0]]
    }



class TestBoard6(TestBoard):
    CONFIG = {
        "camera_class": ThreeDCamera,
        "matrix":[[6,1,4],
            [2,5,8],
            [3,7,0]]
    }


class TestBoard7(TestBoard):
    CONFIG = {
        "camera_class": ThreeDCamera,
        "matrix":[[6,4,7],
            [5,0,8],
            [3,1,2]]
    }


class TestBFS(TestBoard):
    CONFIG = {
        "camera_class": ThreeDCamera,
        "matrix":[[6,4,7],
            [5,0,8],
            [3,1,2]],
        "search_method":'bfs'
    }

class TestAStar(TestBoard):
    CONFIG = {
        "camera_class": ThreeDCamera,
        "matrix":[[6,4,7],
            [5,0,8],
            [3,1,2]],
        "search_method":'a_star'
    }

class TestAStar(TestBoard):
    CONFIG = {
        "camera_class": ThreeDCamera,
        "matrix":MATRICES[0],
        "search_method":'a_star'
    }

class TestAStar2(TestBoard):
    CONFIG = {
        "camera_class": ThreeDCamera,
        "matrix":MATRICES[1],
        "search_method":'a_star'
    }

class TestAStar3(TestBoard):
    CONFIG = {
        "camera_class": ThreeDCamera,
        "matrix":MATRICES[2],
        "search_method":'a_star'
    }
class TestAStar4(TestBoard):
    CONFIG = {
        "camera_class": ThreeDCamera,
        "matrix":MATRICES[3],
        "search_method":'a_star'
    }