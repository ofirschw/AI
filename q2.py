import numpy as np
import random
###########################################  analize boards here ###########################
starting_board =  np.array([[2, 0, 2, 0, 2, 0], [0, 0, 0, 2, 1, 2], [1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 1, 0], [2, 0, 0, 0, 0, 0],[0, 1, 0, 0, 0, 0]])
goal_board = np.array([[2, 0, 2, 0, 0, 0], [0, 0, 0, 2, 1, 2], [1, 0, 0, 0, 0, 2], [0, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 0],[0, 1, 0, 0, 0, 0]])
###########################################  analize boards here ###########################
def check_board(check_board): ##function to check if board is good

    for i in range(len(check_board)):
        for j in range(len(check_board[i])):
            if check_board[i][j] != 1 and check_board[i][j] != 2 and check_board[i][j] != 0:
                 i == 7
                 j == 7
                 print()
                 print("your board is not include only 0,1,2")
                 return 1
            if len(check_board) != 6 or len(check_board[i]) != 6:
                print()
                print("your board is to much longer for this code")
                return 1
    if type (check_board) != np.ndarray:
                print()
                print("your board is not the type for this code please make him np.array")
                return  1
    return 0

def check_blockers (start_board,end_board): #check if @ is same of both tables
    for i in range(len(start_board)):
        for j in range(len(start_board[i])):
            if start_board[i][j] == 1 and end_board[i][j] != 1:
                print()
                print("the start and end board dont have the same number of @, please fix the board")
                return 1
    return 0

def print_board(working_board, index , last_index , detail_output,search_check): # function to print the boards
    if index==1:
        print("Board 1 (starting position):")
        print_from_print(working_board,index,detail_output, last_index,search_check)
    elif index==2 and index!=last_index:
        if search_check!=5 or detail_output == False:
            print("Board 2: ")
        if search_check == 5 and detail_output == True:
            print("Board 2 (probability of selection from population::<", working_board.genetic_prob, ">:", sep='')
        print_from_print(working_board, index, detail_output, last_index, search_check)
    elif index==last_index and np.array_equal(goal_board,working_board.board)==True:
        if search_check == 1 or search_check == 2 or search_check == 3 or search_check == 4 or (search_check==5 and detail_output == False) or (search_check==5) and index>3:
            print("Board" , index,  "(goal position):")
        print_from_print(working_board,index,detail_output, last_index,search_check)
    else:
        if search_check==1 or search_check==2 or search_check==3 or search_check==4 or (search_check==5 and detail_output == False) or (search_check==5 and index>3):
            print("Board", index , ":")
        elif search_check == 5 and detail_output == True:
            check = 2
        print_from_print(working_board,index,detail_output , last_index,search_check)

def print_from_print(working_board , index,detail_output, last_index,search_check): # function to print the boards
    print_board = working_board.board
    flag_genetic_print = 0
    if search_check == 5 and detail_output == True and index == 3:
        print("Board 2 (probability of selection from population::<",working_board.genetic_prob,">:",sep='')
        print(' ', '1', '2', '3', '4', '5', '6')
        for i in range(len(print_board)):
            print((i + 1), ':', sep='', end='')
            for j in range(len(print_board[i])):
                if working_board.genetic_parents[1].board[i][j] == 2:
                    print("*", end=' ')
                elif working_board.genetic_parents[1].board[i][j] == 1:
                    print("@", end=' ')
                else:
                    print(" ", end=' ')
            print()
        if working_board.mutate == 1:
            print("Result board (mutation happened::<yes>):")
        else:
            print("Result board (mutation happened::<no>):")
    print(' ', '1', '2', '3', '4', '5', '6')
    for i in range(len(print_board)):
        print((i + 1), ':', sep='', end='')
        for j in range(len(print_board[i])):
            if print_board[i][j] == 2:
                print("*", end=' ')
            elif print_board[i][j] == 1:
                print("@", end=' ')
            else:
                print(" ", end=' ')

        print()
    if search_check == 3 and detail_output == True: ##print for algo 3
        for w in range(len(working_board.e)):
            print("action: ", end='')
            for z in working_board.steps[w]:
                printindex = 0
                for x in z:
                    if printindex == 0 or printindex == 2:
                        print("(", x, end=',', sep='')

                    else:
                        if printindex == 1:
                            print(x, end=')->', sep='')
                        else:
                            print(x, end=')')
                    printindex = printindex + 1
            print(";", end='', sep="   ")
            print("   probability:", working_board.e[w])
    if search_check == 4 and index<last_index  and detail_output == True:  ##print for algo 4
        for m in range (len(working_board.beams)):
            asci = m+97
            print('----')
            print("Board ",index,chr(asci),":" ,sep='')
            print(' ', '1', '2', '3', '4', '5', '6')
            for i in range(len(print_board)):
                print((i + 1), ':', sep='', end='')
                for j in range(len(print_board[i])):
                    if working_board.beams[m].board[i][j] == 2:
                        print("*", end=' ')
                    elif working_board.beams[m].board[i][j] == 1:
                        print("@", end=' ')
                    else:
                        print(" ", end=' ')
                print()


    if (index >= 2  or (index == last_index and last_index<=2)) and detail_output==True and (search_check ==1 or search_check==2): ##print for algo 1,2
        print("Heuristic: " , working_board.h)
    print('-----')



def find_path(starting_board,goal_board,search_method,detail_output): #find path to the goal board
    if search_method == 1:
        starfunction(starting_board,goal_board,detail_output)
    if search_method == 2:
        chance = 1
        find = 0
        list_random = []
        while chance < 6 and find==0:
            find,list_random = hillclimbingfunction(starting_board,goal_board,detail_output,chance,list_random)
            chance = chance + 1

    if search_method ==3:
        Simulated_annealing_algo(starting_board,goal_board,detail_output)

    if search_method == 4:
        local_beam_search(starting_board,goal_board,3,detail_output)

    if search_method == 5:
        genetic_algorithm(starting_board,goal_board,detail_output)

class board_node(): #every board will get his data
    def __init__ (self,board,parent,goalboard):
        self.board = board
        self.parent = parent
        self.g = 0
        self.h = self.check_h(goalboard)
        self.f = self.g + self.h
        self.e = []
        self.steps = []
        self.beams = []
        self.genetic_prob = 0
        self.genetic_parents = []
        self.mutate = 0
    def set_g (self,g):
        self.g = g
    def set_f (self):
        self.f = self.g + self.h
    def set_e (self,e):
        self.e.append(e)
    def set_steps (self,step):
        self.steps.append(step)

    def check_h(self, goalboard):
        check_numof2 = 0
        check_numof2Goal = 0
        h_result = 0
        for i in range (len(self.board)):
            for j in range (len(self.board[i])):
                if self.board[i][j] == 2:
                    check_numof2 = check_numof2 + 1;
        for i in range(len(goalboard)):
            for j in range(len(goalboard[i])):
                if goalboard[i][j] == 2:
                    check_numof2Goal = check_numof2Goal + 1;
        h = (check_numof2 - check_numof2Goal)
        sum_error = 0
        on_target = 0
        for i in range(len(goalboard)):
            for j in range(len(goalboard[i])):
                if goalboard[i][j] == 2:
                    find_near = 10000
                    if self.board[i][j] != 2:
                        for l in range(len(goalboard)):
                            for k in range(len(goalboard[i])):
                               if self.board[l][k] == 2 and goalboard[l][k]!=2:
                                   near = abs(i-l) +abs(j-k)
                                   if find_near > near:
                                       find_near = near
                    else:
                        find_near = 0
                    sum_error = sum_error + find_near
        return h+sum_error

    def get_children (self,goal_board): #every board can bring the data about his children
        children_list = []
        for i in range (len(self.board)):
            for j in range (len(self.board[i])):
                if j>0 and self.board[i][j] == 2 and self.board[i][j-1] != 2 and self.board[i][j-1]!=1:  # create left child
                    create_child(self.board, i, j, "j-1", children_list,self,goal_board)
                if j<5 and self.board[i][j] == 2  and self.board[i][j+1] != 2 and self.board[i][j+1]!=1:  # create right child
                    create_child(self.board, i, j, "j+1", children_list,self,goal_board)
                if i>0 and self.board[i][j] == 2  and self.board[i-1][j] != 2 and self.board[i-1][j]!=1: #create up child
                    create_child(self.board, i, j, "i-1", children_list,self,goal_board)
                if i<5 and self.board[i][j] == 2 and self.board[i+1][j] != 2 and self.board[i+1][j] != 1 : #create down child
                    create_child(self.board, i, j, "i+1", children_list,self,goal_board)
                if i==5 and self.board[i][j] == 2:
                    create_child(self.board, i, j, "5", children_list,self,goal_board)

        return children_list

def create_child (board,i,j,to,children_list,parent,goal_board): #create child and give him his parent
    if to == "j-1":
        child = np.copy(board)
        child[i][j] = 0
        child[i][j - 1] = 2
        add_child = board_node(child,parent,goal_board)
        add_child.set_steps([i,j,i,j-1])
        children_list.append(add_child)
    if to == "j+1":
        child = np.copy(board)
        child[i][j] = 0
        child[i][j + 1] = 2
        add_child = board_node(child, parent,goal_board)
        add_child.set_steps([i, j, i, j + 1])
        children_list.append(add_child)
    if to == "i-1":
        child = np.copy(board)
        child[i][j] = 0
        child[i-1][j] = 2
        add_child = board_node(child, parent,goal_board)
        add_child.set_steps([i, j, i-1, j])
        children_list.append(add_child)
    if to == "i+1":
        child = np.copy(board)
        child[i][j] = 0
        child[i+1][j] = 2
        add_child = board_node(child,parent,goal_board)
        add_child.set_steps([i, j, i + 1,j])
        children_list.append(add_child)
    if to == "5":
        child = np.copy(board)
        child[i][j] = 0
        add_child = board_node(child,parent,goal_board)
        add_child.set_steps([i, j, "out", j])
        children_list.append(add_child)
#######################################################Print functios########################################################################
def return_path (end_node, detail_output,search_check): #check what was the path of the boards
    print_nodes = []
    print_nodes.append(end_node)
    while end_node.parent is not None:
        print_nodes.append(end_node.parent)
        end_node = end_node.parent
    index = len(print_nodes)
    count_print = 1
    while count_print <= index:
        for pos in reversed(print_nodes):
            print_board(pos,count_print,index , detail_output,search_check)
            count_print = count_print + 1

def return_path_genetic (end_node, detail_output,search_check): #check what was the path of the boards
    print_genetic_nodes = []
    print_genetic_nodes.append(end_node)
    while end_node.parent is None:
        print_genetic_nodes.append(end_node.genetic_parents[0])
        end_node = end_node.genetic_parents[0]
    print_genetic_nodes.append(end_node.parent)
    index = len(print_genetic_nodes)
    count_print = 1
    while count_print <= index:
        for pos in reversed(print_genetic_nodes):
            print_board(pos,count_print,index , detail_output,search_check)
            count_print = count_print + 1
#######################################################Print functios################################################################
#######################################################A-star########################################################################
def starfunction(starting_board,goal_board, detail_output):
    visited = []
    open_list = []
    start_node = board_node (starting_board,None,goal_board)
    open_list.append(start_node)
    end_node = board_node(goal_board,None,goal_board)
    start_node.g=0
    end_node.g = 0
    outer_iteraion = 0
    max_iteraions = 1000;
    iteraion = 1
    while len(open_list)>0: #the main while to find the path, run until open list has nodes or number of iteration to much
        outer_iteraion = outer_iteraion + 1
        node_current = open_list[0]
        current_index = 0
        index = 0
        for pos in open_list: #Take from the open list the node node_current with the lowest f
            if  node_current.f > pos.f:
                node_current = pos
                current_index = index
            index = index + 1
        open_list.pop(current_index)


        if outer_iteraion > max_iteraions: #if number of iteraion is more then 1000
            return print("No path found.")


        if np.array_equal(node_current.board,end_node.board): #if node_current is node_goal we have found the solution
            print("the number of itertaion to solve: " , iteraion)
            return return_path(node_current , detail_output,1)

        children_currents_nodes = node_current.get_children(goal_board) #Generate each state child that come after node_current


        for children in children_currents_nodes: #for each child of node_current
            visted_check = 0
            children.set_g(node_current.g+1) #Set successor_current_cost
            children.set_f()
            check_index = 0
            in_open = 0
            for visit_node in visited:
                if np.array_equal(children.board,visit_node.board): #Check to see if he has visited this child
                    visted_check = 1
            if visted_check == 0:
                for open_node in open_list:
                    if np.array_equal(children.board,open_node.board):
                        in_open = 1
                        if children.g > open_node.g: #if g(child) > successor_current_cost continue
                            continue
                        elif children.g < open_node.g: #if g(child) â‰¤ successor_current_cost continue
                            open_list.pop(check_index)
                            open_list.append(children)
                    check_index = check_index + 1
            if in_open == 0 and visted_check==0:
                open_list.append(children)
        visited.append(node_current) #mark as visited the current node for next steps
        iteraion=iteraion+1;
#######################################################A-star########################################################################
#######################################################Hill-climbing########################################################################
def findRandomSolution(starting_board,goal_board,random_list): ##to find the start child
    random_starting_board = starting_board.get_children(goal_board)# Generate each state child that come after node_current
    listRandom= []
    if len(random_list)<len(random_starting_board) and len(random_starting_board)>0: ##to remember if i was already checked him
        for i in range(len(random_list)):
            listRandom.append(random_list[i])
        length_random_list = len(random_starting_board)
        randomCheck = False
        while randomCheck == False:
            n = random.randint(0, length_random_list - 1)
            found = False
            for i in range(len(listRandom)):
                if listRandom[i] == n:
                    found = True
            if found == True:
                randomCheck == False
            elif len(listRandom)==0 or found == False:
                randomCheck = True
        listRandom.append(n)
        index = 0
        for child in random_starting_board:
            if index == n:
                 return child , listRandom
            else:
                index = index+1
    elif len(random_list)==len(random_starting_board) and len(random_starting_board)>0:
        child = random_starting_board[0]
    elif len(random_starting_board)==0:
        print("i dont have child to make the algo")
        return 0,0
    return child , listRandom


def getBestNeighbour(neigbours): ##to find the best neighbour from the list
    bestRouteLength = neigbours[0].h
    bestNeighbour = neigbours[0]
    for neighbour in neigbours:
        currentRouteLength = neighbour.h
        if currentRouteLength < bestRouteLength:
            bestRouteLength = currentRouteLength
            bestNeighbour = neighbour
    return bestNeighbour, bestRouteLength


def hillclimbingfunction (starting_board,goal_board,detail_output,numberOfchance,random_list):
    start_node = board_node (starting_board,None,goal_board)
    if np.array_equal(start_node.board,goal_board):
         return_path(start_node,detail_output,2)
         return 1,random_list
    currentSolution,random_list = findRandomSolution(start_node,goal_board,random_list) #inilaize the first solution
    if currentSolution ==0 and random_list == 0:
        return 1,random_list
    if np.array_equal(currentSolution.board,goal_board):
        return_path(currentSolution, detail_output, 2)
        return 1, random_list
    currentRouteLength = currentSolution.h
    neighbours  = currentSolution.get_children(goal_board) #find the children of node
    if np.array_equal(currentSolution.board,goal_board):
         return_path(currentSolution,detail_output,2)
         return 1,random_list
    if len(neighbours)==0:
        return 1, random_list
    bestNeighbour, bestNeighbourRouteLength = getBestNeighbour(neighbours)
    while bestNeighbourRouteLength < currentRouteLength: ##keep to search when improveing your solution
        currentSolution = bestNeighbour
        currentRouteLength = bestNeighbourRouteLength
        if  np.array_equal(currentSolution.board, goal_board):
            return_path(currentSolution, detail_output, 2)
            return 1, random_list
        neighbours = currentSolution.get_children(goal_board)
        if len(neighbours)==0:
            return 1,random_list
        bestNeighbour, bestNeighbourRouteLength = getBestNeighbour(neighbours)
    if np.array_equal(currentSolution.board, goal_board):  # if node_current is node_goal we have found the solution
        return_path(currentSolution, detail_output,2)
        return 1,random_list
    elif numberOfchance == 5:
        print("No path found.")
        return 1,random_list
    else:
        return 0,random_list
#######################################################Hill-climbing########################################################################
#######################################################simulated-annealing########################################################################
import math
def Simulated_annealing_algo(starting_board,goal_board,detail_output): #function simulated annealing 1- temperature = 50
    current = board_node (starting_board,None,goal_board) ##to start the algorithm from starting board
    random_list = []
    for t in range (100): ## time = 50
        T =(1-(t+1)/100) # new Temparature
        if  T<=0 or np.array_equal(current.board,goal_board):
            if T<=0 and np.array_equal(current.board,goal_board)==False:
                return print("No path found.")
            if np.array_equal(current.board,goal_board)==True:
                return return_path(current ,detail_output,3)
        current_value=current.h
        next , trash= findRandomSolution(current,goal_board,random_list) ##to start from random child
        current.set_steps(next.steps)
        next.steps = []
        next_value= next.h
        delta_E=current_value-next_value
        e =  math.exp(delta_E/T)
        e_update = 0
        if delta_E>=0: ## check if the next board better the current board
            current.set_e(1)
            e_update = 1
            current = next
        elif e > random.random(): ##in e prob is greater than prob between 0-1 keep to lower one
            current.set_e(e)
            current = next
            e_update = 1
        if e_update == 0:
            current.set_e(e)

#######################################################simulated-annealing########################################################################
#######################################################Local-beam K==3########################################################################
def local_beam_search(starting_board,goal_board,k,detail_output):  # local beam search algorithm with k=3
    x = 100 ##trying of search and open child
    start_node = board_node(starting_board,None,goal_board)
    if np.array_equal(start_node.board,goal_board):
        return return_path(start_node, detail_output, 4)
    rememberK = k ##to remember if i change k because he hasnt 3 child
    children_find = start_node.get_children(goal_board)
    start_best_chlidren , found , goal_child_board = get_best(children_find,goal_board,k,detail_output)
    if start_best_chlidren == 1:
        x = -1
    start_node.beams = start_best_chlidren
    dont_return = 0
    while x>=0:
        if len(start_best_chlidren) < rememberK:
            k = len(start_best_chlidren)
        else:
            k = rememberK
        best_from_k_children = []
        send_list = []
        for i in range (k): ##getting all children from current node
            best_from_k_children.append(start_best_chlidren[i].get_children(goal_board))
        for j in range (len(best_from_k_children)): ## checking the k best children
            for z in range (len(best_from_k_children[j])):
                send_list.append(best_from_k_children[j][z])
        get_best_k_list , found , goal_child_board= get_best(send_list,goal_board,k,detail_output)
        for i in range(k): ##to remember the k nodes
            start_best_chlidren[i].beams = get_best_k_list
        if found == 1:
           return return_path(goal_child_board, detail_output, 4)
        start_best_chlidren = get_best_k_list
        x=x-1
    if dont_return == 0: ##check if didnt fount the solution
        return print("No path found.")


def get_best (children_list,goal_board, indexofiter,detail_output ): #help to find the best h to move forward
    minimum_child = children_list[0]
    return_minimum = []
    child_minimum_index = 0
    found_end = 0
    index = 0
    k_index = 0
    if len(children_list)<=indexofiter: ##if i cant bring k child
        return children_list,0,0
    else:
        while k_index < indexofiter: ##do it K times to find K best
            for child in children_list:  # Take from the open list the node node_current with the lowest h
                if np.array_equal(child.board, goal_board): ##check if this is the end node
                    goal_child = child
                    found_end = 1
                if minimum_child.h > child.h: ##if this is the best keep him
                    minimum_child = child
                    child_minimum_index = index
                index = index + 1
            return_minimum.append(children_list[child_minimum_index])
            children_list.pop(child_minimum_index)
            index = 0
            child_minimum_index = 0
            minimum_child = children_list[0]
            k_index = k_index + 1
        if found_end == 1:
            return return_minimum , 1 , goal_child
        else :
            return return_minimum , 0 , 0
#######################################################Local-beam K==3########################################################################
#######################################################Genetic algorithm########################################################################
def genetic_algorithm(starting_board,goal_board,detail_output): #function genetic algorithm 1 - popolation size 10
    start_node = board_node(starting_board, None, goal_board)
    if np.array_equal(start_node.board,goal_board):
        return print("both of start and end are equals")
    new_population_option = start_node.get_children(goal_board) ## define new popolation sizt 10
    new_population,check,check1 = get_best(new_population_option,goal_board , 10 ,detail_output)
    next_population = []
    indexOfnewpop =0
    time_running = 250
    while len(new_population)<10 and indexOfnewpop < len(new_population_option):
        add_population = new_population_option[indexOfnewpop].get_children(goal_board)
        for i in range(len(add_population)):
            new_population.append(add_population[i])
        new_population,trash1,trash2 = get_best(new_population_option,goal_board , 10 ,detail_output)
        indexOfnewpop = indexOfnewpop + 1
    if indexOfnewpop > len(new_population):
        print("I couldnt find 10 children from the start board and his children, it will try with: " , len(new_population) , "population size")
    fitness_fn(new_population)
    while time_running > 0:
        for i in range(len(new_population)):
            x=random_selection(new_population) ## new borad 1
            y=random_selection(new_population) ## new borad 2
            child= board_node(reproduce(x,y),None , goal_board) ## creat child from board 1 and 2
            child.genetic_parents.append(x)
            child.genetic_parents.append(y)
            mutate_random=random.random()
            if (mutate_random>0.8):
                child.mutate = 1
                child.board , child.h = mutate(child,goal_board) ## create a mutate
            if np.array_equal(child.board, goal_board):
                return return_path_genetic (child, detail_output,5)
            next_population.append(child) ## put child on next generation population
        time_running = time_running - 1
        new_population = next_population
        fitness_fn(new_population)
        next_population = []
    return print("No path found.")
def fitness_fn(population):
    sumH = 0
    sumprob = 0
    for i in range(len(population)):
       sumH = sumH + 1/population[i].h
    for i in range(len(population)):
        population[i].genetic_prob = (1/population[i].h/sumH)

def mutate(child_mutate,goal_board): #function genetic algorithm mutate - best option to move forward
    mutate_children = child_mutate.get_children(goal_board)
    if len(mutate_children) == 0:
        new_zero_board = np.copy(child_mutate.board)
        return new_zero_board , 20
    else:
        return_child,h = getBestNeighbour(mutate_children)
        new_board = np.copy(return_child.board)
        return new_board , h

def reproduce(x,y): #function genetic algorithm to reproduce a new child
    new_child=np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])
    n=6
    c= random.random()*6
    for i in range(n):
        for j in range(n):
            if ((j>=0) & (j<=c)):
                new_child[i][j] = x.board[i][j]
            else:
                new_child[i][j] = y.board[i][j]
    return new_child

def random_selection(population): #function genetic algorithm to create new board by prob
    random_child_prob= random.random()
    prob_list = []
    index_list = np.array([])
    for i in range(len(population)):
        if i==0:
            prob_list.append(population[i].genetic_prob)
        elif i == len(population):
            prob_list.append(1)
        else:
            prob_list.append(prob_list[i-1]+population[i].genetic_prob)
    for i in range(len(prob_list)):
            if random_child_prob <= prob_list[i] and i == 0:
                return population[i]
            elif i+1 < len(prob_list) and i>0:
                if random_child_prob > prob_list[i] and random_child_prob <= prob_list[i+1]:
                    return population[i]
                else:
                    continue
            elif i == len(prob_list)-1:
                return population[i]
#######################################################Genetic algorithm########################################################################

if __name__ == '__main__':   #main function to start
    check_start_board = check_board (starting_board)
    check_end_board = check_board (goal_board )
    check_blockers_board =  check_blockers(starting_board,goal_board)
    if check_start_board == 0 and check_end_board==0 and check_blockers_board == 0: #check if there is any problem with boards
        find_path(starting_board,goal_board,5,True) #find path