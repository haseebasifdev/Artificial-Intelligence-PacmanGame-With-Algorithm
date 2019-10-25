# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import math

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    n = Directions.NORTH
    s = Directions.SOUTH
    e = Directions.EAST
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]
def mediumClassicSearch1 (problem):
    from game import Directions
    n=Directions.NORTH
    s=Directions.SOUTH
    e=Directions.EAST
    w=Directions.WEST
    return[w, w, w, n, n, w, w, w, n, n, w, w, n, n, n, n,e,e,e,s,s,e,e,s,s,s,s,e,e,e,e,e,e,e,e,e,s,s,e,e,e]

def mediumMazeSearch(problem):
    from game import Directions
    n=Directions.NORTH
    s=Directions.SOUTH
    e=Directions.EAST
    w=Directions.WEST
    #return[]
    return[s,s,w,w,w,w,s,s,e,e,e,e,s,s,s,w,w,w,s,s,e,e,e,s,s,w,w,w,s,s,e,e,e,s,s,w,w,w,w]        
def bigMazeSearch(problem):
    from game import Directions
    n=Directions.NORTH
    s=Directions.SOUTH
    e=Directions.EAST
    w=Directions.WEST
    return[n,n,w,w,w,w,n,n,w,w,s,s,w,w,w,w,w,w,w,w,w,w,w,w,w,w,n,n,e,e,n,n,w,w,n,n,n,n,n,n,e,e,e,s,e,n,e,e,n,e,s,e,n,n,n,e,e,n,n,n,n,n,w,s,w,w,s,s,s,w,s,s,w,n,w,w,w,w,w,w,n,e,n,n,n,e,e,e,n,n,n,n,w,w,w,s,w,w,s,s,e,s,w,w,w,w,w,s,s,s,s,s,e,s,s,w,s,s,e,s,w,s,s,w,s]
def mySearch(problem):
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    startState=problem.getStartState()
    childStates=problem.getSuccessors(startState)
    leftChild=childStates[0]

  
    print(startState)
    print(childStates)
    print(leftChild)
    return [s]








def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    initialState=problem.getStartState()
    Fringes=util.Stack()
    explored=set()
    Fringes.push((initialState,[]))
    while not Fringes.isEmpty():
        newstate,path=Fringes.pop()
        
        if problem.isGoalState(newstate):
            return path
        explored.add(newstate)
        
        childstate=problem.getSuccessors(newstate)
        for c in childstate:
            if  c[0] not in explored:
                newpath=path[:]
                newpath.append(c[1])
                Fringes.push((c[0],newpath))
    
    return []
    """"util.raiseNotDefined()
    currentstate=problem.getStartState()
    action=[]
    explored=[]
    maxiteration=0
    while (maxiteration<=40):
        print("Explored",explored)
        childrens=problem.getSuccessors(currentstate)
        action.append(getActionFromTriple(childrens[0]))
        explored.append(childrens[0][0])
        firstchild=childrens[0]
        secondchild=childrens[1]
        firstchildstate=firstchild[0]
        secondchildstate=secondchild[0]
        if firstchildstate not in explored:
            currentstate=firstchildstate
        else:
            currentstate=secondchildstate
            
        maxiteration=maxiteration+1return action """

def getActionFromTriple(triple):
    return triple[1]

def iteration(problem):
    currentstate=problem.getStartState()
    action=[]
    store=[]
    maxiteration=0
    while maxiteration<=20:
        child=problem.getSuccessors(currentstate)
        action.append(getActionFromTriple(child[0]))
        firstchild=child[0]
        firstchildstate=firstchild[0]
        currentstate=firstchildstate
        store=currentstate
        maxiteration=maxiteration+1
    return action
def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    "*** YOUR CODE HERE ***"
    initialState=problem.getStartState()
    print(initialState)
    Fringes=util.Queue()
    explored=set()
    Fringes.push((initialState,[]))
    while not Fringes.isEmpty():
        newstate,path=Fringes.pop()
        
        if problem.isGoalState(newstate):
            return path
        explored.add(newstate)
        
        childstate=problem.getSuccessors(newstate)
        for c in childstate:
            if  c[0] not in explored:
                newpath=path[:]
                newpath.append(c[1])
                Fringes.push((c[0],newpath))
    
    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    fringe=util.PriorityQueue()
    explored=set()
    startStateBlock=problem.getStartState()
    fringe.push((startStateBlock,[]),0)
    while(not fringe.isEmpty()):
        state=fringe.pop()
        stateBlock=state[0]
        statePath=state[1].copy()
        explored.add(stateBlock)
        if (problem.isGoalState(stateBlock)):
            return statePath
        children=problem.getSuccessors(stateBlock)
        for child in children:
            actionToReachChild=child[1]
            costToReachChilld=child[2]
            childPath=statePath.copy()
            childPath.append(actionToReachChild)

            openList=[x[0] for x in fringe.heap if x[0]==child]

            inProcess=child[0] in explored or child[0] in openList
            if (not inProcess):
                fringe.push((child[0],childPath),costToReachChilld)
            elif (child[0] in openList):
                fringe.update((child[0],childPath),costToReachChilld)

    return []


    #util.raiseNotDefined()
    
def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    fringe=util.PriorityQueue()
    explored=set()
    startStateBlock=problem.getStartState()
    gcost=0
    hcost=manhattanHeuristic(startStateBlock,problem)
    #hcost=euclindean_distance(startStateBlock,problem)
    fcost=gcost+hcost
    fringe.push((startStateBlock,[]),fcost)
    while(not fringe.isEmpty()):
        state=fringe.pop()
        stateBlock=state[0]
        statePath=state[1].copy()
        explored.add(stateBlock)
        if (problem.isGoalState(stateBlock)):
            return statePath
        children=problem.getSuccessors(stateBlock)

        for child in children:
            actionToReachChild=child[1]
            gcost=child[2]
            hcost=manhattanHeuristic(child[0],problem)
#            hcost=euclindean_distance(child[0],problem)
            fcost=gcost+hcost
            childPath=statePath.copy()
            childPath.append(actionToReachChild)
            openList=[x[0] for x in fringe.heap if x[0]==child[0]]

            inProcess=child[0] in explored or child[0] in openList
            if (not inProcess):
                fringe.push((child[0],childPath),fcost)
            elif (child[0] in openList):
                fringe.update((child[0],childPath),fcost)
    return []
            
    #util.raiseNotDefined()
def manhattanHeuristic(state,problem):
    cstate=state
    gstate=problem.goal
    hstate=abs(gstate[0]-cstate[0])+abs(gstate[1]-cstate[1])
    return hstate

def euclindean_distance(state,problem):
    cstate=state
    gstate=problem.goal
    hcost=math.sqrt(((gstate[0]-cstate[0])**2)+((gstate[1]-cstate[1])**2))
    return hcost
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
