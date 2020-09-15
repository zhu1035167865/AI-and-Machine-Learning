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
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

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
    Open = util.Stack()
    Open.push(([problem.getStartState()], []))
    while not Open.isEmpty():
        ThisNode = Open.pop()
        StatesInThisNode = ThisNode[0]
        ActionsSoFar = ThisNode[1]
        ThisState = StatesInThisNode[-1]
        if problem.isGoalState(ThisState):
            return ActionsSoFar
        for good in problem.getSuccessors(ThisState):
            if not good[0] in StatesInThisNode:
                Open.push((StatesInThisNode +  [good[0]], ActionsSoFar +  [good[1]]))
    return []

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    Open = util.Queue()
    Open.push(([problem.getStartState()],  []))
    s = {problem.getStartState():  0}
    CostFn = problem.getCostOfActions
    while not Open.isEmpty():
        ThisNode = Open.pop()
        StatesInThisNode = ThisNode[0]
        ActionsSoFar = ThisNode[1]
        ThisState = StatesInThisNode[-1]
        if CostFn(ActionsSoFar) <= s[ThisState]:
            if problem.isGoalState(ThisState):
                return ActionsSoFar
            for good in problem.getSuccessors(ThisState):
                if good[0] not in s or CostFn(ActionsSoFar + [good[1]]) < s[good[0]]:
                    Open.push((StatesInThisNode +  [good[0]], ActionsSoFar +  [good[1]]))
                    s[good[0]] = CostFn(ActionsSoFar +  [good[1]])
    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    def CostFn(Node):
        if hasattr(problem, "costFn"):
            return sum(problem.costFn(State) for State in Node[0])
        return problem.getCostOfActions(Node[1])
    Open = util.PriorityQueueWithFunction(CostFn)
    StartNode = ([problem.getStartState()], [])
    Open.push(StartNode)
    s = {StartNode[0][0]: CostFn(StartNode)}
    while not Open.isEmpty():
        ThisNode = Open.pop()
        StatesInThisNode = ThisNode[0]
        ActionsSoFar = ThisNode[1]
        ThisState = StatesInThisNode[-1]
        if CostFn(ThisNode) <= s[ThisState]:
            if problem.isGoalState(ThisState):
                return ActionsSoFar
            for good in problem.getSuccessors(ThisState):
                NewNode = (StatesInThisNode + [good[0]], ActionsSoFar + [good[1]])
                if good[0] not in s or CostFn(NewNode) < s[good[0]]:
                    Open.push(NewNode)
                    s[good[0]] = CostFn(NewNode)
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    e = lambda Node: (problem.getCostOfActions(Node[1]) + heuristic(Node[0][-1], problem), problem.getCostOfActions(Node[1]))
    Open = util.PriorityQueueWithFunction(e)
    StartNode = ([problem.getStartState()], [])
    Open.push(StartNode)
    s = {StartNode[0][0]: e(StartNode)}
    while not Open.isEmpty():
        ThisNode = Open.pop()
        StatesInThisNode = ThisNode[0]
        ActionsSoFar = ThisNode[1]
        ThisState = StatesInThisNode[-1]
        if e(ThisNode) <= s[ThisState]:
            if problem.isGoalState(ThisState):
                return ActionsSoFar
            for good in problem.getSuccessors(ThisState):
                NewNode = (StatesInThisNode + [good[0]], ActionsSoFar + [good[1]])
                if good[0] not in s or e(NewNode) < s[good[0]]:
                    Open.push(NewNode)
                    s[good[0]] = e(NewNode)
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
