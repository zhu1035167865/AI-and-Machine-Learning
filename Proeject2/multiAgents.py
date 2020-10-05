# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score=0.0
        for i in newGhostStates:
            j=i.getPosition()
            disGhost=manhattanDistance(j,newPos)
            if(i.scaredTimer!=0):
                if(disGhost==0):
                    score+=1000
                else:
                    score+=500/disGhost
            else:
                if(disGhost<2):
                    score-=1000
        for i in currentGameState.getCapsules():
            disCap=manhattanDistance(i,newPos)
            if(disCap==0):
                score+=100
            else:
                score+=10.0/disCap
        food=newFood.asList()        
        for i in food:
            disFood=manhattanDistance(i,newPos)
            score+=10.0/disFood
        nowFood=currentGameState.getFood().asList()
        for i in nowFood:
            disNowFood=manhattanDistance(i,newPos)
            if(disNowFood==0):
                score+=100
        if(action==Directions.STOP):
            score-=100    
        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & 
    PacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        self.PACMAN = 0
        action = self.max_agent(gameState, 0)
        return action
    
    def max_agent(self, state, depth):
        if state.isLose() or state.isWin():
            return state.getScore()
        actions = state.getLegalActions(self.PACMAN)
        best_score = float("-inf")
        preferred = Directions.STOP
        for action in state.getLegalActions(self.PACMAN):
            score = self.min_agent(state.generateSuccessor(self.PACMAN, action), depth, 1)
            if score > best_score:
                best_score = score
                preferred = action
        if depth == 0:
            return preferred
        else:
            return best_score

    def min_agent(self, state, depth, agent):
        if state.isLose() or state.isWin():
            return state.getScore()
        next_agent = agent + 1
        if agent == state.getNumAgents() - 1:
            next_agent = self.PACMAN
        actions = state.getLegalActions(agent)
        best_score = float("inf")
        for action in actions:
            if next_agent == self.PACMAN:
                if depth == self.depth - 1:
                    score = self.evaluationFunction(state.generateSuccessor(agent, action))
                else:
                    score = self.max_agent(state.generateSuccessor(agent, action), depth+1)
            else:
                score = self.min_agent(state.generateSuccessor(agent, action), depth, next_agent)
            best_score = min(score, best_score)
        return best_score
    

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        self.PACMAN = 0
        action = self.max_agent(gameState, 0, float("-inf"), float("inf"))
        return action
    
    def max_agent(self, state, depth, alpha, beta):
        if state.isLose() or state.isWin():
            return state.getScore()
        actions = state.getLegalActions(self.PACMAN)
        best_score = float("-inf")
        preferred = Directions.STOP
        for action in actions:
            score = self.min_agent(state.generateSuccessor(self.PACMAN, action), depth, 1, alpha, beta)
            if score > best_score:
                best_score = score
                preferred = action
            alpha = max(alpha, best_score)
            if best_score > beta:
                return best_score
        if depth == 0:
            return preferred
        else:
            return best_score

    def min_agent(self, state, depth, agent, alpha, beta):
        if state.isLose() or state.isWin():
            return state.getScore()
        next_agent = agent + 1
        if agent == state.getNumAgents() - 1:
            next_agent = self.PACMAN
        actions = state.getLegalActions(agent)
        best_score = float("inf")
        for action in actions:
            if next_agent == self.PACMAN:
                if depth == self.depth - 1:
                    score = self.evaluationFunction(state.generateSuccessor(agent, action))
                else:
                    score = self.max_agent(state.generateSuccessor(agent, action), depth + 1, alpha, beta)
            else:
                score = self.min_agent(state.generateSuccessor(agent, action), depth, next_agent, alpha, beta)
            best_score = min(score, best_score)
            beta = min(beta, best_score)
            if best_score < alpha:
                return best_score
        return best_score



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.ExpectiMax(gameState, 1, 0)

    def ExpectiMax(self, gameState, currentDepth, agentIndex):
        if currentDepth > self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        legalMoves = [action for action in gameState.getLegalActions(agentIndex) if action!='Stop']
        nextIndex = agentIndex + 1
        nextDepth = currentDepth
        if nextIndex >= gameState.getNumAgents():
            nextIndex = 0
            nextDepth += 1
        results = [self.ExpectiMax( gameState.generateSuccessor(agentIndex, action) , nextDepth, nextIndex) for action in legalMoves]
        if agentIndex == 0 and currentDepth == 1: 
            bestMove = max(results)
            bestIndices = [index for index in range(len(results)) if results[index] == bestMove]
            chosenIndex = random.choice(bestIndices) 
            return legalMoves[chosenIndex]
        if agentIndex == 0:
            bestMove = max(results)
            return bestMove
        else:
            bestMove = sum(results)/len(results)
            return bestMove
    
def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():  return float("inf")
    if currentGameState.isLose(): return float("-inf")
    
    def manhattan(xy1, xy2):
        return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])
    
    manhattans_food = [ ( manhattan(currentGameState.getPacmanPosition(), food) ) for food in currentGameState.getFood().asList() ]
    min_manhattans_food = min(manhattans_food)
    manhattans_ghost = [ manhattan(currentGameState.getPacmanPosition(), ghostState.getPosition()) for ghostState in currentGameState.getGhostStates() if ghostState.scaredTimer == 0 ]
    min_manhattans_ghost = -3
    if ( len(manhattans_ghost) > 0 ): 
        min_manhattans_ghost = min(manhattans_ghost)
    manhattans_ghost_scared = [ manhattan(currentGameState.getPacmanPosition(), ghostState.getPosition()) for ghostState in currentGameState.getGhostStates() if ghostState.scaredTimer > 0 ]
    min_manhattans_ghost_scared = 0;
    if ( len(manhattans_ghost_scared) > 0 ): 
        min_manhattans_ghost_scared = min(manhattans_ghost_scared)
    score = scoreEvaluationFunction(currentGameState)
    score += -1.5 * min_manhattans_food
    score += -2 * ( 1.0 / min_manhattans_ghost )
    score += -2 * min_manhattans_ghost_scared
    score += -20 * len(currentGameState.getCapsules())
    score += -4 * len(currentGameState.getFood().asList())
    return score
# Abbreviation
better = betterEvaluationFunction
