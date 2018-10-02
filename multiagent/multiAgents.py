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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        # chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        chosenIndex = bestIndices[0]  # Pick randomly among the best

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
        # return successorGameState.getScore()
        closestGhoastDist = min([manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])
        # print closestGhoastDist

        if closestGhoastDist:
            closestGhoastDist = -10 / closestGhoastDist
        else:
            closestGhoastDist = -9999

        # if closestGhoastDist:
        #     closestGhoastDist = -0.000001
        # else:
        #     closestGhoastDist = -9999

        foodList = newFood.asList()
        if foodList:
            closestFoodDist = 99999999
            for food in foodList:
                closestFoodDist = min(manhattanDistance(newPos, food), closestFoodDist)
        else:
            closestFoodDist = 0

        return (closestGhoastDist - 100 * len(foodList)-2 * closestFoodDist)

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
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

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
        """
        "*** YOUR CODE HERE ***"
        agentCount = gameState.getNumAgents()

        def minimax(state, curDepth, agentId):
            if agentId == agentCount:
                if curDepth == self.depth:
                    return self.evaluationFunction(state)
                else:
                    return minimax(state, curDepth + 1, 0)
            else:
                actions = state.getLegalActions(agentId)
                if len(actions) == 0:
                    return self.evaluationFunction(state)
                next = (minimax(state.generateSuccessor(agentId, action), curDepth, agentId + 1) for action in actions)

                if agentId == 0:
                    value = -999999
                    for action in actions:
                        value = max(value, minimax(state.generateSuccessor(agentId, action), curDepth, agentId + 1))
                else:
                    value = 999999
                    for action in actions:
                        value = min(value, minimax(state.generateSuccessor(agentId, action), curDepth, agentId + 1))
                return value

        # result = max(gameState.getLegalActions(0),
        #              key=lambda x: minimax(gameState.generateSuccessor(0, x), 1, 1))
        actions = gameState.getLegalActions(0)
        finalScore = -999999
        for action in actions:
            nextState = gameState.generateSuccessor(0, action)
            curScore = minimax(nextState, 1, 1)
            if curScore > finalScore:
                resAction = action
                finalScore = curScore

        return resAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def alphaBetaSearch(state):
            # value, bestAction = -999999, None
            # alpha, beta = -99999999, 99999999
            value, bestAction = None, None
            alpha, beta = None, None

            for action in state.getLegalActions(0):
                value = max(value, minValue(state.generateSuccessor(0, action), 1, 1, alpha, beta))
                if alpha is None:
                    alpha = value
                    bestAction = action
                else:
                    alpha, bestAction = max(value, alpha), action if value > alpha else bestAction
            return bestAction

        def maxValue(state, agentIdx, depth, alpha, beta):
            if depth > self.depth:
                return self.evaluationFunction(state)
            value = None
            # value = -99999999
            for action in state.getLegalActions(agentIdx):
                succ = minValue(state.generateSuccessor(agentIdx, action), agentIdx + 1, depth, alpha, beta)
                if value is None:
                    value = succ
                else:
                    value = max(value, succ)
                if beta is not None and value > beta:
                # if value >= beta:
                    return value
                alpha = max(alpha, value)
            # if value != -99999999:
            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)

        def minValue(state, agentIdx, depth, alpha, beta):
            if agentIdx == state.getNumAgents():
                return maxValue(state, 0, depth + 1, alpha, beta)
            value = None
            # value = 99999999
            for action in state.getLegalActions(agentIdx):
                succ = minValue(state.generateSuccessor(agentIdx, action), agentIdx + 1, depth, alpha, beta)
                if value is None:
                    value = succ
                else:
                    value = min(value, succ)
                if alpha is not None and value < alpha:
                # if value <= alpha:
                    return value
                if beta is None:
                    beta = value
                else:
                    beta = min(beta, value)
            if value is not None:
            # if value != 99999999:
                return value
            else:
                return self.evaluationFunction(state)

        resAction = alphaBetaSearch(gameState)

        return resAction

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

        agentsCount = gameState.getNumAgents()
        def expectimax_search(state, depth, agentId):
            if agentId == agentsCount:
                if depth == self.depth:
                    return self.evaluationFunction(state)
                else:
                    return expectimax_search(state, depth + 1, 0)
            else:
                actions = state.getLegalActions(agentId)
                if len(actions) == 0:
                    return self.evaluationFunction(state)

                if agentId == 0:
                    value = -9999999
                    for action in actions:
                        value = max(value, expectimax_search(state.generateSuccessor(agentId, action), depth, agentId + 1))
                    return value
                else:
                    value = 0
                    for action in actions:
                        value += expectimax_search(state.generateSuccessor(agentId, action), depth, agentId + 1)
                    return value / float(len(actions))

        finalScore = -999999
        for action in gameState.getLegalActions(0):
            curScore = expectimax_search(gameState.generateSuccessor(0, action), 1, 1)
            if curScore > finalScore:
                resAction = action
                finalScore = curScore

        return resAction

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newCapsules = currentGameState.getCapsules()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    foodList = newFood.asList()
    foodDistance = [0]
    for pos in foodList:
        foodDistance.append(manhattanDistance(newPos, pos))

    finalValue = 0
    for capsule in newCapsules:
        finalValue += 1/manhattanDistance(newPos, capsule)

    closestGhoastDist = min([manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])

    numberofPowerPellets = len(currentGameState.getCapsules())

    foodCountScore = len(newFood.asList(False))
    sumScaredTimes = sum(newScaredTimes)
    foodDistanceScore = 0
    if sum(foodDistance) > 0:
        foodDistanceScore = 1.0 / sum(foodDistance)

    finalValue += currentGameState.getScore() + foodDistanceScore + foodCountScore

    if sumScaredTimes > 0:
        finalValue += sumScaredTimes + (-1 * numberofPowerPellets) + (-1 * closestGhoastDist)
    else:
        if closestGhoastDist < 4:
            finalValue += closestGhoastDist + numberofPowerPellets
        else:
            finalValue += numberofPowerPellets + 4
    return finalValue


# Abbreviation
better = betterEvaluationFunction

