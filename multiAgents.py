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
        return successorGameState.getScore()

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
        # Return the action returned from the maxValue function
        return self.maxValue(gameState, self.depth)[1]

    # Pacman
    def maxValue(self, gameState, depth):
        # If game is won, lost or we're at the bottom of the tree, return an evaluation of the gamestate
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState), Directions.STOP

        # Pacman want to maximize values
        # bestValue therefore starts at -infinity to find a higher value than that
        bestValue = -float("inf")
        # Find the possible directions Pacman can move. Pacman is agent 0
        legalActions = gameState.getLegalActions(0)
        # Default action is to STOP
        returnAction = Directions.STOP

        # For each possible direction Pacman can move
        for action in legalActions:
            # Recursion, find the ghost moves that will happen as a consequence of Pacmans move
            value = self.minValue(gameState.generateSuccessor(0, action), depth, 1)[0]

            # If value is higher than bestValue, remember it and the corresponding action
            if value >= bestValue:
                bestValue = value
                returnAction = action

        return bestValue, returnAction

    # Ghosts
    def minValue(self, gameState, depth, agent):
        # If game is won, lost or we're at the bottom of the tree, return an evaluation of the gamestate
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState), Directions.STOP

        # Ghosts want to minimize values
        # minValue therefore starts at infinity to find a lower value than that
        bestValue = float("inf")
        # Find the possible directions this Ghost can move. Ghosts are agents >= 1
        legalActions = gameState.getLegalActions(agent)
        # Default action is to STOP
        returnAction = Directions.STOP

        for action in legalActions:
            # Completed finding moves for all ghosts, go back to packman and decrease depth by 1
            if (agent == gameState.getNumAgents() - 1):
                # Recursion, find the pacman moves that will happen as a consequence of the Ghosts moves
                value = self.maxValue(gameState.generateSuccessor(agent, action), depth - 1)[0]
            # Still have more ghosts to find moves for
            else:
                # Recursion, call to the same function. Agent=1 the first time and then increases as each ghost is looked at
                value = self.minValue(gameState.generateSuccessor(agent, action), depth, agent + 1)[0]

            # If value is lower than minValue, remember it and the corresponding action
            if value <= bestValue:
                bestValue = value
                returnAction = action

        return bestValue, returnAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        # Return the action returned from the maxValue function
        return self.maxValue(gameState, self.depth, -float("inf"), float("inf"))[1]

    # Pacman
    def maxValue(self, gameState, depth, alpha, beta):
        # If game is won, lost or we're at the bottom of the tree, return an evaluation of the gamestate
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState), Directions.STOP

        # Pacman want to maximize values
        # bestValue therefore starts at -infinity to find a higher value than that
        bestValue = -float("inf")
        # Find the possible directions Pacman can move. Pacman is the agent at 0
        legalActions = gameState.getLegalActions(0)
        # Default action is to STOP
        returnAction = Directions.STOP

        # For each possible direction Pacman can move
        for action in legalActions:
            # Recursion, find the ghost moves that will happen as a consequence of Pacmans move
            value = self.minValue(gameState.generateSuccessor(0, action), depth, 1, alpha, beta)[0]

            # If value is higher than bestValue, remember it and the corresponding action
            if value >= bestValue:
                bestValue = value
                returnAction = action

            # Find the highest of alpha and value
            alpha = max(alpha, value)

            # If value is higher than beta, prune all later actions for the current Pacman position
            if value > beta:
                break

        return bestValue, returnAction

    # Ghosts
    def minValue(self, gameState, depth, agent, alpha, beta):
        # If game is won, lost or we're at the bottom of the tree, return an evaluation of the gamestate
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState), Directions.STOP

        # Ghosts want to minimize values
        # minValue therefore starts at infinity to find a lower value than that
        bestValue = float("inf")
        # Find the possible directions this Ghost can move. Ghosts are agents at >= 1
        legalActions = gameState.getLegalActions(agent)
        # Default action is to STOP
        returnAction = Directions.STOP

        for action in legalActions:
            # Completed finding moves for all ghosts, go back to packman and decrease depth by 1
            if (agent == gameState.getNumAgents() - 1):
                # Recursion, find the pacman moves that will happen as a consequence of the Ghosts moves
                value = self.maxValue(gameState.generateSuccessor(agent, action), depth - 1, alpha, beta)[0]
            # Still have more ghosts to find moves for
            else:
                # Recursion, call to the same function. Agent=1 the first time and then increases as each ghost is looked at
                value = self.minValue(gameState.generateSuccessor(agent, action), depth, agent + 1, alpha, beta)[0]

            # If value is lower than minValue, remember it and the corresponding action
            if value <= bestValue:
                bestValue = value
                returnAction = action

            # Find the lowest of beta and value
            beta = min(beta, value)

            # If alpha is higher than values, prune all later actions for the current Ghost position
            if alpha > value:
                break

        return bestValue, returnAction

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
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

