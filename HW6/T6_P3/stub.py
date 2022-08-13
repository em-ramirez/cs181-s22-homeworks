# Imports.
from argparse import Action
from os import stat
import numpy as np
import numpy.random as npr
import pygame as pg

# uncomment this for animation
# from SwingyMonkey import SwingyMonkey

# uncomment this for no animation
from SwingyMonkeyNoAnimation import SwingyMonkey


X_BINSIZE = 200
Y_BINSIZE = 100
X_SCREEN = 1400
Y_SCREEN = 900


class Learner(object):
    """
    This agent jumps randomly.
    """

    def __init__(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None

        self.counter = 0
        self.epsilon = 0.0001
        self.alpha = 0.2
        self.gamma = 0.8

        # We initialize our Q-value grid that has an entry for each action and state.
        # (action, rel_x, rel_y)
        self.Q = np.zeros((2, X_SCREEN // X_BINSIZE, Y_SCREEN // Y_BINSIZE))

    def reset(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None

        self.counter = 0

    def discretize_state(self, state):
        """
        Discretize the position space to produce binned features.
        rel_x = the binned relative horizontal distance between the monkey and the tree
        rel_y = the binned relative vertical distance between the monkey and the tree
        """

        rel_x = int((state["tree"]["dist"]) // X_BINSIZE)
        rel_y = int((state["tree"]["top"] - state["monkey"]["top"]) // Y_BINSIZE)
        return (rel_x, rel_y)

    def action_callback(self, state):
        """
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        """

        # TODO (currently monkey just jumps around randomly)
        # 1. Discretize 'state' to get your transformed 'current state' features.
        # 2. Perform the Q-Learning update using 'current state' and the 'last state'.
        # 3. Choose the next action using an epsilon-greedy policy.

        self.counter += 1

        if self.counter == 1:
            self.last_state = state
            self.last_action = 0

            return self.last_action

        if npr.rand() <= self.epsilon:
            rand_act = None
            if npr.rand() < 0.5:
                rand_act = 1
            else:
                rand_act = 0

            state_pos = self.discretize_state(state)
            self.Q[rand_act][state_pos[0]][state_pos[1]] = self.last_reward

            self.last_state = state
            self.last_action = rand_act
            # # UNCOMMENT FOR EPSILON DECAY FUNCTION:
            # self.epsilon -= 0.00001

            return rand_act
        
        curstate_pos = self.discretize_state(state)
        oldstate_pos = self.discretize_state(self.last_state)

        new_action = np.argmax([self.Q[a][curstate_pos[0]][curstate_pos[1]] for a in range(2)])

        oldstate_q = self.Q[self.last_action][oldstate_pos[0]][oldstate_pos[1]]
        newstate_q = self.Q[new_action][curstate_pos[0]][curstate_pos[1]]
        self.Q[self.last_action][oldstate_pos[0]][oldstate_pos[1]] = oldstate_q + self.alpha*(self.last_reward + self.gamma*(newstate_q - oldstate_q))

        self.last_state = state
        self.last_action = new_action

        return new_action

    def reward_callback(self, reward):
        """This gets called so you can see what reward you get."""

        self.last_reward = reward


def run_games(learner, hist, iters=100, t_len=100):
    """
    Driver function to simulate learning by having the agent play a sequence of games.
    """
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=True,  # Don't play sounds.
                             text="Epoch %d" % (ii),  # Display the epoch on screen.
                             tick_length=t_len,  # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()
    pg.quit()
    return


if __name__ == '__main__':
    # Select agent.
    agent = Learner()

    # Empty list to save history.
    hist = []

    # Run games. You can update t_len to be smaller to run it faster.
    run_games(agent, hist, 100, 100)
    print("PRINTING HIST:", len(hist))
    print(hist)
    print("MAX OF HISTORY:", max(hist))
    print("AVERAGE OF HISTORY: ", sum(hist) / len(hist))

    # Save history. 
    np.save('hist', np.array(hist))
