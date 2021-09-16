#TODO instantiate a maze, 

import sys
import logging
import numpy as np
from joblib import hash, dump
import os

from deer.default_parser import process_args
from deer.agent import NeuralAgent
from deer.learning_algos.CRAR_keras import CRAR
from maze_env import MyEnv as maze_env
import deer.experiment.base_controllers as bc

from deer.policies import EpsilonGreedyPolicy

import matplotlib.pyplot as plt

class Defaults:
    # ----------------------
    # Experiment Parameters
    # ----------------------
    STEPS_PER_EPOCH = 2000
    EPOCHS = 100
    STEPS_PER_TEST = 200
    PERIOD_BTW_SUMMARY_PERFS = 1
    
    # ----------------------
    # Environment Parameters
    # ----------------------
    FRAME_SKIP = 2

    # ----------------------
    # DQN Agent parameters:
    # ----------------------
    UPDATE_RULE = 'rmsprop'
    LEARNING_RATE = 0.0005
    LEARNING_RATE_DECAY = 1.#0.995
    DISCOUNT = 0.9
    DISCOUNT_INC = 1
    DISCOUNT_MAX = 0.99
    RMS_DECAY = 0.9
    RMS_EPSILON = 0.0001
    MOMENTUM = 0
    CLIP_NORM = 1.0
    EPSILON_START = 1.0
    EPSILON_MIN = 1.0
    EPSILON_DECAY = 10000
    UPDATE_FREQUENCY = 1
    REPLAY_MEMORY_SIZE = 1000000
    BATCH_SIZE = 32
    FREEZE_INTERVAL = 1000
    DETERMINISTIC = False
    SEED = 123456
    DUMPNAME = ''
    MODE = 0
    HIGH_DIM_OBS = True

HIGH_INT_DIM = True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # --- Parse parameters ---
    parameters = process_args(sys.argv[1:], Defaults)
    input_transferred = "transferred_lr" + str(parameters.learning_rate) + "_lrd" + str(parameters.learning_rate_decay) 
    input_normal = "normal_lr" + str(parameters.learning_rate) + "_lrd" + str(parameters.learning_rate_decay)
    if parameters.deterministic:
        rng = np.random.RandomState(parameters.seed)
        print(" deterministic, seed: ",parameters.seed)
        input_transferred = "transferred_seed" + str(parameters.seed) + "_lr" + str(parameters.learning_rate) + "_lrd" + str(parameters.learning_rate_decay) 
        input_normal = "normal_seed" + str(parameters.seed) + "_lr" + str(parameters.learning_rate) + "_lrd" + str(parameters.learning_rate_decay) 
    else:
        rng = np.random.RandomState()
    if parameters.dumpname != "":
        input_transferred = parameters.dumpname + "_transferred"
        input_normal = parameters.dumpname
    print("input transferred nnt=", input_transferred)
    print("input normal nnet= ", input_normal)


    input_normal = parameters.dumpname
    
    # --- Instantiate environment ---
    env = maze_env(rng, higher_dim_obs=parameters.high_dim_obs)
    
    # --- Instantiate learning_algo ---
    learning_algo = CRAR(
        env,
        parameters.rms_decay,
        parameters.rms_epsilon,
        parameters.momentum,
        parameters.clip_norm,
        parameters.freeze_interval,
        parameters.batch_size,
        parameters.update_rule,
        rng,
        double_Q=True,
        high_int_dim=HIGH_INT_DIM,
        internal_dim=3,
        div_entrop_loss=1.)
    
    learning_algo2 = CRAR(
        env,
        parameters.rms_decay,
        parameters.rms_epsilon,
        parameters.momentum,
        parameters.clip_norm,
        parameters.freeze_interval,
        parameters.batch_size,
        parameters.update_rule,
        rng,
        double_Q=True,
        high_int_dim=HIGH_INT_DIM,
        internal_dim=3,
        div_entrop_loss=1.)

    train_policy = EpsilonGreedyPolicy(learning_algo, env.nActions(), rng, 1.)
    test_policy = EpsilonGreedyPolicy(learning_algo, env.nActions(), rng, 0.1)

    train_policy2 = EpsilonGreedyPolicy(learning_algo2, env.nActions(), rng, 1.)
    test_policy2 = EpsilonGreedyPolicy(learning_algo2, env.nActions(), rng, 0.1)

    # --- Instantiate agent ---
    agent = NeuralAgent(
        env,
        learning_algo,
        parameters.replay_memory_size,
        max(env.inputDimensions()[i][0] for i in range(len(env.inputDimensions()))),
        parameters.batch_size,
        rng,
        train_policy=train_policy,
        test_policy=test_policy)

    agent2 = NeuralAgent(
        env,
        learning_algo2,
        parameters.replay_memory_size,
        max(env.inputDimensions()[i][0] for i in range(len(env.inputDimensions()))),
        parameters.batch_size,
        rng,
        train_policy=train_policy2,
        test_policy=test_policy2)

    print("The parameters are: {}".format(parameters))

    #1 load normal and transferred network
    # agent.setNetwork("test_4165747fe50541da92a5ea2698b190b90bc006d5.epoch=97")
    # agent2.setNetwork("nnet.epoch=40")

    agent.setNetwork(input_normal) #IMPORTANT: first one gets normal visuals
    # agent2.setNetwork(input_transferred) #IMPORTANT: this one gets inverted visuals
    #SO if you want to compare two normal nnets, remove the true in agent2.getabstract state!!
    # agent.setNetwork("test31")
    # agent2.setNetwork("test36")
    iterations = 40
    totaldiff = 0
    for i in range(iterations):
        #2 generate an environment and get abstract states
        agent.resetEnv()
        abstract_state = agent.getAbstractState()
        print("abstract_state= ", abstract_state)

        plt.figure()
        plt.imshow(np.squeeze(abstract_state))
        plt.show()

        # abstract_state2 = agent2.getAbstractState(True)
        # # abstract_state2 = agent2.getAbstractState()
        # # print("abstract_state2= ", abstract_state2)

        # plt.figure()
        # plt.imshow(np.squeeze(abstract_state2))
        # plt.show()

        # #3 compute L2 distance
        # diff = np.sqrt(np.sum((abstract_state-abstract_state2)**2))
        # print(diff)
        # totaldiff += diff
    # diff = np.linalg.norm(abstract_state-abstract_state2)
    # print("average diff= ", totaldiff/iterations)


    #TODO repeat 20 times or something
    #TODO write to a file or something input_normal + "L2distanceAbstractState"