
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
    MODE = 3
    HIGH_DIM_OBS = True

HIGH_INT_DIM = True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # --- Parse parameters ---
    parameters = process_args(sys.argv[1:], Defaults)
    input_nnet = "normal_lr" + str(parameters.learning_rate) + "_lrd" + str(parameters.learning_rate_decay)
    if parameters.deterministic:
        rng = np.random.RandomState(parameters.seed)
        print(" deterministic, seed: ",parameters.seed)
        input_nnet = "normal_seed" + str(parameters.seed) + "_lr" + str(parameters.learning_rate) + "_lrd" + str(parameters.learning_rate_decay) 
    else:
        rng = np.random.RandomState()
    if parameters.dumpname != "":
        input_nnet = parameters.dumpname
    print("input nnet= ", input_nnet)

    #automatically detect if u need inverse env
    reversedEnv = False
    if "_transferred" in input_nnet:
        reversedEnv = True

    # --- Instantiate environment ---
    env = maze_env(rng, higher_dim_obs=parameters.high_dim_obs, show_game=True, reverse=reversedEnv)
    
    # --- Instantiate learning_algo ---
    learning_algo = CRAR(
        env,
        rng,
        double_Q=True,
        high_int_dim=HIGH_INT_DIM,
        internal_dim=3,
        div_entrop_loss=1.)
    
    train_policy = EpsilonGreedyPolicy(learning_algo, env.nActions(), rng, 1.)
    test_policy = EpsilonGreedyPolicy(learning_algo, env.nActions(), rng, 0.1)

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

    # --- load saved network and test
    # agent.setNetwork("test_4165747fe50541da92a5ea2698b190b90bc006d5.epoch=97")
    agent.setNetwork(input_nnet) #tesot01

    avg = agent._total_mode_reward
    print(" _total_mode_reward: ", agent._total_mode_reward, ", nmbr of episode: ", agent._totalModeNbrEpisode, ", average per episode: ", avg)

    Epoch_length = 200
    mode = parameters.mode #mode 3 has planning depth 6#mode 2 ahs planning 3
    agent.startMode(mode, Epoch_length)
    agent.run(1, Epoch_length)

    avg = agent._total_mode_reward / agent._totalModeNbrEpisode
    print(" _total_mode_reward: ", agent._total_mode_reward, ", nmbr of episode: ", agent._totalModeNbrEpisode, ", average per episode: ", avg)
    
    #just testing the saved nnet (possibly by visualizing the actions in the env)