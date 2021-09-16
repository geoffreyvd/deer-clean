"""Maze launcher

Author: Vincent Francois-Lavet
"""

import sys
import logging
import numpy as np
from joblib import hash, dump
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
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
    EPOCHS = 50
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
    LEARNING_RATE = 0.00008 #used to be 
    LEARNING_RATE_DECAY = 1 #used to be 1 and commented 0.995 #maybe change to 0.95 since epoch has 2000 steps (instead of 5000 with 0.9)
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
N_SAMPLES=200000


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # --- Parse parameters ---
    parameters = process_args(sys.argv[1:], Defaults)
    fname = "transferred" 
    input_nnet = "normal"
    if parameters.deterministic:
        rng = np.random.RandomState(parameters.seed)
        print(" deterministic, seed: ",parameters.seed)
        fname += "_seed" + str(parameters.seed)
        input_nnet += "_seed" + str(parameters.seed)
    else:
        rng = np.random.RandomState()
    if parameters.dumpname != "":
        fname = parameters.dumpname + "_transferred"
        input_nnet = parameters.dumpname
        if parameters.deterministic:
            fname += "_seed" + str(parameters.seed)

    fname += "_lr" + str(parameters.learning_rate) + "_lrd" + str(parameters.learning_rate_decay) 
    
    if parameters.mode == 1:
        fname += "_resetencoder"
    if parameters.mode == 2:
        fname += "_partialfreezeencoder"
    # if parameters.mode == 3:
    #     #TODO dont freeze but very small lr for others models

    print("saving nnet,plot and score under name=", fname)
    print("input nnet= ", input_nnet)


    # --- Instantiate environment ---
    # env = maze_env(rng, higher_dim_obs=HIGHER_DIM_OBS)
    env = maze_env(rng, higher_dim_obs=parameters.high_dim_obs, reverse=True)
    
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
        div_entrop_loss=1.,
        fname=fname)
    
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

    # --- Create unique filename for FindBestController ---
    print("The parameters are: {}".format(parameters))

    # --- Bind controllers to the agent ---
    # Before every training epoch (periodicity=1), we want to print a summary of the agent's epsilon, discount and 
    # learning rate as well as the training epoch number.
    agent.attach(bc.VerboseController(
        evaluate_on='epoch', 
        periodicity=1))
    
    # Every epoch end, one has the possibility to modify the learning rate using a LearningRateController. Here we 
    # wish to update the learning rate after every training epoch (periodicity=1), according to the parameters given.
    agent.attach(bc.LearningRateController(
        initial_learning_rate=parameters.learning_rate, 
        learning_rate_decay=parameters.learning_rate_decay,
        periodicity=1))
    
    # Same for the discount factor.
    agent.attach(bc.DiscountFactorController(
        initial_discount_factor=parameters.discount, 
        discount_factor_growth=parameters.discount_inc, 
        discount_factor_max=parameters.discount_max,
        periodicity=1))
        
    # As for the discount factor and the learning rate, one can update periodically the parameter of the epsilon-greedy
    # policy implemented by the agent. This controllers has a bit more capabilities, as it allows one to choose more
    # precisely when to update epsilon: after every X action, episode or epoch. This parameter can also be reset every
    # episode or epoch (or never, hence the resetEvery='none').
    agent.attach(bc.EpsilonController(
        initial_e=parameters.epsilon_start, 
        e_decays=parameters.epsilon_decay, 
        e_min=parameters.epsilon_min,
        evaluate_on='action',
        periodicity=1,
        reset_every='none'))

    agent.run(1, N_SAMPLES)
    
    #print (agent._dataset._rewards._data[0:500])
    #print (agent._dataset._terminals._data[0:500])
    print("end gathering data")
    old_rewards=agent._dataset._rewards._data
    old_terminals=agent._dataset._terminals._data
    old_actions=agent._dataset._actions._data
    old_observations=agent._dataset._observations[0]._data

    # During training epochs, we want to train the agent after every [parameters.update_frequency] action it takes.
    # Plus, we also want to display after each training episode (!= than after every training) the average bellman
    # residual and the average of the V values obtained during the last episode, hence the two last arguments.
    agent.attach(bc.TrainerController(
        evaluate_on='action', 
        periodicity=parameters.update_frequency, 
        show_episode_avg_V_value=True, 
        show_avg_Bellman_residual=True))
    
    agent.attach(bc.FindBestController(
        validationID=2,
        testID=None,
        unique_fname=fname))

    agent.attach(bc.InterleavedTestEpochController(
        id=2, 
        epoch_length=parameters.steps_per_test,
        periodicity=1,
        show_score=True,
        summarize_every=1))

    # --- Run the experiment ---
    try:
        os.mkdir("params")
    except Exception:
        pass
    dump(vars(parameters), "params/" + fname + ".jldump")
    agent.gathering_data=False
    # agent.setNetwork("test_4165747fe50541da92a5ea2698b190b90bc006d5.epoch=97")
    agent.setNetwork(input_nnet)

    #freeze network except encoder
    if parameters.mode == 0:
        agent._learning_algo.freezeAllLayersExceptEncoder()
    if parameters.mode == 1:
        agent._learning_algo.freezeAllLayersExceptEncoder()
        agent._learning_algo.resetEncoder()
    if parameters.mode == 2:
        agent._learning_algo.freezeAllLayersExceptEncoderPartially()
    # if parameters.mode == 3:
    #     #TODO dont freeze but very small lr for others models
    print("without reset")
    
    agent.run(parameters.epochs, parameters.steps_per_epoch)
