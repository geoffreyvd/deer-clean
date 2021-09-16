"""
Code for the CRAR learning algorithm using Keras

"""

import numpy as np
from keras.optimizers import SGD,RMSprop, Adam
from keras import backend as K
from ..base_classes import LearningAlgo
from .NN_CRAR_keras_categoric_abs import NN # Default Neural network used
#import tensorflow as tf
#config = tf.ConfigProto()
#config.gpu_options.allow_growth=True
#sess = tf.Session(config=config)

import matplotlib.pyplot as plt
#this did work for me:
import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
import copy

UPDATE_AFTER_STEPS= 500. 


def mean_squared_error_p(y_true, y_pred):
    """ Modified mean square error that clips
    """
    return K.clip(K.max(  K.square( y_pred - y_true )  ,  axis=-1  )-1,0.,100.)     # = modified mse error L_inf
    #return K.clip(K.mean(  K.square( y_pred - y_true )  ,  axis=-1  )-1,0.,100.)   # = modified mse error L_2

def exp_dec_error(y_true, y_pred):
    return K.exp( - 5.*K.sqrt( K.clip(K.sum(K.square(y_pred), axis=-1, keepdims=True),0.000001,10) )  ) # tend to increase y_pred

def cosine_proximity2(y_true, y_pred):
    """ This loss is similar to the native cosine_proximity loss from Keras
    but it differs by the fact that only the two first components of the two vectors are used
    """
    y_true = K.l2_normalize(y_true[:,0:2], axis=-1)
    y_pred = K.l2_normalize(y_pred[:,0:2], axis=-1)
    return -K.sum(y_true * y_pred, axis=-1)

class CRAR(LearningAlgo):
    """
    Combined Reinforcement learning via Abstract Representations (CRAR) using Keras
    
    Parameters
    -----------
    environment : object from class Environment
        The environment in which the agent evolves.
    rho : float
        Parameter for rmsprop. Default : 0.9
    rms_epsilon : float
        Parameter for rmsprop. Default : 0.0001
    momentum : float
        Momentum for SGD. Default : 0
    clip_norm : float
        The gradient tensor will be clipped to a maximum L2 norm given by this value.
    freeze_interval : int
        Period during which the target network is freezed and after which the target network is updated. Default : 1000
    batch_size : int
        Number of tuples taken into account for each iteration of gradient descent. Default : 32
    update_rule: str
        {sgd,rmsprop}. Default : rmsprop
    random_state : numpy random number generator
        Set the random seed.
    double_Q : bool, optional
        Activate or not the double_Q learning.
        More informations in : Hado van Hasselt et al. (2015) - Deep Reinforcement Learning with Double Q-learning.
    neural_network : object, optional
        Default is deer.learning_algos.NN_keras
    """
 
    def __init__(self, environment, rho=0.9, rms_epsilon=0.0001, momentum=0, clip_norm=0, freeze_interval=1000, batch_size=32, update_rule="rmsprop", random_state=np.random.RandomState(), double_Q=False, neural_network=NN, **kwargs):
        """ Initialize the environment
        
        """
        LearningAlgo.__init__(self,environment, batch_size)

        self._rho = rho
        self._rms_epsilon = rms_epsilon
        self._momentum = momentum
        self._clip_norm = clip_norm
        self._update_rule = update_rule
        self._freeze_interval = freeze_interval
        self._double_Q = double_Q
        self._random_state = random_state
        self.update_counter = 0    
        self._high_int_dim = kwargs.get('high_int_dim',False)
        self._internal_dim = kwargs.get('internal_dim',2)
        self._div_entrop_loss = kwargs.get('div_entrop_loss',5.)

        self.lossR=0
        self.lossCap_encoder=0 #keep everything in a radius of 1

        #memorize losses to plot
        self.lossR_list=[]
        self.lossCap_encoder_list=[]
        

        self.learn_and_plan = neural_network(self._batch_size, self._input_dimensions, self._n_actions, self._random_state, high_int_dim=self._high_int_dim, internal_dim=self._internal_dim)
        self.encoder = self.learn_and_plan.encoder_model()        
        self.R = self.learn_and_plan.float_model()
        # used to fit rewards
        self.full_R = self.learn_and_plan.full_float_model(self.encoder,self.R)
        self.cap_encoder = self.learn_and_plan.encoder_cap_model(self.encoder)
               
        # Grab all the parameters in self.params
        layers=self.encoder.layers+self.R.layers
        self.params = [ param
                    for layer in layers 
                    for param in layer.trainable_weights ]

        # Compile all models
        self._compile()


    def getAllParams(self):
        """ Provides all parameters used by the learning algorithm

        Returns
        -------
        Values of the parameters: list of numpy arrays
        """
        params_value=[]
        for i,p in enumerate(self.params):
            params_value.append(K.get_value(p))
        return params_value

    def setAllParams(self, list_of_values):
        """ Set all parameters used by the learning algorithm

        Arguments
        ---------
        list_of_values : list of numpy arrays
             list of the parameters to be set (same order than given by getAllParams()).
        """
        for i,p in enumerate(self.params):
            K.set_value(p,list_of_values[i])
   
    def train(self, states_val, actions_val, rewards_val, next_states_val, terminals_val):
        """
        Train CRAR from one batch of data.

        Parameters
        -----------
        states_val : numpy array of objects
            Each object is a numpy array that relates to one of the observations
            with size [batch_size * history size * size of punctual observation (which is 2D,1D or scalar)]).
        actions_val : numpy array of integers with size [self._batch_size]
            actions[i] is the action taken after having observed states[:][i].
        rewards_val : numpy array of floats with size [self._batch_size]
            rewards[i] is the reward obtained for taking actions[i-1].
        next_states_val : numpy array of objects
            Each object is a numpy array that relates to one of the observations
            with size [batch_size * history size * size of punctual observation (which is 2D,1D or scalar)]).
        terminals_val : numpy array of booleans with size [self._batch_size]
            terminals[i] is True if the transition leads to a terminal state and False otherwise

        Returns
        -------
        Average loss of the batch training for the Q-values (RMSE)
        Individual (square) losses for the Q-values for each tuple
        """
        
        onehot_actions = np.zeros((self._batch_size, self._n_actions))
        onehot_actions[np.arange(self._batch_size), actions_val] = 1
        onehot_actions_rand = np.zeros((self._batch_size, self._n_actions))
        onehot_actions_rand[np.arange(self._batch_size), np.random.randint(0,2,(32))] = 1
        states_val=list(states_val)
        next_states_val=list(next_states_val)
            
        Es_=self.encoder.predict(next_states_val)
        Es=self.encoder.predict(states_val)
        R=self.R.predict([Es,onehot_actions]) #interesting, try this appraoch
                   
        if(self.update_counter%UPDATE_AFTER_STEPS==0):
            print ("Printing a few elements useful for debugging:")
            #print ("states_val[0][0]")
            #print (states_val[0][0])
            #print ("next_states_val[0][0]")
            #print (next_states_val[0][0])
            print ("actions_val[0], rewards_val[0], terminals_val[0]")
            print (actions_val[0], rewards_val[0], terminals_val[0])
            print ("Es[0],Es_[0]")
            if(Es.ndim==4):
                print (np.transpose(Es, (0, 3, 1, 2))[0],np.transpose(Es_, (0, 3, 1, 2))[0])    # data_format='channels_last' --> 'channels_first'
            else:
                print (Es[0],Es_[0])
            print ("R[0]")
            print (R[0])
            
        # Fit rewards
        self.lossR+=self.full_R.train_on_batch(states_val+[onehot_actions], rewards_val)    
        # self.lossCap_encoder+=self.cap_encoder.train_on_batch(states_val+[onehot_actions],np.reshape(np.zeros_like(Es),(self._batch_size,-1)))

        self.lossCap_encoder+=self.cap_encoder.train_on_batch(states_val,np.zeros_like(Es))


        totalLoss = self.lossR + self.lossCap_encoder
        
        if(self.update_counter%UPDATE_AFTER_STEPS==0):
            lossRAvg = self.lossR/UPDATE_AFTER_STEPS
            lossCap_encoderAvg = self.lossCap_encoder/UPDATE_AFTER_STEPS
            self.lossR=0    
            self.lossR_list.append(lossRAvg)
            self.lossCap_encoder_list.append(lossCap_encoderAvg)
            print("loss R", lossRAvg)
            print("loss R", lossCap_encoderAvg)

            plt.plot(range(1, len(self.lossR_list)+1), self.lossR_list, label="R", color='b')
            plt.plot(range(1, len(self.lossR_list)+1), self.lossCap_encoder_list, label="C", color='r')
            plt.legend()
            plt.xlabel("Number of epochs (x500)")
            plt.ylabel("Loss")
            plt.savefig("losses.pdf")
            plt.close()
            # plt.show()
                    
        if(self.update_counter%100==0):
            print ("Number of training steps:"+str(self.update_counter)+".")
        
        self.update_counter += 1        

        return np.sqrt(totalLoss),totalLoss/self.update_counter


    def chooseBestAction(self, state, mode, *args, **kwargs):
        """ Get the best action for a pseudo-state

        Arguments
        ---------
        state : list of numpy arrays
             One pseudo-state. The number of arrays and their dimensions matches self.environment.inputDimensions().
        mode : int
            Identifier of the mode (-1 is reserved for the training mode).

        Returns
        -------
        The best action : int
        """

        # self._n_actions
        print (" TODO random best eaction")
        return 1
        
    def _compile(self):
        """ Compile all the optimizers for the different losses
        """
        if (self._update_rule=="sgd"):
            optimizer3=SGD(lr=self._lr, momentum=self._momentum, nesterov=False, clipnorm=self._clip_norm) # to possibly modify them separately
            optimizer4=SGD(lr=self._lr, momentum=self._momentum, nesterov=False, clipnorm=self._clip_norm)
            optimizer5=SGD(lr=self._lr, momentum=self._momentum, nesterov=False, clipnorm=self._clip_norm)
        elif (self._update_rule=="rmsprop"):
            optimizer3=RMSprop(lr=self._lr, rho=self._rho, epsilon=self._rms_epsilon, clipnorm=self._clip_norm) # to possibly modify them separately
            optimizer4=RMSprop(lr=self._lr, rho=self._rho, epsilon=self._rms_epsilon, clipnorm=self._clip_norm)
            # optimizer4=Adam(lr=self._lr, epsilon=self._rms_epsilon, clipnorm=self._clip_norm)
            optimizer5=RMSprop(lr=self._lr, rho=self._rho, epsilon=self._rms_epsilon, clipnorm=self._clip_norm)

        else:
            raise Exception('The update_rule '+self._update_rule+' is not implemented.')
        
        self.full_R.compile(optimizer=optimizer3, loss='mse') # Fit rewards
        #TODO whats the differenc ebewteen mse and msep 
        self.encoder.compile(optimizer=optimizer4,
                  loss=mean_squared_error_p)
        self.cap_encoder.compile(optimizer=optimizer5,
                  loss='mse')
        #TODO use different loss exp_dec_error or smaller exponentiality

        #TODO different loss (cross entropy or something)
        # self.encoder.compile(optimizer=optimizer4,
        #           loss='binary_crossentropy')
        # dont use binary_crossentropy, as we dont have target labels in 1 or 0 fashion
        #maybe different optimizer? adam?
        #also check sparse categorical crossentrpoy loss
        #maybe different loss, blog uses MSE in case of straight through
        #tomorrow just try taking a sample and using MSE, if ti doesnt work just pass it thorugh see if it works

    def setLearningRate(self, lr):
        """ Setting the learning rate

        Parameters
        -----------
        lr : float
            The learning rate that has to be set
        """
        self._lr = lr
        print ("New learning rate set to "+str(self._lr)+".")
        # Changing the learning rates (NB:recompiling seems to lead to memory leaks!)
        K.set_value(self.full_R.optimizer.lr, self._lr)

        K.set_value(self.encoder.optimizer.lr, self._lr)
        K.set_value(self.cap_encoder.optimizer.lr, self._lr)