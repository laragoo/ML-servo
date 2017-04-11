from servo import Servo
import numpy as np
import random
from keras.models import Sequential
import tensorflow as tf
#from keras.engine.training import collect_trainable_weights    # TODO: remove
import json
import pandas as pd

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
#import timeit                                                 # TODO: remove
from os import listdir


OU = OU()       #Ornstein-Uhlenbeck Process

# NOTE: currently for training using preset data - not good for online training/game-playing
def playGame(train_indicator=1, live=0, files=None, tset=0, band=0):
    '''
    train_indicator: 1 means train, 0 means run
    live: 1 means live training (on system), 0 means pre-training (using saved data files)
    files: list of data files to use for pre-training
    '''
    print "Live: " + str(live)

    BUFFER_SIZE = 100000    # TODO?
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001     #Target Network HyperParameters
    LRA = .000001#0.0001    #Learning rate for Actor
    LRC = .001#0.001     #Lerning rate for Critic

    action_dim = 1  # DAC output
    state_dim = 3   # of sensor inputs

    np.random.seed(1337)

    EXPLORE = 100000.
    episode_count = 0
    if live:
        tset = tset
        episode_count = 1
    else:
        episode_count = len(files)
    max_steps = 50/0.1     # max s that we don't get buffer overrun, divided by time step
    reward = 0
    done = False
    step = 0
    epsilon = 1

    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)    # Create replay buffer

    env = Servo(live, files, tset, band)

    #Now load the weight
    print "Loading weights..."
    try:
        actor.model.load_weights("actormodel.h5")
        critic.model.load_weights("criticmodel.h5")
        actor.target_model.load_weights("actormodel.h5")
        critic.target_model.load_weights("criticmodel.h5")
        print "Weights loading succesful."
    except:
        print "Weights not found."


    print "Starting experiment..."
    try:
        for i in range(episode_count):
            print "Episode " + str(i)
            print "Replay Buffer " + str(buff.count())

            if not live:
                max_steps = env.open_file(files[i])

            ob = env.reset()
            s_t = np.hstack((ob.temperature, ob.Tset, ob.error))
            total_reward = 0.

            idx = 0
            while idx < max_steps:
                loss = 0
                epsilon -= 1.0 / EXPLORE

                if live:
                    #a_t = np.zeros([1,action_dim])
                    #noise_t = np.zeros([1,action_dim])

                    a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))

                    #noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0],  0.0 , 0.60, 0.30)
                    #noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0],  0.5 , 1.00, 0.10)
                    #noise_t = train_indicator * max(epsilon, 0) * OU.function(a_t_original,  0.5 , 1.00, 0.10)
                    noise_t = 0#train_indicator * max(epsilon, 0) * OU.function(a_t_original,  0.5 , 1.00, 0.10)
                    #noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1 , 1.00, 0.05)

                    # network outputs real number between [0,1] - we need an integer between [0,255]
                    a_t = a_t_original + noise_t
                    a_t = min( max(a_t,0), 1)
                    a_t = int(round(255*(a_t)))
                    a_t = [ a_t ]
                    #a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
                    #a_t[0][2] = a_t_original[0][2] + noise_t[0][2]
                else:
                    a_t = [env.choose_action()]

                # retrieve observation, reward, if-done, and info (??)
                # NOTE: DONEs are important in live!
                ob, r_t, done, info = env.step(a_t[0], ob)  # TODO - env.step function - including done and info!!
                # NOTE - info doesn't ever seem to be used??
                s_t1 = np.hstack((ob.temperature, ob.Tset, ob.error))

                idx += 1
                if idx == max_steps:
                    done = 1

                # EXPERIENCE REPLAY STARTS ---------------------------------------------

                # NOTE: if not live (pretraining), r_t, s_t1, done will be null/0 values
                buff.add(s_t, a_t[0], r_t, s_t1, done)      # Add to replay buffer

                # batch update
                batch = buff.getBatch(BATCH_SIZE)
                states = np.asarray([e[0] for e in batch])
                actions = np.asarray([e[1] for e in batch])
                rewards = np.asarray([e[2] for e in batch])
                new_states = np.asarray([e[3] for e in batch])
                dones = np.asarray([e[4] for e in batch])
                y_t = np.asarray([e[1] for e in batch])     # same as states?

                # TODO - understand what the target q values are and what the critic network is doing???

                #if live:
                target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])

                for k in range(len(batch)):
                    if dones[k]:
                        y_t[k] = rewards[k]
                    else:
                        y_t[k] = rewards[k] + GAMMA*target_q_values[k]

                # EXPERIENCE REPLAY ENDS -----------------------------------------------

                # actor is parametrized policy that defines how actions are selected
                # critic is a method that appraises each action the agent takes in the environment
                #   with some positive or negative scalar value; the critic estimates some
                #   quantity to produce its appraisal (q value - predicted cumulative return)
                # parameters of the actor are then updated wrt critic's appraisal

                if (train_indicator):
                    # NOTE: there are 2 steps involved in minimizing loss during training for a batch:
                    #   1 - compute gradients
                    #   2 - apply gradients
                    # during live training, we first train the critic model on the set of states, actions taken,
                    # and the assumed future reward; we thne use the actor model at its current state to predict
                    # the next action that should be taken, then have the critic model "criticize" it - i.e.,
                    # we compute the gradients of the critic model, given a set of states and the actions
                    # the actor model is proposing; we apply those gradients to the actor model, to train it
                    # then of course we train the actor and critic target networks
                    #
                    # during pre-training, we ignore the critic model entirely (TODO?) and simply train the actor
                    # on the pre-training data, which is manipulated to do what we think it should do, using its
                    # own gradients (rather than those of the critic model)
                    if live:
                        # update critic by minimizing loss
                        loss += critic.model.train_on_batch([states,actions], y_t)
                        # update actor policy using sampled policy gradient
                        a_for_grad = actor.model.predict(states)        # prediction of next action given state from actor model
                        grads = critic.gradients(states, a_for_grad)    # get sampled policy gradients from critic network, using predicted actions
                        actor.train(states, grads)                      # train using states and appropriate actions?
                        # update target network
                        actor.target_train()
                        critic.target_train()
                    else:
                        # update critic by minimizing loss
                        loss += critic.model.train_on_batch([states,actions], y_t)
                        # update actor policy using sampled policy gradient
                        a_for_grad = actor.model.predict(states)        # prediction of next action given state from actor model
                        grads = critic.gradients(states, a_for_grad)    # get sampled policy gradients from critic network, using predicted actions
                        actor.train(states, grads)                      # train using states and appropriate actions?
                        # update target network
                        actor.target_train()
                        critic.target_train()
                        '''
                        # train actor network on hypothesized ideal actions
                        actor.model.train_on_batch(states, actions)
                        # update target network
                        actor.target_train()
                        '''

                total_reward += r_t
                s_t = s_t1

                print 'Episode ' + str(i) + ', Step ' + str(step) + ', Temp ' + str(ob.temperature) + ', Tset ' + str(ob.Tset) + ', OUT ' + str(a_t[0]) + ', Reward ' + str(r_t) + ', Loss ' + str(loss)
                #print 'Episode ' + str(i) + ', Temp ' + str(s_t[0]) + ', Tset ' + str(s_t[1]) + ', Action ' + str(a_t[0])

                step += 1
                if done:
                    break

            #if np.mod(i, 3) == 0:
            # IN IF STARTS HERE --------------------------------------------------------
            if (train_indicator):
                print '\nSaving model...'
                actor.model.save_weights("actormodel.h5", overwrite=True)
                with open("actormodel.json", "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)
                print 'Actor saved'

                critic.model.save_weights("criticmodel.h5", overwrite=True)
                with open("criticmodel.json", "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)
                print 'Critic saved'

            # IN IF ENDS HERE ----------------------------------------------------------
            print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
            print 'Total Steps: ' + str(step) + '\n'


        # TESTING ------------------------------------------------------------------
        if not live:
            print "\nTESTING:\n"
            #data = np.load('data/servo_pid_450_7_b25.npz')
            data = np.load('data/servo_pid_450_7_b25.npz')
            df = pd.DataFrame(columns=['temperature','Tset','error','out'])
            df['temperature'] = data['temperature']
            df['Tset'] = data['Tset'][0]
            df['error'] = df['Tset'] - df['temperature']

            mean_t, std_t, mean_ts, std_ts, mean_et, std_et = env.get_norm_vars()

            df['temperature'] = (df['temperature'] - mean_t) / std_t
            df['Tset'] = (df['Tset'] - mean_ts) / std_ts
            df['error'] = (df['error'] - mean_et) / std_et

            df['out'] = data['out']
            # turn to integer DAC value between 0-255
            df['out'] = df['out'].apply(lambda x: round(max(min(x,255),0))).astype(int)
            # turn to real value between 0-1
            df['out'] = df['out'] / 255.


            '''
            # find out value necessary to maintain at temperature - estimate
            band = ((df[df['temperature'] > df['Tset']]['temperature'])).max() - df['Tset']
            out_val = int(round(((df[df['temperature'] > df['Tset']-band]['out'])).mean()))
            start_idx = ((df[df['temperature'] > df['Tset']-band])).index[0]
            df['out'][start_idx:] = out_val
            '''

            for _,row in df.iterrows():
                st = np.array([[row[0], row[1], row[2]]])
                #st = st.reshape(1,st.shape[1])
                act = np.array([row[-1]])
                pa = int(round(255*( actor.model.predict(st) )))
                #pa = actor.target_model.predict(st)
                ea = actor.model.evaluate(st, act)
                a = int(round(255*( act )))
                print 'Temp ' + str(row[0]*std_t + mean_t) + ', Tset ' + str(row[1]*std_ts + mean_ts) + ', Action ' + str(a) + ', P.Action ' + str(pa) + ', E.Action ' + str(ea)

            print "\n\nTARGET\n\n"
            for _,row in df.iterrows():
                st = np.array([[row[0], row[1], row[2]]])
                #st = st.reshape(1,st.shape[1])
                act = np.array([row[-1]])
                pa = int(round(255*( actor.target_model.predict(st) )))
                #pa = actor.target_model.predict(st)
                ea = actor.model.evaluate(st, act)
                a = int(round(255*( act )))
                print 'Temp ' + str(row[0]*std_t + mean_t) + ', Tset ' + str(row[1]*std_ts + mean_ts) + ', Action ' + str(a) + ', P.Action ' + str(pa) + ', E.Action ' + str(ea)

            # END TESTING -------------------------------------------------------------

    except KeyboardInterrupt:   # for manual early stopping
        print "\n SERVO INTERRUPTED - SHUTTING DOWN"
        env.end()

    # TODO: function to end servo
    env.end()  # This is for shutting down servo
    print 'Done.'

if __name__ == "__main__":
    live = 0        # if 1, train live on servo; if 0, train from old data
    train_indicator = 1
    if live:
        files = None
        tset = 450
        band = 1
    else:
        files = [f for f in listdir('data/')]
        tset = 0
        band = 0
    playGame(train_indicator=train_indicator, live=live, files=files, tset=tset, band=band)
