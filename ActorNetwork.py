from keras.initializations import normal
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K

HIDDEN1_UNITS = 3
HIDDEN2_UNITS = 3*HIDDEN1_UNITS

class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        #Now create the model
        self.model , self.weights, self.state = self.create_actor_network(state_size, action_size)
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size)
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())#self.sess.run(tf.initialize_all_variables())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,

            self.action_gradient: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, state_size,action_dim):
        print "Building actor network..."
        S = Input(shape=[state_size])
        h0 = Dense(HIDDEN1_UNITS, activation='relu', init='glorot_uniform')(S)
        h1 = Dense(HIDDEN2_UNITS, activation='sigmoid',init='glorot_uniform')(h0)
        #h1 = Dense(HIDDEN2_UNITS, activation='relu', init='glorot_uniform')(h0)
        # in output layer, we use the normal initialization with mu=0, sigma=1e-4 to ensure
        # the initial outputs for the policy are near 0
        DAC = Dense(1,activation='sigmoid', init='glorot_uniform')(h1)
        #DAC = Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)
        model = Model(input=S,output=DAC)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        return model, model.trainable_weights, S
