import numpy as np
# from os import path
import numpy as np
import copy
import collections as col
import os
import time
import pandas as pd

from ws339_18 import Arduino

'''
Proposed optimization objectives for our reward function:
- maximize velocity
- minimize oscillation / maximize stability
- maximize (temp) distance covered

Note that we want to maximize cumulative reward over a total run, as well as on
every individual step


fn(1): MAXIMIZE ERROR IMPROVEMENT:

Case: increasing toward set temperature:
(t_0 - Tset_0)/band         t_0 - Tset_0        -40         8
-------------------    =    ------------    =   ---     =  ---  = 1.143
(t_1 - Tset_1)/band         t_1 - Tset_1        -35         7

(old_temp-new_temp)/old_temp
getting better:
(40-35)/40 = 5/40 = 1/8
(40-30)/40 = 10/40 = 1/4
(-40--30)/-40 = -10/-40 = 1/4
(-40--35)/-40 = -5/-40 = 1/8
(15-10)/15 = 5/15 = 1/3
getting worse:
(35-40)/35 = -5/35 = -1/7
(30-40)/30 = -10/30 = -1/3
(-30--40)/-30 = 10/-30 = -1/3
(-35--40)/-35 = 5/-35 = -1/7
(10-15)/10 = -5/10 = -1/2

Case: decreasing toward set temperature:
(t_0 - Tset_0)/band         t_0 - Tset_0        40          8
-------------------    =    ------------    =   ---     =  ---  = 1.143
(t_1 - Tset_1)/band         t_1 - Tset_1        35          7

SO this works regardless of whether we're increasing or decreasing

(t_0 - Tset_0)/band         t_0 - Tset_0        -40         8
-------------------    =    ------------    =   ---     =  ---  = 1.333
(t_1 - Tset_1)/band         t_1 - Tset_1        -30         6

AND it's directly proportional to the amount of improvement.

(t_0 - Tset_0)/band         t_0 - Tset_0        -30         3
-------------------    =    ------------    =   ---     =  ---  = 1.500
(t_1 - Tset_1)/band         t_1 - Tset_1        -20         2

Also note that the reward increase is higher the closer we get - this may become a
problem and something we have to address.

BUT, if there is some overshoot in either direction, this measure is inversely
proportional, so if it's negative we take the absolute value of its inverse:

Case: increasing toward set temperature, overshoot:
e_temp_0        (t_0 - Tset_0)/band         t_0 - Tset_0        -5         -1               -2
--------    =   -------------------    =    ------------    =   ---     =  ---  = -0.5  ->  --- = -2.000
e_temp_1        (t_1 - Tset_1)/band         t_1 - Tset_1        10          2                1

Case: decreasing toward set temperature, overshoot:
e_temp_0        (t_0 - Tset_0)/band         t_0 - Tset_0        10         -2               -1
--------    =   -------------------    =    ------------    =   ---     =  ---  = -2    ->  --- = -0.500
e_temp_1        (t_1 - Tset_1)/band         t_1 - Tset_1        -5          1                2

SO, if the measure is negative, we take the absolute value of its inverse. We can
also taking its absolute value, unless we really want to disproportionately penalize
overshoot. (We can test using both schemes - taking the absolute value, or not).

However this doesn't account for entering or leaving the band area - so if either the
numerator or denominator is within the band, we can divide by the band; as an example,
using a band of 5:

Case: increasing toward set temperature, overshoot:
e_temp_0        (t_0 - Tset_0)/band         t_0 - Tset_0       -5/5        -1              -10
--------    =   -------------------    =    ------------    =   ---     =  ---  = -0.1  ->  --- = -10.000
e_temp_1        (t_1 - Tset_1)/band         t_1 - Tset_1        10         10                1

Case: decreasing toward set temperature, overshoot:
e_temp_0        (t_0 - Tset_0)/band         t_0 - Tset_0        10         -10              -1
--------    =   -------------------    =    ------------    =   ---     =  ---  = -10   ->  --- = -0.1
e_temp_1        (t_1 - Tset_1)/band         t_1 - Tset_1       -5/5         1               10

This looks good, and can be what we try initally for this particular measure.



fn(2): MINIMIZE ERROR RELATIVE TO BAND:

The greater the band, the less we are concerned about fluctuations. The smaller the
band, the more we are concered.

So we can simply use the absolute value of the recorded error.

So we can do:

R_t = fn(1) - fn(2)     OR
R_t = fn(1)             OR
R_t = fn(2)

and also add other functions to minimize/maximize

But it doesn't seem like the reward is normalized in the TORCS application?

'''

# NOTE: This program is setup to be an interface between the servo arduino sketch
#       and the python ML algorithm



class Servo:
    #terminal_judge_start = 100  # If after 100th timestep still no progress, terminated
    #termination_limit_progress = 1  # [K/min], episode terminates if temperature increase is less than this limit
    #default_speed = 50
    default_out = 127

    initial_reset = True    # TODO

    def __init__(self, live, files, tset, band):

        if live:
            print "Live!!!!"
            self.tset = tset
            self.band = band
            a = Arduino()
            a.send("SET dt 100") # period in milliseconds
            a.send("START")
            self.a = a

        else:
            dat = pd.DataFrame(columns=['temperature','Tset','out'])
            for fname in files:
                data = np.load('data/' + fname)
                df = pd.DataFrame(columns=['temperature','Tset','out'])
                df['temperature'] = data['temperature']
                df['Tset'] = data['Tset'][0]
                df['error'] = df['Tset'] - df['temperature']
                # unnecessary
                #df['out'] = data['out']
                #df['out'] = df['out'].apply(lambda x: round(max(min(x,255),0))).astype(int)
                if len(dat) == 0:
                    dat = df
                else:
                    dat = dat.append(df)

            self.mean_t = dat['temperature'].mean()
            self.mean_ts = dat['Tset'].mean()
            self.mean_et = dat['error'].mean()
            self.std_t = dat['temperature'].std()
            self.std_ts = dat['Tset'].std()
            self.std_et = dat['error'].std()
            print self.mean_t
            print self.mean_ts
            print self.mean_et
            print self.std_t
            print self.std_ts
            print self.std_et

            del dat
            del data
            del df

        self.initial_run = True
        self.live = live
        self.current_data = None
        self.d_idx = 0
        self.current_action = 0

    def step(self, u, ob):
        self.time_step += 1

        if self.live:
            # apply action
            self.a.send("SET out " + str(u))

        err_0 = self.observation.error
        self.observation = self.make_observation()
        ob = self.get_obs()
        if ob.temperature == -1 and ob.Tset == -1 and ob.e_temperature == -1:
            done = 1
            reward = 0
        else:
            done = 0

            err_1 = ob.error
            de = err_1 - err_0

            #reward = (temp_1-temp_0)/temp_1*100
            reward = 0
            big, med, small = 1, .67, .33
            if err_1 < -.1:
                if de < 0:
                    reward = -1 * big
                elif de == 0:
                    reward = 0#-1 * small
                elif de > 0:
                    if self.live:
                        if abs(err_1) > self.band:
                            if de > .75:           # arbitrary threshold
                                reward = 1 * big
                            else:
                                reward = 1 * med
                        else:
                            reward = 1 * big
                    else:
                        if de > .75:           # arbitrary threshold
                            reward = 1 * big
                        else:
                            reward = 1 * med
            elif err_1 < .1 and err_1 > -.1:
                if de < 0:
                    reward = 1 * med
                elif de == 0:
                    reward = 1 * small
                elif de > 0:
                    reward = 1 * med
            elif err_1 > .1:
                if de < 0:
                    if de < -.75:           # arbitrary threshold
                        reward = 1 * big
                    else:
                        reward = 1 * med
                elif de == 0:
                    reward = 0#-1 * small
                elif de > 0:
                    reward = -1 * big

        return ob, reward, done, {}

    def reset(self, relaunch=False):
        print "Resetting"
        self.time_step = 0  # TODO: remove?

        if self.live:
            self.reset_servo()
        self.observation = self.make_observation()
        self.initial_reset = False
        return self.get_obs()

    def end(self):
        if self.live:
            print "Ending servo"
            self.a.send("SET out 0")
            self.a.send("STOP")
        return

    def get_obs(self):
        return self.observation

    def reset_servo(self):
        # TODO - redo
        if self.live:
            print "Resetting servo"
            self.a.send("SET out 0")
            self.a.send("STOP")
            self.a.send("SET dt 100")
            self.a.send("START")
            print "SERVO RELAUNCHED"

    def make_observation(self):
        names = ['temperature','Tset','error']
        Observation = col.namedtuple('Observation', names)

        if self.live:
            while True:
                line = self.a.getResp()
                if line[1:2] == ':':
                    break
            parts = line.split(':')
            if parts[0] == 'T':
                temp = float(parts[1])
            tset = self.tset
            etemp = tset - temp
            return Observation(temperature=temp, Tset=tset, error=etemp)
        else:
            try:
                _,dat = self.current_data.next()
                temp = np.array(dat['temperature'], dtype=np.float32)
                tset = np.array(dat['Tset'], dtype=np.float32)
                etemp = np.array(dat['error'], dtype=np.float32)
                self.current_action = int(round(255*dat['out']))
                self.d_idx += 1
            except StopIteration:   # in case iterator runs out
                print "Iterator ran out!"
                temp, tset, etemp = -1, -1, -1
            return Observation(temperature=temp, Tset=tset, error=etemp)

    def get_norm_vars(self):
        return (self.mean_t, self.std_t, self.mean_ts, self.std_ts, self.mean_et, self.std_et)

    def open_file(self, fname):
        print "Opening file \'" + fname + "\'"
        data = np.load('data/' + fname)
        df = pd.DataFrame(columns=['temperature','Tset','error','out'])
        df['temperature'] = data['temperature']
        df['Tset'] = data['Tset'][0]
        df['error'] = df['Tset'] - df['temperature']

        df['temperature'] = (df['temperature'] - self.mean_t) / self.std_t
        df['Tset'] = (df['Tset'] - self.mean_ts) / self.std_ts
        df['error'] = (df['error'] - self.mean_et) / self.std_et

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

        self.current_data = df.iterrows()
        self.d_idx = 0
        return len(df)-1

    def choose_action(self):
        if self.live:
            print "\nGet your shit together!\n"
            return -1
        else:
            return self.current_action
