#!/usr/bin/env python
from numpy import random
import numpy as np

''' DO NOT EDIT THIS SECTION '''


class Reaction:
    propensities_list = ['mobius_propensity', 'decreasing_hill_propensity',
                         'increasing_hill_propensity', 'mobius_sum_propensity',
                         'dual_feedback_decreasing_hill_propensity',
                         'dual_feedback_increasing_hill_propensity', 'decreasing_heviside_propensity',
                         'weird_toy_propensity', 'switching_decreasing_hill_propensity',
                         'switching_increasing_hill_propensity', 'boolean_propensity']
    distributions_list = ['gamma_distribution', 'trivial_distribution',
                          'bernoulli_distribution', 'uniform_distribution']

    def __init__(self, state_change_vector, parts_of_state_vector,
                 propensity_id, propensity_params,
                 distribution_id, distribution_params):
        self.change_vec = state_change_vector
        self.parts_of_vec = parts_of_state_vector
        if type(propensity_id) == str:
            self.prop_id = Reaction.propensities_list.index(propensity_id)
        else:
            self.prop_id = propensity_id
        self.prop_par = propensity_params
        if type(distribution_id) == str:
            self.dist_id = Reaction.distributions_list.index(distribution_id)
        else:
            self.dist_id = distribution_id
        self.dist_par = distribution_params

    ''' DO NOT EDIT THIS SECTION '''

    def propensity(self, x, queue):
        return getattr(self, Reaction.propensities_list[self.prop_id])(x, queue)

    def distribution(self):
        return getattr(self, Reaction.distributions_list[self.dist_id])()

    ''' Propensities start here.
    After adding a propensity function to the list of definitions, 
    append the name to props_list. '''

    def mobius_propensity(self, state_vector, queue):
        """ For a constant function f(x) = c, assign the vector [c,0,1,0]
            For a linear map f(x) = a * x,    assign the vector [0,a,1,0] """
        x = state_vector[self.parts_of_vec]
        return (self.prop_par[0] + self.prop_par[1] * x) / (self.prop_par[2] + self.prop_par[3] * x)

    def decreasing_hill_propensity(self, state_vector, queue):
        x = state_vector[self.parts_of_vec]
        scale = self.prop_par[0]
        threshold = self.prop_par[1]
        exponent = self.prop_par[2]
        if self.prop_par[3] == 0:
            return scale * (threshold / (x + threshold)) ** exponent
        else:
            exp_thresh = threshold ** exponent
            return scale * exp_thresh / (x ** exponent + exp_thresh)

    def increasing_hill_propensity(self, state_vector, queue):
        x = state_vector[self.parts_of_vec]
        scale = self.prop_par[0]
        threshold = self.prop_par[1]
        exponent = self.prop_par[2]
        if self.prop_par[3] == 0:
            return scale * (x / (x + threshold)) ** exponent
        else:
            exp_x = x ** exponent
            return scale * exp_x / (exp_x + threshold ** exponent)

    def mobius_sum_propensity(self, state_vector, queue):
        total_species = np.sum(state_vector)
        x = state_vector[self.parts_of_vec]
        return (self.prop_par[0] + self.prop_par[1] * x) / (self.prop_par[2] + self.prop_par[3] * total_species)

    def dual_feedback_decreasing_hill_propensity(self, state_vector, queue):
        scale = self.prop_par[0]
        threshold0 = self.prop_par[1]
        threshold1 = self.prop_par[2]
        factor = self.prop_par[3]
        exponent = self.prop_par[4]
        return scale * (threshold0 / (threshold0 + state_vector[self.parts_of_vec])) ** exponent * \
            (threshold1 / factor + state_vector[(self.parts_of_vec + 1) % 2]) / \
            (threshold1 + state_vector[(self.parts_of_vec + 1) % 2])

    def dual_feedback_increasing_hill_propensity(self, state_vector, queue):
        scale = self.prop_par[0]
        threshold0 = self.prop_par[1]
        threshold1 = self.prop_par[2]
        factor = self.prop_par[3]
        exponent = self.prop_par[4]
        return scale * (threshold0 / (threshold0 + state_vector[(self.parts_of_vec + 1) % 2])) ** exponent * \
            (threshold1 / factor + state_vector[self.parts_of_vec]) / \
            (threshold1 + state_vector[self.parts_of_vec])

    def decreasing_heviside_propensity(self, state_vector, queue):
        scale = self.prop_par[0]
        threshold = self.prop_par[1]
        return scale*(state_vector[self.parts_of_vec] < threshold)

    def weird_toy_propensity(self, state_vector, queue):
        state = state_vector[self.parts_of_vec]
        scale = self.prop_par[0]
        threshold = self.prop_par[1]
        if not queue:
            self.prop_par[2] = False
        flag = self.prop_par[2]
        if flag == True:
            return 0
        elif state >= threshold:
            self.prop_par[2] = True
            return 0
        return scale

    def switching_decreasing_hill_propensity(self, state_vector, queue):
        switch = state_vector[self.parts_of_vec[0]]
        if switch == self.prop_par[0]:
            return 0
        else:
            x = state_vector[self.parts_of_vec[1]]
            scale = self.prop_par[1]
            threshold = self.prop_par[2]
            exponent = self.prop_par[3]
            if self.prop_par[4] == 0:
                return scale * ( threshold / ( x + threshold ) ) ** exponent
            else:
                exp_thresh = threshold ** exponent
                return scale * exp_thresh / ( x ** exponent + exp_thresh)

    def switching_increasing_hill_propensity(self, state_vector, queue):
        switch = state_vector[self.parts_of_vec[0]]
        if switch == self.prop_par[0]:
            return 0
        else:
            x = state_vector[self.parts_of_vec[1]]
            scale = self.prop_par[1]
            threshold = self.prop_par[2]
            exponent = self.prop_par[3]
            if self.prop_par[4] == 0:
                return scale * ( x / (x + threshold)) ** exponent
            else:
                exp_x = x ** exponent
                return scale * exp_x / (exp_x + threshold ** exponent)

    def boolean_propensity(self, state_vector, queue):
        if state_vector[self.parts_of_vec] == self.prop_par[0]:
            return self.prop_par[1]
        else:
            return 0

    ''' Distributions start here.
    After adding a distribution to the list of definitions,
    append the name to distr_list. '''

    def gamma_distribution(self):
        if self.dist_par[1] == 0:
            return self.trivial_distribution()
        mean = self.dist_par[0]
        stdev = self.dist_par[1]
        return random.gamma(shape=(mean / stdev) ** 2, scale=stdev ** 2 / mean)

    def trivial_distribution(self):
        mean = self.dist_par[0]
        return mean

    def bernoulli_distribution(self):
        mean = self.dist_par[0]
        stdev = self.dist_par[1]
        return mean - 1 + 2 * stdev * random.randint(2)

    def uniform_distribution(self):
        mean = self.dist_par[0]
        stdev = self.dist_par[1]
        return random.uniform(low=(mean-np.sqrt(3)*stdev), high=(mean+np.sqrt(3)*stdev))


class ScheduleChange:

    def __init__(self, completion_time, change_vector):
        self.comp_time = completion_time
        self.change_vec = change_vector


