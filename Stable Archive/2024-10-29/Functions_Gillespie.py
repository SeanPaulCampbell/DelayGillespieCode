#!/usr/bin/env python
import Classes_Gillespie as Classy
import numpy as np
import heapq as q
import numba as nb
import time as t

'''
The current version of this code requires editing of this file in order to simulate different systems.
Lines throughout this file with the ~%~ symbol denote lines that must be altered.
'''


def gillespie(reactions_list, stop_time, initial_state_vector, initial_queue=[], system_size=1):
    service_queue = initial_queue
    [state_vector, current_time, time_series] = initialize(initial_state_vector, initial_queue) # ~%~
    current_state = False
    propensity_table = [] # ~%~
    for num in range(128): # ~%~
        propensity_table.append(calculate_propensities(np.array([num]), reactions_list, system_size)) # ~%~
    while current_time < stop_time:
        cumulative_propensities = propensity_table[state_vector[0]] # ~%~
        next_event_time = draw_next_event_time(current_time, cumulative_propensities[-1])
        if reaction_will_complete(service_queue, next_event_time):
            next_reaction = q.heappop(service_queue)
            state_vector += next_reaction[1]
            current_time = next_reaction[0]
            if transitioned(state_vector, current_state, system_size, 8, 30):
                current_state = not current_state
                time_series.append({"time" : current_time, "state" : state_vector/system_size, "queue" : [reaction[0] for reaction in service_queue], "transition" : True})
            else:
                time_series.append({"time" : current_time, "state" : state_vector/system_size, "queue" : [reaction[0] for reaction in service_queue], "transition" : False}) # ~%~
            continue
        current_time = next_event_time
        next_reaction = reactions_list[choose_reaction(cumulative_propensities)]
        processing_time = next_reaction.distribution()
        if processing_time == 0:
            state_vector = state_vector + next_reaction.change_vec
            if transitioned(state_vector, current_state, system_size, 8, 30):
                current_state = not current_state
                time_series.append({"time" : current_time, "state" : state_vector/system_size, "queue" : [reaction[0] for reaction in service_queue], "transition" : True})
            else:
                time_series.append({"time" : current_time, "state" : state_vector/system_size, "queue" : [reaction[0] for reaction in service_queue], "transition" : False}) # ~%~
        else:
            q.heappush(service_queue, (current_time + processing_time, next_reaction.change_vec))
    return time_series


def gillespie_transitions(reactions_list, initial_state_vector, low_threshold, high_threshold, stop_number, initial_queue=[], system_size=1):
    service_queue = initial_queue
    [state_vector, current_state, current_time, time_series] = initialize_transitions(initial_state_vector, low_threshold, high_threshold)
    recorded_transitions = 0
    propensity_table = [] # ~%~
    for num in range(128): # ~%~
        propensity_table.append(calculate_propensities(np.array([num]), reactions_list, system_size)) # ~%~
    while recorded_transitions < stop_number:
        cumulative_propensities = propensity_table[state_vector[0]] # ~%~
        next_event_time = draw_next_event_time(current_time, cumulative_propensities[-1])
        if reaction_will_complete(service_queue, next_event_time):
            next_reaction = q.heappop(service_queue)
            state_vector += next_reaction[1]
            current_time = next_reaction[0]
            if transitioned(state_vector, current_state, system_size, low_threshold, high_threshold): # ~%~
                current_state = not current_state
                recorded_transitions += 1
                time_series.append({"time" : current_time, "high_low" : current_state}) # ~%~
            continue
        current_time = next_event_time
        next_reaction = reactions_list[choose_reaction(cumulative_propensities)]
        processing_time = next_reaction.distribution()
        if processing_time == 0:
            state_vector = state_vector + next_reaction.change_vec
            if transitioned(state_vector, current_state, system_size, low_threshold, high_threshold): # ~%~
                current_state = not current_state
                recorded_transitions += 1
                time_series.append({"time" : current_time, "high_low" : current_state}) # ~%~
        else:
            q.heappush(service_queue, (current_time + processing_time, next_reaction.change_vec))
    return time_series


def initialize(initial_state_vector, initial_queue):
    state_vector = initial_state_vector
    current_time = 0
    time_series = [{"time" : 0, "state" : state_vector, "queue" : [reaction[0] for reaction in initial_queue], "transition" : False}] # ~%~
    return [state_vector, current_time, time_series]


def initialize_transitions(initial_state_vector, low_threshold, high_threshold):
    state_vector = initial_state_vector
    current_state = (state_vector > (low_threshold + high_threshold) / 2)
    current_time = 0
    time_series = [{"time" : 0, "high_low" : current_state}] # ~%~
    return [state_vector, current_state, current_time, time_series]


def calculate_propensities(x, reactions_list, system_size=1):
    propensities = np.zeros(np.shape(reactions_list), dtype=np.float32)
    for index in range(np.size(reactions_list)):
        propensities[index] = np.float32(reactions_list[index].propensity(x/system_size)*system_size)
    return np.cumsum(propensities)


def reaction_will_complete(queue, next_event_time):
    if queue:
        if next_event_time > queue[0][0]:
            return True
    return False

'''
The transitioned function must be edited for different systems.
'''

def transitioned(state, side, size, low, high): # ~%~
    if side:
        return (state/size <= low)
    else:
        return (state/size >= high)


@nb.jit("float64(float64, float32)")
def draw_next_event_time(current_time, summed_propensities):
    return current_time + np.random.exponential(scale=(1 / summed_propensities))


@nb.jit("int16(float32[:])")
def choose_reaction(cumulative_propensities):
    u = np.random.uniform()
    next_reaction_index = min(
        np.where(cumulative_propensities >= cumulative_propensities[-1] * u)[0])
    return next_reaction_index


def recursive_list_formation(parameter_ranges, long_list):
    if len(parameter_ranges) != 0:
        par_range = len(parameter_ranges[0])
        current_length = len(long_list)
        long_list = long_list * par_range
        for index1 in range(par_range):
            for index2 in range(index1 * current_length, (index1 + 1) * current_length):
                long_list[index2] = long_list[index2] + [parameter_ranges[0][index1]]
        return recursive_list_formation(parameter_ranges[1:], long_list)
    else:
        return long_list


def list_for_parallelization(parameter_ranges):
    long_list = []
    for index in range(len(parameter_ranges[0])):
        long_list.append([parameter_ranges[0][index]])
    return recursive_list_formation(parameter_ranges[1:], long_list)
