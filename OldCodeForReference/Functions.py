import Classes as Classy
import numpy as np
import pandas as pd


''' main Distributed Delay Stochastic Simulation Algorithm 
    NOTE: There is no checking for negative values in this version.'''


def gillespie(reactions_list, stop_time, initial_state_vector):
    [state_vector, current_time, service_queue, time_series] = initialize(initial_state_vector)
    while current_time < stop_time:
        cumulative_propensities = calculate_propensities(state_vector, reactions_list)
        next_event_time = draw_next_event_time(current_time, cumulative_propensities)
        if reaction_will_complete(service_queue, next_event_time):
            [state_vector, current_time] = trigger_next_reaction(service_queue, state_vector)
            time_series = write_to_time_series(time_series, current_time, state_vector)
            continue
        current_time = next_event_time
        next_reaction = choose_reaction(cumulative_propensities, reactions_list)
        processing_time = next_reaction.distribution()
        if processing_time == 0:
            state_vector = state_vector + next_reaction.change_vec
            time_series = write_to_time_series(time_series, current_time, state_vector)
        else:
            add_reaction(service_queue, current_time + processing_time, next_reaction)
    return dataframe_to_numpyarray(time_series)


def initialize(initial_state_vector):
    state_vector = initial_state_vector
    current_time = 0
    service_queue = []
    time_series = pd.DataFrame([[current_time, state_vector]], columns=['time', 'state'])
    return [state_vector, current_time, service_queue, time_series]


''' calculate_propensities creates an array with the cumulative sum of the propensity functions. '''


def calculate_propensities(x, reactions_list):
    propensities = np.zeros(np.shape(reactions_list))
    for index in range(np.size(reactions_list)):
        propensities[index] = reactions_list[index].propensity(x)
    return np.cumsum(propensities)


def reaction_will_complete(queue, next_event_time):
    if len(queue) > 0:
        if next_event_time > queue[0].comp_time:
            return True
    return False


def draw_next_event_time(current_time, cumulative_propensities):
    return current_time + np.random.exponential(scale=(1 / cumulative_propensities[-1]))


''' choose_reaction rolls a biased die to determine which reaction will take place or be scheduled next.
    Simple as it gets, as optimal as it gets... I think. '''


def choose_reaction(cumulative_propensities, reactions_list):
    u = np.random.uniform()
    next_reaction_index = min(np.where(cumulative_propensities > cumulative_propensities[-1] * u)[0])
    return reactions_list[next_reaction_index]


''' add_reaction, while not a pure function, does what it is supposed to,
    inserts into the queue a new delayed reaction sorted by completion time. '''


def add_reaction(queue, schedule_time, next_reaction):
    reaction = Classy.ScheduleChange(schedule_time, next_reaction.change_vec)
    if len(queue) == 0:
        return queue.append(reaction)
    else:
        for k in range(len(queue)):
            if reaction.comp_time < queue[k].comp_time:
                return queue.insert(k, reaction)
    return queue.append(reaction)


''' trigger_next_reaction has the side effect of removing the first entry of the queue it was passed. '''


def trigger_next_reaction(queue, state_vector):
    next_reaction = queue.pop(0)
    state_vector = state_vector + next_reaction.change_vec
    current_time = next_reaction.comp_time
    return [state_vector, current_time]


def write_to_time_series(time_series, current_time, state_vector):
    return time_series.append(pd.DataFrame([[current_time, state_vector]],
                                           columns=['time', 'state']), ignore_index=True)


''' dataframe_to_numpyarray allows us to use the more efficient DataFrame class to record time series
    and then convert that object back into a usable numpy array. '''


def dataframe_to_numpyarray(framed_data):
    timestamps = np.array(framed_data[['time']])
    states = framed_data[['state']]
    arrayed_data = np.zeros([max(np.shape(timestamps)), np.shape(states.iloc[0, 0])[0] + 1])
    arrayed_data[:, 0] = timestamps.transpose()
    for index in range(max(np.shape(timestamps))):
        arrayed_data[index, 1:] = states.iloc[index, 0]
    return arrayed_data
