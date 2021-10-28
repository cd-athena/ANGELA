import Utils
import User

#######################################
# Functions related to ABR algorithms #
#######################################

harmonic_mean_weights = []
harmonic_mean_rate = []


# For harmonic mean in SARA
def init(nusers):
    for x in range(nusers):
        harmonic_mean_weights.append([])
        harmonic_mean_rate.append([])


# https://en.wikipedia.org/wiki/Harmonic_mean#Weighted_harmonic_mean
def set_harmonic_mean_values(userId, segment_size, data_rate):
    harmonic_mean_weights[userId].append(segment_size)  # KB
    harmonic_mean_rate[userId].append(data_rate * 1000 / 8)  # Mbps to KB/s


def get_harmonic_mean(userId):
    w = 0
    s = 0
    for x in range(len(harmonic_mean_weights[userId])):
        s = s + float(harmonic_mean_weights[userId][x]) / float(harmonic_mean_rate[userId][x])
        w = w + float(harmonic_mean_weights[userId][x])

    H = w / s
    return H


# SARA ABR Algorithm
def SARA(userId, n_qualities, segment_duration, current_quality_index, current_buffer_sec, next_segment_number):
    bitrate_index_set = list(range(0, n_qualities))
    list_segment_size = [0] * len(bitrate_index_set)
    for x in bitrate_index_set:
        list_segment_size[x] = float(Utils.getSegmentSize(userId, next_segment_number, x, segment_duration))  # KB

    r = bitrate_index_set  # Set of bitrate index [0, 1, 2, 3 ...]
    r_min = r[0]
    r_max = r[bitrate_index_set[len(bitrate_index_set) - 1]]
    r_curr = current_quality_index  # current quality index

    I = 2 * segment_duration  # Buffer constant
    Balpha = 7 * segment_duration  # Buffer constant
    Bbeta = 14 * segment_duration  # Buffer constant
    B_curr = current_buffer_sec  # Current buffer occupancy in seconds
    W_n = list_segment_size  # KB List of the sizes of the segments for bitrates
    H_n = get_harmonic_mean(userId)  # Weighted Harmonic mean download rate for the first n segments
    next_bitrate = 0  # result of the algorithm, index
    wait_time = 0  # The wait time before downloading the next segment

    # print("SARA decision: ")
    # print("Current buffer(s): " + str(B_curr) + " Current throughput (Mbps): " + str(int(throughput_Client_Edge))
    #      + " Bitrate Q0: " + str(list_segment_size[0])
    #      + " Bitrate Q1: " + str(list_segment_size[1])
    #      + " Bitrate Q2: " + str(list_segment_size[2])
    #      + " Bitrate Q3: " + str(list_segment_size[3]))

    # Initialization:
    if B_curr <= I:  # Fast start
        next_bitrate = r_min
        sara_phase = 0
    else:

        # If we are at maximum quality
        if r_curr >= r_max:
            r_curr_next = r_max
        else:
            r_curr_next = r_curr + 1

        if (W_n[r_curr_next] / H_n) > (B_curr - I):
            sara_phase = 1
            for i in reversed(r):
                if i <= r_curr and (W_n[i] / H_n) <= (B_curr - I):
                    break
            next_bitrate = i
            wait_time = 0
        elif B_curr <= Balpha:  # Additive increase
            sara_phase = 2

            # If we are at maximum quality
            if r_curr_next >= r_max:
                r_curr_next = r_max
            else:
                r_curr_next = r_curr + 1

            if (W_n[r_curr_next] / H_n) < (B_curr - I):
                next_bitrate = r_curr + 1  # Increase by one level  # Here it could request more than maximum quality... fixed at the end
            else:
                next_bitrate = r_curr
            wait_time = 0
        elif B_curr <= Bbeta:  # Aggressive switching
            sara_phase = 3
            for i in reversed(r):
                if i >= r_curr and (W_n[i] / H_n) <= (B_curr - I):
                    break
            next_bitrate = i
            wait_time = 0
        elif B_curr > Bbeta:  # Delayed download
            sara_phase = 4
            for i in reversed(r):
                if i >= r_curr and (W_n[i] / H_n) <= (B_curr - Balpha):
                    break
            next_bitrate = i
            wait_time = B_curr - Bbeta
        else:
            sara_phase = 5
            next_bitrate = r_curr
            wait_time = 0

    if next_bitrate >= r_max:
        next_bitrate = r_max
    # print("SARA request quality index: " + str(next_bitrate) + " SARA phase: " + str(sara_phase))

    # return [next_bitrate, wait_time]
    return next_bitrate


# Linear function for BBA
def buffer_to_rate(current_buffer, rmax, rmin, r, c):
    result = (rmax - rmin) / c * current_buffer + (rmin - r * (rmax - rmin) / c)
    return int(result)


# "A Buffer-Based Approach to Rate Adaptation: Evidence from a Large Video Streaming Service"
def BBA(current_quality_index, current_buffer, n_qualities):
    buffer_size = 240  # seconds
    reservoir = 90  # seconds
    cushion = 126  # seconds

    rate_prev = current_quality_index
    Rmax = n_qualities - 1  # index

    if rate_prev == Rmax:
        rate_plus = Rmax
    else:
        rate_plus = rate_prev + 1

    if rate_prev == 0:
        rate_minus = 0
    else:
        rate_minus = rate_prev - 1

    f_b = buffer_to_rate(current_buffer, Rmax, 0, reservoir, cushion)

    if current_buffer <= reservoir:
        rate_next = 0
    elif current_buffer >= (reservoir + cushion):
        rate_next = Rmax
    elif f_b >= rate_plus:
        rate_next = f_b - 1
    elif f_b <= rate_minus:
        rate_next = f_b + 1
    else:
        rate_next = rate_prev

    return rate_next


# Throughput based algorithm from the paper entitled: "Adaptation method for video streaming over HTTP/2"
def TBA(userId, n_qualities, estimated_throughput):
    next_segment_number = User.get_segment_number(userId)
    segment_duration = User.get_segment_duration(userId)
    next_quality = 0
    bitrate_index_set = list(range(0, n_qualities))
    list_segment_bitrate = [0] * len(bitrate_index_set)
    segment_duration_in_s = Utils.mapIndexToSegDuration(segment_duration)
    for x in bitrate_index_set:
        list_segment_bitrate[x] = float(
            Utils.getSegmentSize(userId, next_segment_number, x, segment_duration)) * 8 / segment_duration_in_s  # kbps

    estimated_throughput_in_kbps = estimated_throughput * 1000  # Mbps to kbps

    safety_margin = 0.2  # They use this value in the paper
    for x in range(n_qualities):
        if list_segment_bitrate[x] <= (estimated_throughput_in_kbps * (1 - safety_margin)):
            next_quality = x

    return next_quality
