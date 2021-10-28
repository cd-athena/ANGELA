#!/usr/bin/python
# -*- coding: latin-1 -*-

import User
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot
from statistics import mean
import User

historical_bitrates_requested = []
historical_radio_throughput = []
historical_predicted_throughput = []
historical_qindex_requested = []
historical_QoE_score = []
historical_buffer = []

# For estimate the throughput
req_time = []
rec_time = []
req_size = []
n_packets = 0
accumulative_E2E_latency = 0  # estimate latency

bitrate_ladder = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 900, 1200, 1500, 2000, 2500, 3000, 4000, 5000, 6000,
                  8000]


# For harmonic mean in SARA
def init(nusers):
    global estimated_throughput
    global req_size
    global req_time
    global rec_time
    estimated_throughput = [0.1] * nusers
    req_time = [0] * nusers
    rec_time = [0] * nusers
    req_size = [0] * nusers
    for x in range(nusers):
        historical_bitrates_requested.append([])  # Plot purposes # kbps
        historical_qindex_requested.append([])
        historical_QoE_score.append([])
        historical_radio_throughput.append([])  # Plot purposes
        historical_predicted_throughput.append([])
        historical_buffer.append([])


def get_n_segments_requested(userId):
    return len(historical_bitrates_requested[userId])


def set_req_time(userId, value):
    req_time[userId] = value


def set_rec_time(userId, value):
    rec_time[userId] = value


def set_req_size(userId, value):
    req_size[userId] = value


def get_req_size(userId):
    return req_size[userId]


def get_rec_minus_req(userId):
    t = rec_time[userId] - req_time[userId]
    return t


def get_estimated_throughput(userId):
    return estimated_throughput[userId]


def get_predicted_throughput(userId, future_throughput_Client_Edge, throughput_Client_Edge, est_throughput):
    # Future throughput prediction, we predict future radio throughput and adjust the estimated throughput
    # We calculate the time that would have taken to sent the last segment with future network conditions,
    # Then use that factor to modify the estimated throughput

    if (throughput_Client_Edge - future_throughput_Client_Edge) != 0:
        # Time that would be added for that segment with future network conditions
        added_time = (get_req_size(userId) * 0.008) / (throughput_Client_Edge - future_throughput_Client_Edge)
        if get_rec_minus_req(userId) != 0:
            factor = get_rec_minus_req(userId) + added_time / get_rec_minus_req(userId)
        else:
            factor = 1
        est_throughput = est_throughput * factor

    return est_throughput


def estimateThroughput(userId, wasPrefetched, latency_Server_Edge, latency_Edge_Client, BW_Server_Edge):
    time_ms = rec_time[userId] - req_time[userId]

    # Estimate mean E2E latency
    global accumulative_E2E_latency
    global n_packets
    accumulative_E2E_latency += time_ms
    n_packets += 1

    if wasPrefetched:
        total_time = time_ms - (latency_Edge_Client * 2)
    else:
        total_time = time_ms - (latency_Edge_Client * 2 + latency_Server_Edge * 2 - req_size[userId] / (BW_Server_Edge))

    estimated_throughput[userId] = req_size[userId] / (total_time / 1000)  # megabits / seg


def set_historical_bitrate(userId, bitrate):
    historical_bitrates_requested[userId].append(bitrate)


def get_historical_bitrate(userId, n_seg):
    return historical_bitrates_requested[userId][n_seg]


def set_historical_qindex(userId, quality_index):
    historical_qindex_requested[userId].append(quality_index)


def get_historical_qindex(userId, n_seg):
    return historical_qindex_requested[userId][n_seg]


def set_historical_QoE_score(userId, value):
    historical_QoE_score[userId].append(value)


def set_historical_predicted_throughput(userId, value):
    historical_predicted_throughput[userId].append(int(value * 1000))


def set_historical_buffer(userId, value):
    historical_buffer[userId].append(value)


def get_historical_QoE_score(userId, n_seg):
    return historical_QoE_score[userId][n_seg]


def get_last_historical_qindex(userId):
    if len(historical_qindex_requested[userId]) >= 2:
        r = historical_qindex_requested[userId][-2]
    else:
        r = 0  # High number so there is no prefetching
    return r


# Plot purposes only
def set_historical_e2e_throughput(userId, data_rate):
    historical_radio_throughput[userId].append(data_rate * 1000)  # Mbps to kbps


# https://dl.acm.org/doi/pdf/10.1145/2155555.2155570
# Segment quality: 4 = 4K; 3 = 1080p; 2 = 720p; 1 = 360p;
def mapIndexToResolution(index):  # mapIndexToResolution
    res = ""
    if index == 19:
        res = "\"1920x1080\""
    elif index == 18:
        res = "\"1920x1080\""
    elif index == 17:
        res = "\"1920x1080\""
    elif index == 16:
        res = "\"1920x1080\""
    elif index == 15:
        res = "\"1920x1080\""
    elif index == 14:
        res = "\"1920x1080\""
    elif index == 13:
        res = "\"1280x720\""
    elif index == 12:
        res = "\"1280x720\""
    elif index == 11:
        res = "\"1280x720\""
    elif index == 10:
        res = "\"1280x720\""
    elif index == 9:
        res = "\"854x480\""
    elif index == 8:
        res = "\"854x480\""
    elif index == 7:
        res = "\"480x360\""
    elif index == 6:
        res = "\"480x360\""
    elif index == 5:
        res = "\"480x360\""
    elif index == 4:
        res = "\"480x360\""
    elif index == 3:
        res = "\"480x360\""
    elif index == 2:
        res = "\"320x240\""
    elif index == 1:
        res = "\"320x240\""
    elif index == 0:
        res = "\"320x240\""
    else:
        print("ERROR in segmentQualityIndex2resolution function, index value: " + index)

    return res


# Segment length: 3 = 10 sec; 2 = 6 sec; 1 = 2 sec;
def mapIndexToSegDuration(index):
    duration = 0
    if index == 1:
        duration = 2
    elif index == 2:
        duration = 6
    elif index == 3:
        duration = 10
    else:
        print("ERROR in segmentLengthIndex2seconds function, index value: " + index)

    return duration


def mapQualityToBitrate(q):
    return bitrate_ladder[q]


# This function measure the mean of the throughput list, not counting values of 0
# samples_number = 20 we get the mean of last 20 values
def meanThroughput(historical_throughput, samples_number):
    counter = 0
    cumulative_throughput = 0
    length = len(historical_throughput)

    if len(historical_throughput) >= samples_number:
        for x in range(length - samples_number, length):
            counter = counter + 1
            cumulative_throughput = cumulative_throughput + historical_throughput[x]
    else:
        for x in range(length):
            counter = counter + 1
            cumulative_throughput = cumulative_throughput + historical_throughput[x]

    cumulative_throughput = cumulative_throughput / counter

    return cumulative_throughput


# This function provide segment size of a segment given segment number, duration and quality.
# Makes use of global variables updated in read_dataset()
def getSegmentSize(userId, segment_number, segment_quality, segment_duration):
    # Segment quality: 0 (lowest) is Q1
    # Segment duration: 3 = 10 sec; 2 = 6 sec; 1 = 2 sec;
    # Segment number starts in 1, but in index thats 0
    segment_size = 0.0

    seg_duration_s = str(mapIndexToSegDuration(segment_duration))

    videoId = User.get_videoId(userId)  # 0 = Big Buck Bunny, 1 = Elephants Dream, 2 = Tears of Steel
    if videoId == 0:
        filename = "dataset/bunny_" + seg_duration_s + "s_Q" + str(segment_quality + 1) + ".txt"
    elif videoId == 1:
        filename = "dataset/ed_" + seg_duration_s + "s_Q" + str(segment_quality + 1) + ".txt"
    elif videoId == 2:
        filename = "dataset/fam_" + seg_duration_s + "s_Q" + str(segment_quality + 1) + ".txt"

    f = open(filename, "r")
    lines = f.read().splitlines()
    segment_size = lines[segment_number - 1]

    if segment_size[-1] == 'K':
        segment_size = segment_size[:-1]
        segment_size = float(segment_size)
        segment_size = int(segment_size)
    elif segment_size[-1] == 'M':
        segment_size = segment_size[:-1]
        segment_size = float(segment_size)
        segment_size = int(segment_size * 1024)

    f.close()

    return segment_size  # in KB


def plotGraph(userId):
    data1 = historical_bitrates_requested[userId]
    data2 = historical_radio_throughput[userId]
    # data2 = historical_predicted_throughput[userId]
    data3 = historical_qindex_requested[userId]

    data1 = data1[:220]
    data2 = data2[:220]
    data3 = data3[:220]

    print("Bitrates:")
    print(data1)
    print("Radio:")
    print(data2)
    print("Index:")
    print(data3)

    font = {'weight': 'bold', 'size': 20}

    matplotlib.rc('font', **font)

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('n request')
    ax1.set_ylabel('bitrate (kbps)', color=color)
    ax1.plot(data1, color=color, label='Bitrates requested')
    ax1.plot(data2, color="green", label='Radio throughput')
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Quality index', color=color)  # we already handled the x-label with ax1
    ax2.plot(data3, color=color, label='Quality index')
    ax2.tick_params(axis='y', labelcolor=color)

    legend = ax1.legend(loc='lower center', shadow=True, fontsize='x-large') and ax2.legend(loc='lower right',
                                                                                            shadow=True,
                                                                                            fontsize='x-large')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


# We use historical_bitrates_requested and historical_radio_throughput (radio)
# We plot fairness index over time and mean fairness index of the session
def CalculateFairness(n_users, clustering_policy):
    n_requests = len(historical_bitrates_requested[0])

    sum_xi = 0
    sum_square_xi = 0

    for userId in range(n_users):
        Ti = mean(historical_bitrates_requested[userId])
        Oi = mean(historical_radio_throughput[userId])
        Xi = Ti / Oi

        # The korean paper only consider mean bitrate:
        Xi = Ti  # Comment or uncomment line

        sum_xi = sum_xi + Xi
        sum_square_xi = sum_square_xi + (Xi ** 2)

    fairness_over_time = (sum_xi ** 2) / (n_users * sum_square_xi)

    # Mean fairness score for the session
    print("Mean fairness for all the users " + str(fairness_over_time))

    # For each cluster
    for userId in range(n_users):
        clusterId = User.get_clusterId(userId)
        # print("User " + str(userId) + " in cluster id: " + str(clusterId))
        sum_xi = 0
        sum_square_xi = 0
        users_in_cluster = 0

        for x in range(n_users):
            if User.get_clusterId(x) == clusterId:
                users_in_cluster += 1

                Ti = mean(historical_bitrates_requested[x])
                Oi = mean(historical_radio_throughput[x])
                Xi = Ti / Oi

                # The korean paper only consider mean bitrate:
                Xi = Ti  # Comment or uncomment line

                sum_xi = sum_xi + Xi
                sum_square_xi = sum_square_xi + (Xi ** 2)

        fairness_over_time = (sum_xi ** 2) / (users_in_cluster * sum_square_xi)

        if clustering_policy == "Video":
            if userId < 3:
                # Mean fairness score for that cluster
                print("Mean fairness for cluster of user " + str(userId) + " is " + str(fairness_over_time))
        elif clustering_policy == "Thr":
            if userId == 0:
                print("Mean fairness for user " + str(userId) + " is: " + str(fairness_over_time))
            if userId == 10:
                print("Mean fairness for user " + str(userId) + " is: " + str(fairness_over_time))
            if userId == 20:
                print("Mean fairness for user " + str(userId) + " is: " + str(fairness_over_time))
        elif clustering_policy == "Par":
            if userId == 0:
                # Mean fairness score for that cluster
                print("Mean fairness for cluster 0 (premium) is " + str(fairness_over_time))
            elif userId == 1:
                print("Mean fairness for cluster 1 is " + str(fairness_over_time))
        else:
            print("Fairness user " + str(userId) + " is " + str(fairness_over_time))


def stats(n_users):
    # Mean bitrate
    mean_bitrate = 0
    mean_bitrate_P = 0
    p_users = 0
    mean_bitrate_NP = 0
    np_users = 0
    for uid in range(n_users):
        print("radio throughput user " + str(uid))
        print(historical_radio_throughput[uid])
        mean_bitrate = mean_bitrate + mean(historical_bitrates_requested[uid])
        if uid % 2 == 0:
            mean_bitrate_P += mean(historical_bitrates_requested[uid])
            p_users += 1
        else:
            mean_bitrate_NP += mean(historical_bitrates_requested[uid])
            np_users += 1

    mean_bitrate = mean_bitrate / n_users
    mean_bitrate_P = mean_bitrate_P / p_users
    #  mean_bitrate_NP = mean_bitrate_NP / np_users
    print("--------------------")
    print("Mean bitrate (kbps): " + str(int(mean_bitrate)))
    print("Mean bitrate premium (kbps): " + str(int(mean_bitrate_P)))
    print("Mean bitrate non-premium (kbps): " + str(int(mean_bitrate_NP)))

    n_stalls = 0
    mean_stall_duration = 0
    n_possible_switches = 0
    n_switches = 0
    n_switches_magnitude = 0
    n_switches_magnitude_qualities = 0
    mean_radio_thr = 0
    n_switches_magnitude_kbps = 0
    for uid in range(n_users):
        # print("user " + str(uid) + " has this number of stalls: " + str(user.get_n_stalls(uid)))
        n_stalls += User.get_n_stalls(uid)
        mean_stall_duration += User.get_rebuffering_acumulated(uid)
        n_possible_switches += len(historical_qindex_requested[uid]) - 1
        for x in range(len(historical_qindex_requested[uid])):
            if x >= 1:
                if historical_qindex_requested[uid][x] != historical_qindex_requested[uid][x - 1]:
                    # If there is a segment switch
                    n_switches += 1
                    n_switches_magnitude_qualities += abs(
                        historical_qindex_requested[uid][x] - historical_qindex_requested[uid][x - 1])
                    n_switches_magnitude_kbps += abs(
                        historical_bitrates_requested[uid][x] - historical_bitrates_requested[uid][x - 1])

        # print("Mean radio throughput: " + str(uid) + " is " + str(mean(historical_radio_throughput[uid])))
        mean_radio_thr += mean(historical_radio_throughput[uid])

    # print("Mean radio throughput total: " + str(mean_radio_thr/n_users))

    # print("Switches (% of total) " + str((n_switches * 100 / n_possible_switches)))
    # print("nº switches: " + str(n_switches) + " nº requests: " + str(n_possible_switches))
    print("Mean switching magnitude (kbps) " + str(int((n_switches_magnitude_kbps / n_switches))))
    print("Mean switching magnitude (qualities) " + str(float((n_switches_magnitude_qualities / n_switches))))
    print("Number of stallings: " + str(n_stalls))
    if n_stalls > 0:
        print("Mean stalling duration (ms): " + str(int((mean_stall_duration / n_stalls))))

    print("Mean latency is " + str(int(accumulative_E2E_latency / n_packets)))


def ECAS_stats(n_users):
    print("ECAS-HAS stats:")

    premium_users = 0
    mean_QoE_premium = 0
    non_premium_users = 0
    mean_QoE_non_premium = 0

    for uid in range(n_users):
        if uid % 2 == 0:
            mean_QoE_premium += mean(historical_QoE_score[uid])
            premium_users += 1
        else:
            mean_QoE_non_premium += mean(historical_QoE_score[uid])
            non_premium_users += 1

    # print("Premium users mean QoE: " + str(int(mean_QoE_premium/premium_users)))
    # if non_premium_users > 0:
    # print("Non-premium users mean QoE: " + str(int(mean_QoE_non_premium / non_premium_users)))

    # Fairness

    sum_xi = 0
    sum_square_xi = 0

    for userId in range(n_users):
        Ti = mean(historical_QoE_score[userId])
        Oi = mean(historical_radio_throughput[userId])
        Xi = Ti / Oi

        # Consider only the score or divide by the radio link throughput, comment or uncomment
        Xi = Ti  # Comment or uncomment line

        sum_xi = sum_xi + Xi
        sum_square_xi = sum_square_xi + (Xi ** 2)

    fairness_over_time = (sum_xi ** 2) / (n_users * sum_square_xi)

    # Mean fairness score for the session
    print("Mean fairness for all the users " + str(fairness_over_time))

    # For each cluster
    for userId in range(n_users):
        clusterId = User.get_subscription(userId)
        # print("User " + str(userId) + " in cluster id: " + str(clusterId))
        sum_xi = 0
        sum_square_xi = 0
        users_in_cluster = 0

        for x in range(n_users):
            if User.get_clusterId(x) == clusterId:
                users_in_cluster += 1

                Ti = mean(historical_bitrates_requested[x])
                Oi = mean(historical_radio_throughput[x])
                Xi = Ti / Oi

                # The korean paper only consider mean bitrate:
                # Xi = Ti  # Comment or uncomment line

                sum_xi = sum_xi + Xi
                sum_square_xi = sum_square_xi + (Xi ** 2)

        fairness_over_time = (sum_xi ** 2) / (users_in_cluster * sum_square_xi)

        if userId == 0:
            # Mean fairness score for that cluster
            print("Mean fairness for cluster 0 (premium) is " + str(fairness_over_time))
        elif userId == 1:
            print("Mean fairness for cluster 1 is " + str(fairness_over_time))


def parseQoE(nusers):
    n_users = nusers
    qoe = [0] * n_users

    sum = 0
    for x in range(n_users):
        filename = "itu-p1203-codecextension/output" + str(x) + ".txt"
        f = open(filename, "r")
        lines = f.readlines()
        # sentence = "  \"O46\": 3.895265870128569, "
        overall_QoE = '\"O46\":'

        for line in lines:
            words = line.split()
            if overall_QoE in words:
                st = words[1]
                st = st[:-1]
                print(st)
                # print("QoE of user " + str(x) + " is: " + str(st))
                qoe[x] = float(st)
                sum = sum + float(st)

    print("Mean QoE for all the users: " + str(sum / n_users))
    sum0 = 0
    sum1 = 0
    sum2 = 0
    c0 = 0
    c1 = 0
    c2 = 0
    fairness = "Par"
    if fairness == "Video":
        for x in range(n_users):
            if x % 3 == 0:
                sum0 = sum0 + qoe[x]
                c0 += 1
            elif x % 3 == 1:
                sum1 = sum1 + qoe[x]
                c1 += 1
            elif x % 3 == 2:
                sum2 = sum2 + qoe[x]
                c2 += 1
    elif fairness == "Thr":
        for userId in range(n_users):
            # We cluster the users in 3 groups, depending on throughput
            if userId < (n_users / 3):
                sum0 = sum0 + qoe[userId]
                c0 += 1
            elif userId < (2 * n_users / 3):
                sum1 = sum1 + qoe[userId]
                c1 += 1
            else:
                sum2 = sum2 + qoe[userId]
                c2 += 1
    elif fairness == "Par":
        for userId in range(n_users):
            if userId % 2 == 0:
                sum0 = sum0 + qoe[userId]
                c0 += 1
            else:
                sum1 = sum1 + qoe[userId]
                c1 += 1

    # print("Mean QoE cluster 1: " + str(sum0/c0))
    # print("Mean QoE cluster 2: " + str(sum1/c1))
    # print("Mean QoE cluster 3: " + str(sum2/c2))
    return (sum / n_users)


def write_data(ss_penalty, stalls_penalty, threshold1, threshold2, qoe, filename):
    filename = "Simulate_GRU_Predictions.txt"
    f = open(filename, "a")
    f.write(str(ss_penalty) + ",")
    f.write(str(stalls_penalty) + ",")
    f.write(str(threshold1) + ",")
    f.write(str(threshold2) + ",")
    f.write(str(qoe))
    f.write("\n")
    f.close()


def writeDataset(nusers):
    n_users = nusers
    for userId in range(n_users):
        filename = "SARA_" + str(userId) + ".txt"
        f = open(filename, "a")
        n_seg_req = get_n_segments_requested(userId)
        for n_req in range(n_seg_req):
            bitrate = get_historical_bitrate(userId, n_req)
            quality_index = get_historical_qindex(userId, n_req)
            f.write(str(userId) + "," + str(n_req) + "," + str(int(bitrate)) + "," + str(quality_index))
            f.write("\n")

    f.close()
