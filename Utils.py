#!/usr/bin/python
# -*- coding: latin-1 -*-

import User
from matplotlib import pyplot as plt
import matplotlib.pyplot
from statistics import mean
import User
import numpy
from numpy import random
from scipy import stats
import math

historical_bitrates_requested = []
historical_radio_throughput = []
historical_predicted_throughput = []
historical_qindex_requested = []
historical_QoE_score = []
historical_buffer = []
video_popularity = []

# For estimate the throughput
req_time = []
rec_time = []
req_size = []
n_packets = 0
accumulative_E2E_latency = 0  # estimate latency
cache = []
cacheSize = 0  # KB
cacheLimit = 256000  # KB

qualityDistribution = [0, 0, 0, 0, 0, 0]  # six qualities
bitrate_ladder_type = 0


# For harmonic mean in SARA
def init(nusers, n_videos, bitrate_ladder_typ):
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

    global video_popularity
    video_popularity = random.zipf(a=2, size=n_videos)
    global bitrate_ladder_type
    bitrate_ladder_type = bitrate_ladder_typ

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


def mapIndexToResolution(index):  # mapIndexToResolution
    """
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
    """
    res = ""
    if index == 5:
        res = "\"3840x2160\""
    elif index == 4:
        res = "\"2560x1440\""
    elif index == 3:
        res = "\"1920x1080\""
    elif index == 2:
        res = "\"1280x720\""
    elif index == 1:
        res = "\"640x360\""
    elif index == 0:
        res = "\"320x180\""
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
    """
    if videoId < 3:
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

    """

    if bitrate_ladder_type == 1:
        if segment_quality == 0:
            segment_size = 1150
        elif segment_quality == 1:
            segment_size = 2150
        elif segment_quality == 2:
            segment_size = 3500
        elif segment_quality == 3:
            segment_size = 8500
    else:
        # Dataset with 6 qualities, 4 seconds, fixed size over segments
        if segment_quality == 0:
            segment_size = 100
        elif segment_quality == 1:
            segment_size = 375
        elif segment_quality == 2:
            segment_size = 1150
        elif segment_quality == 3:
            segment_size = 2150
        elif segment_quality == 4:
            segment_size = 3500
        elif segment_quality == 5:
            segment_size = 8500

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
    mean_quality = 0.0
    p_users = 0
    mean_bitrate_NP = 0
    np_users = 0
    for uid in range(n_users):
        print("radio throughput user " + str(uid))
        print(historical_radio_throughput[uid])
        mean_bitrate = mean_bitrate + mean(historical_bitrates_requested[uid])
        mean_quality = mean_quality + mean(historical_qindex_requested[uid])
        if uid % 2 == 0:
            mean_bitrate_P += mean(historical_bitrates_requested[uid])
            p_users += 1
        else:
            mean_bitrate_NP += mean(historical_bitrates_requested[uid])
            np_users += 1

    mean_bitrate = mean_bitrate / n_users
    mean_quality = float(mean_quality / n_users)
    mean_bitrate_P = mean_bitrate_P / p_users
    #  mean_bitrate_NP = mean_bitrate_NP / np_users
    print("--------------------")
    print("Mean bitrate (kbps): " + str(int(mean_bitrate)))
    print("Mean quality: " + str(mean_quality))
    #print("Mean bitrate premium (kbps): " + str(int(mean_bitrate_P)))
    #print("Mean bitrate non-premium (kbps): " + str(int(mean_bitrate_NP)))

    n_stalls = 0
    mean_stall_duration = 0
    n_possible_switches = 0
    n_switches = 0
    n_switches_magnitude = 0
    n_switches_magnitude_qualities = 0
    mean_radio_thr = 0
    n_switches_magnitude_kbps = 0
    for uid in range(n_users):
        #print("user " + str(uid) + " has this number of stalls: " + str(User.get_n_stalls(uid)))
        #print("user " + str(uid) + " has this mean bitrate: " + str(mean(historical_bitrates_requested[uid])))
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
    if n_switches == 0:
        print("Mean switching magnitude (kbps) " + str(int(0)))
        print("Mean switching magnitude (qualities) " + str(float(0)))
    else:
        print("Mean switching magnitude (kbps) " + str(int((n_switches_magnitude_kbps / n_switches))))
        print("Mean switching magnitude (qualities) " + str(float((n_switches_magnitude_qualities / n_switches))))
    print("Number of stallings: " + str(n_stalls))
    if n_stalls > 0:
        print("Mean stalling duration (ms): " + str(int((mean_stall_duration / n_stalls))))

    #print("Mean latency is " + str(int(accumulative_E2E_latency / n_packets)))


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
                #print(st)
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


def writeMLdata(nusers):
    n_users = nusers
    filename = "BBA_ML.csv"
    f = open(filename, "a")
    f.write("Buffer, Bandwidth, PreviousQuality, PreviousBandwidth, Quality")
    f.write("\n")
    for userId in range(n_users):
        for s in range(get_n_segments_requested(userId)):
            buffer = int(historical_buffer[userId][s] * 1000)
            bandwidth = historical_predicted_throughput[userId][s]
            if s == 0:
                previous_q = 0
                previos_bw = 0
            else:
                previous_q = historical_qindex_requested[userId][s - 1]
                previos_bw = historical_predicted_throughput[userId][s - 1]
            q = historical_qindex_requested[userId][s]

            #print(str(buffer) + "," + str(bandwidth) + "," + str(previous_q) + "," + str(previos_bw) + "," + str(q))
            f.write(str(buffer) + "," + str(bandwidth) + "," + str(previous_q+1) + "," + str(previos_bw) + "," + str(q+1))
            f.write("\n")

    f.close()


# historical_buffer[userId][n_seg]
# historical_predicted_throughput[userId][n_seg] # Download bandwidth (Mbps),


#############
## CACHING ##
#############


def createCache(n_users, n_videos, n_segments, n_qualities):
    global cache
    n_features = 4
    #cache = numpy.zeros((n_videos, n_segments, n_qualities, n_features))
    cache = numpy.zeros((n_videos, n_segments+n_users, n_qualities, n_features)) # Desynchronization


def isCached(videoId, segmentId, qualityId):
    return cache[videoId][segmentId][qualityId][0] != 0


def addToCache(userId, videoId, segmentId, qualityId, size, cachingPolicy):
    global cache, cacheSize, cacheLimit
    wasCached = True

    segmentId = segmentId + userId # Desynchronization

    if size < cacheLimit:
        # If not already cached
        if cache[videoId][segmentId][qualityId][0] == 0:
            wasCached = False
            # if there is space, is cached
            if cacheSize + size < cacheLimit:
                cache[videoId][segmentId][qualityId][0] = size
                cache[videoId][segmentId][qualityId][1] = 0
                cache[videoId][segmentId][qualityId][2] = 0
                cache[videoId][segmentId][qualityId][3] = getVideoPopularity(videoId) * getQualityPopularity(qualityId)

                cacheSize += size
            # if there is no space, we do a sustitution
            else:
                if cachingPolicy == "LRU":
                    cacheSustitution(1, size, videoId, segmentId, qualityId)
                elif cachingPolicy == "LFU":
                    cacheSustitution(2, size, videoId, segmentId, qualityId)
                elif cachingPolicy == "LPU":
                    cacheSustitution(3, size, videoId, segmentId, qualityId)
        else:
            # already cached
            cache[videoId][segmentId][qualityId][1] = 0
            cache[videoId][segmentId][qualityId][2] += 1

        updateCache()

    else:
        # If the size of the segment is larger than the cache limit, it cannot be cached
        wasCached = False

    return wasCached

def updateCache():
    # Update features
    for i in range(len(cache)):
        for j in range(len(cache[i])):
            for k in range(len(cache[i][j])):
                if isCached(i, j, k):
                    cache[i][j][k][1] -= 1


def cacheFindLowestValue(featureId):
    if featureId == 1:
        lowest_value = 0
    elif featureId == 2:
        lowest_value = 99999
    elif featureId == 3:
        lowest_value = 1

    lowest_value_videoId = -1
    lowest_value_segmentId = -1
    lowest_value_qualityId = -1

    for i in range(len(cache)):
        for j in range(len(cache[i])):
            for k in range(len(cache[i][j])):
                if isCached(i, j, k):
                    if cache[i][j][k][featureId] < lowest_value:
                        lowest_value = cache[i][j][k][featureId]
                        lowest_value_videoId = i
                        lowest_value_segmentId = j
                        lowest_value_qualityId = k

    return lowest_value_videoId, lowest_value_segmentId, lowest_value_qualityId


def cacheSustitution(featureId, size, videoId, segmentId, qualityId):
    global cacheSize

    while cacheSize + size > cacheLimit:
        # free space
        old_videoId, old_segmentId, old_qualityId = cacheFindLowestValue(featureId)
        cacheSize = cacheSize - cache[old_videoId][old_segmentId][old_qualityId][0]
        cache[old_videoId][old_segmentId][old_qualityId][0] = 0
        cache[old_videoId][old_segmentId][old_qualityId][1] = 0
        cache[old_videoId][old_segmentId][old_qualityId][2] = 0
        cache[old_videoId][old_segmentId][old_qualityId][3] = 0

    # update
    cache[videoId][segmentId][qualityId][0] = size
    cacheSize += size
    cache[videoId][segmentId][qualityId][1] = 0
    cache[videoId][segmentId][qualityId][2] = 0
    cache[videoId][segmentId][qualityId][3] = getVideoPopularity(videoId) * getQualityPopularity(qualityId)
    #print("Video popularity: " + str(getVideoPopularity(videoId)) + " Quality popularity: " + str(getQualityPopularity(qualityId)) + " LPU score: " + str(getVideoPopularity(videoId) * getQualityPopularity(qualityId)))


def visualizeCache():
    print(cache)


def getCacheSize():
    return cacheSize


######################
## VIDEO POPULARITY ##
######################

# Return normalized video popularity (0,1)
def getVideoPopularity(videoId):

    global video_popularity
    total = sum(video_popularity)
    if total == 0:
        popularity = 0
    else:
        popularity = video_popularity[videoId] / total

    return popularity


def getRandomVideoId(n_videos):
    r = random.uniform(0, 1)
    s = 0
    videoId = 0

    while r > (s + getVideoPopularity(videoId)):
        videoId += 1
        if videoId < n_videos-1:
            s = s + getVideoPopularity(videoId)
        else:
            r = 999  # out of while

    #print("Video Id: " + str(videoId))
    return videoId

# Return normalized quality popularity (0,1)
def getQualityPopularity(qualityId):
    total = sum(qualityDistribution)
    if total == 0:
        popularity = 0
    else:
        popularity = qualityDistribution[qualityId] / total

    return popularity


def updateQualityPopularity(qualityId):
    qualityDistribution[qualityId] += 1



def showVideoPopularity(n_users):
    for x in range(n_users):
        videoId = User.get_videoId(x)
        print("User " + str(x) + " watch video id " + str(videoId))


# Transrating time in ms
def getTransratingTime(bitrate_ladder_type, init_Q, final_Q):
    # Bitrate ladder types:
    # (0) 200, 750, 2300, 4300
    # (1) 2300, 4300, 7000, 17000
    # (2) 200, 750, 2300, 4300, 7000, 17000

    wait_ms = 0

    if bitrate_ladder_type == 1:
        init_Q = init_Q + 2

    if init_Q == 1:
        if final_Q == 0:
            wait_ms = 286
    if init_Q == 2:
        if final_Q == 0:
            wait_ms = 538
        elif final_Q == 1:
            wait_ms = 852
    if init_Q == 3:
        if final_Q == 0:
            wait_ms = 735
        elif final_Q == 1:
            wait_ms = 1060
        elif final_Q == 2:
            wait_ms = 1993
    if init_Q == 4:
        if final_Q == 0:
            wait_ms = 914
        elif final_Q == 1:
            wait_ms = 1202
        elif final_Q == 2:
            wait_ms = 2124
        elif final_Q == 3:
            wait_ms = 3856
    if init_Q == 5:
        if final_Q == 0:
            wait_ms = 1731
        elif final_Q == 1:
            wait_ms = 2008
        elif final_Q == 2:
            wait_ms = 3081
        elif final_Q == 3:
            wait_ms = 4519
        elif final_Q == 4:
            wait_ms = 6819

    wait_ms = wait_ms / 100
    return wait_ms


def getSuperresolutionTime(quality):
    #quality from 0 to 5, but bitrate ladder is reduced to 1 to 4
    computing_time = 0
    if quality == 2:
        computing_time = 1778
    elif quality == 3:
        computing_time = 2747
    elif quality == 4:
        computing_time = 4437
    elif quality == 5:
        computing_time = 4437
    computing_time = computing_time/8
    return computing_time
