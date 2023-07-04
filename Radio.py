#!/usr/bin/python
# -*- coding: latin-1 -*-

import csv
import User
import random
from matplotlib import pyplot as plt
import numpy as np

################################
# Functions related to radio management
################################

# Historical radio throughput. Works like 2D array. in time radio_time[5] we have the throughput radio_throughput[5]
radio_time = []
radio_throughput = []
IO = 1  # 1 = SISO, 2 = MIMO 2x2, 4 = MIMO 4x4
num_users = 0
generation = "LTE"   # "LTE" or "5G"
scheduler = "PF"
bandwidth = "20MHz"  # 10MHz 5MHz
traces = "DATASET"  # "NS3" (we get MCS and have to parse to throughput) or "DATASET" (we have the throughput)
CN_latency = 0
CN_BW = 0


def init(n_users, gen, trace, mobilityPattern_or_filename, CN_latency_, CN_BW_):
    global num_users
    num_users = n_users
    global generation
    generation = gen
    global traces
    traces = trace
    for i in range(n_users):
        radio_time.append([])
        radio_throughput.append([])
        if trace == "DATASET":
            parse_Dataset_Traces(i, mobilityPattern_or_filename)
        if trace == "DATASET_file":
            parse_Dataset_Automatic(i) # userId = traceId
    if trace == "NS3":
        parse_NS3_Traces()
    global CN_latency
    CN_latency = CN_latency_
    global CN_BW
    CN_BW = CN_BW_


# TO ACCESS: radio_time[userId][position])
# TO ACCESS: radio_throughput[userId][position])
# In X position we have the time and the correspondent throughput

# This function map MCS to TB index
def MCS2TBSindex(MCS):
    TBS_index = 0
    if int(MCS) < 10:
        TBS_index = MCS
    elif int(MCS) < 17:
        TBS_index = MCS - 1
    elif int(MCS) < 29:
        TBS_index = MCS - 2

    x = random.randint(0, 99)
    if x < 50:
        TBS_index -= 1
    if x < 40:
        TBS_index -= 1
    if x < 30:
        TBS_index -= 1
    if x < 20:
        TBS_index -= 1
    if x < 10:
        TBS_index -= 1

    return TBS_index


# This function map TB index to throughput, considering 100 PRB (20 MHz, 1 user)
# 3GPP LTE Standard 36.213 - TB Size table page 87
def mapTBS2Throughput(TBS_index):
    throughput = 0  # bits each TTI
    if TBS_index == 0:
        throughput = 2792
    elif TBS_index == 1:
        throughput = 3624
    elif TBS_index == 2:
        throughput = 4584
    elif TBS_index == 3:
        throughput = 5736
    elif TBS_index == 4:
        throughput = 7224
    elif TBS_index == 5:
        throughput = 8760
    elif TBS_index == 6:
        throughput = 10296
    elif TBS_index == 7:
        throughput = 12216
    elif TBS_index == 8:
        throughput = 14112
    elif TBS_index == 9:
        throughput = 15840
    elif TBS_index == 10:
        throughput = 17568
    elif TBS_index == 11:
        throughput = 19848
    elif TBS_index == 12:
        throughput = 22920
    elif TBS_index == 13:
        throughput = 25456
    elif TBS_index == 14:
        throughput = 28336
    elif TBS_index == 15:
        throughput = 30576
    elif TBS_index == 16:
        throughput = 32856
    elif TBS_index == 17:
        throughput = 36696
    elif TBS_index == 18:
        throughput = 39232
    elif TBS_index == 19:
        throughput = 43816
    elif TBS_index == 20:
        throughput = 46888
    elif TBS_index == 21:
        throughput = 51024
    elif TBS_index == 22:
        throughput = 55056
    elif TBS_index == 23:
        throughput = 57336
    elif TBS_index == 24:
        throughput = 61664
    elif TBS_index == 25:
        throughput = 63667
    elif TBS_index == 26:
        throughput = 75376

    return throughput


# We follow https://5g-tools.com/5g-nr-throughput-calculator/
# DL, FDD, 1 Component Carrier, No MU-MIMO/Massive MIMO, BW:50MHz FR1 µ:30kHz:
def MCStoThroughput5G(MCS):
    throughput = 0
    MCS = int(MCS)
    if MCS == 28:
        throughput = 854
    elif MCS == 27:
        throughput = 820
    elif MCS == 26:
        throughput = 786
    elif MCS == 25:
        throughput = 740
    elif MCS == 24:
        throughput = 696
    elif MCS == 23:
        throughput = 648
    elif MCS == 22:
        throughput = 600
    elif MCS == 21:
        throughput = 554
    elif MCS == 20:
        throughput = 510
    elif MCS == 19:
        throughput = 466
    elif MCS == 18:
        throughput = 420
    elif MCS == 17:
        throughput = 394
    elif MCS == 16:
        throughput = 396
    elif MCS == 15:
        throughput = 370
    elif MCS == 14:
        throughput = 332
    elif MCS == 13:
        throughput = 294
    elif MCS == 12:
        throughput = 260
    elif MCS == 11:
        throughput = 226
    elif MCS == 10:
        throughput = 204
    elif MCS == 9:
        throughput = 204
    elif MCS == 8:
        throughput = 180
    elif MCS == 7:
        throughput = 158
    elif MCS == 6:
        throughput = 134
    elif MCS == 5:
        throughput = 114
    elif MCS == 4:
        throughput = 92
    elif MCS == 3:
        throughput = 76
    elif MCS == 2:
        throughput = 58
    elif MCS == 1:
        throughput = 48
    elif MCS == 0:
        throughput = 36

    return throughput  # In Mbps


# This function open the file UlTxPhyStats generated in NS-3 and return a table with time - throughput.
# Makes use of functions: MCS2TBSindex and mapTBS2Throughput
def parse_NS3_Traces():
    csv_file = open('radioMetrics5KMDisc.txt', 'r')
    # csv_file = open('radio25UE2500m.txt', 'r')
    # csv_file = open('radioMetrics1UE.txt', 'r')
    # csv_file = open('25UE15ms.txt', 'r')
    first = True  # to avoid copy legend "'% time', 'mcs'],"
    number_columns = 10  # Fixed number of columns of csv file (node, action, file...)
    with csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            # Row 0 is time, row 5 MCS
            """
            Convert MCS to bandwidth:
            Define bandwidth. Then map it to a number of Resource Blocks. We assume always 20 MHz so 100 PRB. But we can implement this in a future
            Divide resource block by number of clients. I assume 1 client. For more clients (future upgrade of the testbed) we should get the number of PRB and then map TBS index to thoughput according to the number of PRB per user.
            Map MCS to TBS index
            Map TBS index to bits per TTI (3gpp document 36.213)
            """

            if first:
                first = False
            else:
                MCS = row[5]

                if generation == "LTE":
                    TBS_index = MCS2TBSindex(int(MCS))
                    throughput = mapTBS2Throughput(TBS_index)  # bits each ms
                    throughput = throughput / 1000  # Mbps (*1000 to s, /1kk to Mb)
                elif generation == "5G":
                    throughput = MCStoThroughput5G(MCS)

                if IO == 2:
                    # MIMO 2x2
                    throughput = throughput * 2
                elif IO == 4:
                    # MIMO 4x4
                    throughput = throughput * 4
                # Else is SISO, througput * 1
                # That is throughput with 20 MHz, i.e. 100 PRB

                userId = int(row[2]) - 1
                radio_time[userId].append(row[0])
                radio_throughput[userId].append(throughput)

    # plt.plot(radio_throughput[userId])
    # plt.show()
    csv_file.close()


def parse_Dataset_Traces(userId, mobilityPattern):

    if mobilityPattern == "bus":
        n_files = 16
    elif mobilityPattern == "car":
        n_files = 53
    elif mobilityPattern == "pedestrian":
        n_files = 31
    elif mobilityPattern == "static":
        n_files = 15
    elif mobilityPattern == "train":
        n_files = 20

    id = (userId%n_files) + 1

    filename = "radio_dataset/" + mobilityPattern + '/' + str(id) + ".csv"

    csv_file = open(filename, 'r')
    offset = 10  # we start 10 second later to avoid 0 throughput
    counter = 0  # to avoid copy legend "'% time', 'mcs'],"
    number_columns = 10  # Fixed number of columns of csv file (node, action, file...)
    time_counter = 0
    with csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            counter += 1
            if counter > offset:
                DL_bitrate = float(row[12]) / 1000  # kbps to Mbps
                if DL_bitrate == 0:
                    DL_bitrate = 0.01

                radio_time[userId].append(time_counter)
                radio_throughput[userId].append(DL_bitrate)

                time_counter += 1000  # 1 second between traces

    # plt.plot(radio_throughput[userId])
    # plt.show()
    csv_file.close()


def parse_Dataset_Automatic(userId):

    # We select 5 radio profiles
    n = userId%5
    if n == 0:
        radiofile_number = 10
    elif n == 1:
        radiofile_number = 26
    elif n == 2:
        radiofile_number = 81
    elif n == 3:
        radiofile_number = 100
    elif n == 4:
        radiofile_number = 119 # 116

    # We might overwrite here to test with 135 users each one with different radio file
    #radiofile_number = int(userId+1)

    User.set_tracenumber(userId, radiofile_number)
    if radiofile_number < 10:
        payload = "00"
    elif radiofile_number < 100:
        payload = "0"
    else:
        payload = ""

    trace_name = "R_" + payload + str(radiofile_number)

    filename = "automatic_test_dataset/" + trace_name + ".csv"

    print("RADIO TRACES: " + str(filename))

    csv_file = open(filename, 'r')
    offset = 10  # we start 10 second later to avoid 0 throughput
    counter = 0  # to avoid copy legend "'% time', 'mcs'],"
    number_columns = 10  # Fixed number of columns of csv file (node, action, file...)
    time_counter = 0
    with csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            counter += 1
            if counter > offset:
                DL_bitrate = float(row[12]) / 1000  # kbps to Mbps
                if DL_bitrate == 0:
                    DL_bitrate = 0.01

                radio_time[userId].append(time_counter)
                radio_throughput[userId].append(DL_bitrate)

                time_counter += 1000  # 1 second between traces

    # plt.plot(radio_throughput[userId])
    # plt.show()
    csv_file.close()


# This function provides the downlink throughput given a time and throughput table generated in parse_MCS()
def throughputGivenTime(userId, time):

    global traces

    if traces == "NS3":
        time = 9000  # We do this for using a 10 second file so is faster as the values does not change during the simulation... FIX!!!!

    index = 0
    throughput = int(radio_throughput[userId][index])
    lower_bound = 0  # of time
    higher_bound = int(radio_time[userId][index])
    max_index = len(radio_time[userId]) - 1


    if time >= int(radio_time[userId][max_index]):
        #print("-------------------------------------------------------------------------------")
        print("Warning: throughputGivenTime - you request a time higher than the radio data time")
        print("User and trace id: " + str(userId))
        #print("Time requested: " + str(time))
        #print("Values only between " + str(radio_time[userId][0]) + " and " + str(radio_time[userId][max_index]))
        #print("-------------------------------------------------------------------------------")
    else:
        while not ((time < higher_bound) and (time >= lower_bound)):
            index = index + 1
            lower_bound = higher_bound
            if index <= max_index:
                higher_bound = int(radio_time[userId][index])
            else:
                index = max_index -1 # we should fix this

    throughput = float(radio_throughput[userId][index])

    if traces == "NS3":
        throughput_initial = throughput

        if bandwidth == "20MHz":
            PRB_factor = 1
        elif bandwidth == "10MHz":
            PRB_factor = 2
        elif bandwidth == "5MHz":
            PRB_factor = 4

        throughput = throughput_initial / PRB_factor

        # Scheduling
        if scheduler == "RR":
            # equal to the number of clients
            throughput = throughput / num_users
        elif scheduler == "PF":

            # In PF, each user has a weight factor of (throughput / Sum all throughputs)
            sum_throughputs = 0
            for x in range(num_users):
                sum_throughputs = sum_throughputs + radioThroughputGivenTimeNS3(x, time)
            PF_factor = throughput_initial / sum_throughputs
            throughput = throughput * PF_factor

    return throughput  # Mbps


def calculateTxTimeRAN(userId, time, data):
    global CN_latency
    global CN_BW
    # Data in KB; throughput in Mbps
    throughput = throughputGivenTime(userId, time)
    # RAN_tx_time = data * 0.008 / throughput - data * 0.008 / CN_BW - CN_latency*0.001

    # Assuming the dataset provide radio throughput
    RAN_tx_time = data * 0.008 / throughput

    return RAN_tx_time


def radioThroughputGivenTimeNS3(userId, time):

    # time = 9000  # Fix value for debugging
    index = 0
    throughput = int(radio_throughput[userId][index])

    lower_bound = 0  # of time
    higher_bound = int(radio_time[userId][index])

    max_index = len(radio_time[userId]) - 1

    if time > int(radio_time[userId][max_index]):
        print("-------------------------------------------------------------------------------")
        print("ERROR: throughputGivenTime - you request a time higher than the radio data time")
        print("Time requested: " + str(time))
        print("Values only between " + str(radio_time[userId][0]) + " and " + str(radio_time[userId][max_index]))
        print("-------------------------------------------------------------------------------")
    else:
        while not ((time < higher_bound) and (time >= lower_bound)):
            index = index + 1
            lower_bound = higher_bound
            higher_bound = int(radio_time[userId][index])

    throughput = float(radio_throughput[userId][index])

    return throughput

"""
#  Mean radio throughput
def mean_radio_throughput(userId, time, window):
    # window (ms) of time to perform the mean
    mean = 0
    step = 50  # step (ms) of time to take the samples of radio throughput

    if time < window:
        window = time

    for x in range(int(window / step)):
        mean = mean + radioThroughputGivenTime(userId, time - window + step)

    mean = mean / (window / step)

    return mean  # in Mbps
"""

"""
def predict_next_throughput(userId, time, segment_duration):
    seg_duration = segment_duration * 1000  # s to ms
    window = seg_duration * 2  # window (ms) of time to measure the tendency

    current_throughput = radioThroughputGivenTime(userId, time+1)
    #  Step 1: Calculate tendency
    if time > window:
        past_throughput = radioThroughputGivenTime(userId, time+1 - window)
        slope = (current_throughput - past_throughput) / window
    else:
        past_throughput = radioThroughputGivenTime(userId, 1)
        slope = (current_throughput - past_throughput) / (time+1)

    # Step 2: Predict a future value
    future_throughput = current_throughput + slope * seg_duration

    # Step 3: Average mean future throughput
    future_mean_throughput = (future_throughput + current_throughput) / 2

    return future_mean_throughput
"""
