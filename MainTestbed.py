import Utils
import P1203_Evaluation
import ABR
import Radio
import User
import EdgeAssisted
import sys
import numpy

from numpy import random

"""
Units: 
- KB for size, kilobyte. 1 KB = 8000 bits
- Mbps or Mbit/s for throughput. 1.000.000 bits per second. 1 MB/s (Megabyte) = 8 Mbps.
- ms for time. 1 s = 1000 ms
"""


class Server:
    def __init__(self, BW_Server_Edge, latency_Server_Edge):
        self.BW_Server_Edge = BW_Server_Edge  # Mbps
        self.Latency_Server_Edge = latency_Server_Edge

    def sendSegment2Edge(self, time, userId):
        segment_number = User.get_segment_number(userId)
        segment_duration = User.get_segment_duration(userId)
        segment_quality = User.get_segment_quality(userId)

        #Utils.updateQualityPopularity(segment_quality) # For LPU cache

        # We get the segment size from the dataset
        seg_size_KB = Utils.getSegmentSize(userId, segment_number, segment_quality, segment_duration)
        seg_size_megabits = float(seg_size_KB) * 0.008

        Utils.set_req_size(userId, seg_size_megabits)

        prefetching_policy = User.get_prefetching_policy(userId)
        if prefetching_policy == 4:
            # Transrating, we requested the highest
            seg_size_KB = Utils.getSegmentSize(userId, segment_number, n_qualities - 1, segment_duration)
            seg_size_megabits = float(seg_size_KB) * 0.008
        elif prefetching_policy == 7:
            # superresolution, we requested the 2nd lowest
            seg_size_KB = Utils.getSegmentSize(userId, segment_number, 1, segment_duration)
            seg_size_megabits = float(seg_size_KB) * 0.008

        tx_time_sec = seg_size_megabits / self.BW_Server_Edge  # seconds
        end_time = time + tx_time_sec * 1000  # ms
        end_time = end_time + self.Latency_Server_Edge

        return end_time


class Edge:
    def __init__(self, BW_Edge_Server, latency_Edge_Server):
        self.BW_Edge_Server = BW_Edge_Server  # Mbps
        self.latency_Edge_Server = latency_Edge_Server

    # Send packet from edge to Client
    def sendPacket2Client(self, time_s, userId):

        segment_number = User.get_segment_number(userId)
        segment_quality = User.get_segment_quality(userId)
        segment_duration = User.get_segment_duration(userId)

        seg_size_KB = Utils.getSegmentSize(userId, segment_number, segment_quality, segment_duration)

        time = time_s
        last_q = Utils.get_last_historical_qindex(userId)

        # Caching and prefetching
        # First, we check if the segment was cached
        global possible_hits
        possible_hits += 1

        cachingProb = 10
        ranCaching = random.uniform(0, 100)
        if ranCaching > cachingProb:
            wasCached = False
        else:
            wasCached = True
        #wasCached = Utils.addToCache(userId, User.get_videoId(userId), segment_number, segment_quality, seg_size_KB, cachingPolicy)
        wasCached = False  # Deactivate caching
        if wasCached:
            # it was cached
            time = prefetch_time[userId]
            wasPrefetched[userId] = True  # was served from the edge
            # Caching stats
            global caching_hits
            caching_hits += 1
        else:
            # If it is not cached, we check if the segment was prefetched
            global prefetched_hits
            global backhaul_data_sent_KB
            global needed_data_sent_KB
            last_q = Utils.get_last_historical_qindex(userId)
            prefetching_policy = User.get_prefetching_policy(userId)
            needed_data_sent_KB += Utils.getSegmentSize(userId, segment_number, segment_quality, segment_duration)
            if segment_number >= 2:
                if prefetching_policy == 0:
                    #  No prefetching
                    backhaul_data_sent_KB += Utils.getSegmentSize(userId, segment_number, segment_quality, segment_duration)
                elif prefetching_policy == 1:
                    # Prefetch all
                    for x in range(n_qualities):
                        backhaul_data_sent_KB += Utils.getSegmentSize(userId, segment_number, x, segment_duration)

                    time = prefetch_time[userId]
                    prefetched_hits += 1
                    wasPrefetched[userId] = True
                elif prefetching_policy == 2:
                    # Prefetch same
                    backhaul_data_sent_KB += Utils.getSegmentSize(userId, segment_number, segment_quality, segment_duration)
                    if last_q != segment_quality:
                        backhaul_data_sent_KB += Utils.getSegmentSize(userId, segment_number, last_q, segment_duration)
                    else:
                        time = prefetch_time[userId]
                        prefetched_hits += 1
                        wasPrefetched[userId] = True

                elif prefetching_policy == 3:
                    # Prefetch same, +1, -1
                    backhaul_data_sent_KB += Utils.getSegmentSize(userId, segment_number, segment_quality, segment_duration)
                    if last_q != segment_quality:
                        backhaul_data_sent_KB += Utils.getSegmentSize(userId, segment_number, last_q, segment_duration)
                    if last_q + 1 != segment_quality and last_q + 1 < 20:
                        backhaul_data_sent_KB += Utils.getSegmentSize(userId, segment_number, last_q + 1, segment_duration)
                    if last_q - 1 != segment_quality and last_q - 1 >= 0:
                        backhaul_data_sent_KB += Utils.getSegmentSize(userId, segment_number, last_q - 1, segment_duration)
                    if segment_quality == last_q or segment_quality == last_q + 1 or segment_quality == last_q - 1:
                        time = prefetch_time[userId]
                        prefetched_hits += 1
                        wasPrefetched[userId] = True
                elif prefetching_policy == 4:
                    # Transrating, we sent the highest quality
                    backhaul_data_sent_KB += Utils.getSegmentSize(userId, segment_number, n_qualities - 1, segment_duration)
                    # And then we do the transrating process
                    global bitrate_ladder_type # this should be in other place, move it.
                    transrating_time = Utils.getTransratingTime(bitrate_ladder_type,n_qualities - 1,segment_quality)
                    # 100% hit rate as we transrate to the desired quality requested by the user
                    transrating_time = transrating_time / 2
                    time = prefetch_time[userId] + transrating_time
                    wasPrefetched[userId] = True
                    prefetched_hits += 1

                elif prefetching_policy == 5:
                    # Markov
                    # Less than 10 requests for that quality = no reliable, we follow ABR, no prefetching
                    # More than 10 requests for that quality – we follow Markov tree, prefetching the quality with most probability to be requested

                    backhaul_data_sent_KB += Utils.getSegmentSize(userId, segment_number, last_q, segment_duration)

                    sum_markov = 0
                    highest = -1
                    highest_q = 0 # We want the quality to prefetch that was requested the most
                    for x in range(n_qualities):
                        r = last_q
                        n_r = User.get_markov(userId,r,x)
                        sum_markov = sum_markov + n_r
                        if n_r > highest:
                            highest = n_r
                            highest_q = x

                    if sum_markov >= 10:
                        predicted = highest_q
                        if predicted != segment_quality:
                            backhaul_data_sent_KB += Utils.getSegmentSize(userId, segment_number, last_q,
                                                                          segment_duration)
                        else:
                            time = prefetch_time[userId]
                            prefetched_hits += 1
                            wasPrefetched[userId] = True

                elif prefetching_policy == 6:
                    backhaul_data_sent_KB += Utils.getSegmentSize(userId, segment_number, last_q,
                                                                  segment_duration)
                    # Machine learning
                    if ABR_Algorithm == "TBA" or ABR_Algorithm == "BBA":
                        hit = True
                    else:
                        accuracy = 92.5
                        ran = random.uniform(0, 100)
                        if ran > accuracy:
                            hit = False
                        else:
                            hit = True
                    if hit:
                        # was successfully prefetched
                        ML_time = 100  # ms
                        time = prefetch_time[userId] + ML_time
                        wasPrefetched[userId] = True
                        prefetched_hits += 1
                    else:
                        # Note: we use last_q as predicted quality as simplification
                        backhaul_data_sent_KB += Utils.getSegmentSize(userId, segment_number, last_q,
                                                                  segment_duration)
                elif prefetching_policy == 7:
                    # super resolution, ONLY with bitrate ladder C
                    backhaul_data_sent_KB += Utils.getSegmentSize(userId, segment_number, 1,
                                                                  segment_duration)
                    # And then we do the transrating process
                    computing_time = Utils.getSuperresolutionTime(segment_quality)
                    time = prefetch_time[userId] + computing_time
                    wasPrefetched[userId] = True
                    prefetched_hits += 1
            else:
                backhaul_data_sent_KB += Utils.getSegmentSize(userId, segment_number, segment_quality, segment_duration)
                wasPrefetched[userId] = False

        seg_size_KB = Utils.getSegmentSize(userId, segment_number, segment_quality, segment_duration)
        RAN_tx_time = Radio.calculateTxTimeRAN(userId, time, seg_size_KB)
        end_time = time + RAN_tx_time * 1000  # ms
        end_time = end_time + User.get_latency(userId)

        # QoE metadata
        global QoEmetadata
        global metadata_start
        codec = "\"hevc\""
        fps = str(24.0)

        duration = Utils.mapIndexToSegDuration(segment_duration)
        segment_kbps = float(seg_size_KB) * 8 / duration
        if first_segment[userId]:
            first_segment[userId] = False
            segment_metadata = P1203_Evaluation.segmentGenerator(str(segment_kbps), codec,
                                                                 str(Utils.mapIndexToSegDuration(segment_duration)),
                                                                 fps, str(Utils.mapIndexToResolution(segment_quality)),
                                                                 str(metadata_start[userId]), True)
        else:
            segment_metadata = P1203_Evaluation.segmentGenerator(str(segment_kbps), codec,
                                                                 str(Utils.mapIndexToSegDuration(segment_duration)),
                                                                 fps, str(Utils.mapIndexToResolution(segment_quality)),
                                                                 str(metadata_start[userId]), False)
        QoEmetadata[userId] = QoEmetadata[userId] + segment_metadata

        metadata_start[userId] = metadata_start[userId] + Utils.mapIndexToSegDuration(segment_duration)

        return end_time

    def sendRequest2Server(self, time, userId):
        prefetch_time[userId] = time

        # Superresolution
        if User.get_prefetching_policy(userId) == 7:
            segment_quality = User.get_segment_quality(userId)
            if segment_quality == 0:
                User.set_segment_quality(userId, 1)
            elif segment_quality == 5:
                User.set_segment_quality(userId, 4)

        segment_number = User.get_segment_number(userId)
        segment_quality = User.get_segment_quality(userId)
        segment_duration = User.get_segment_duration(userId)
        quality_to_request = segment_quality

        segment_size_ladder = [0] * n_qualities
        for x in range(n_qualities):
            segment_size_ladder[x] = float(Utils.getSegmentSize(userId, segment_number + 1, x, segment_duration))  # KB

        seg_duration = Utils.mapIndexToSegDuration(segment_duration)  # index to seconds

        # At the begining, we let the Client ABR algorithm start the video streaming, once the buffer is filled and
        # it is estabilized, EADAS start working
        if time > (EADAS_startup_time * 2000):
            # if user.get_buffer(userId) > user.get_buffer_target(userId):
            # EADAS
            if mode == 'EADAS':
                quality_to_request = EdgeAssisted.EADAS(time, userId, n_qualities, segment_quality,
                                                        segment_size_ladder)

                User.set_segment_quality(userId, quality_to_request)

        # ECAS
        if mode == 'ECAS':
            quality_to_request = EdgeAssisted.ECAS(userId, time, n_qualities,
                                                   self.BW_Edge_Server, self.latency_Edge_Server)
            User.set_segment_quality(userId, quality_to_request)

        if mode == 'korean' and segment_number > 1:
            quality_to_request = EdgeAssisted.KoreanEdgeBasedAlgorithm(userId, time, n_qualities)
            User.set_segment_quality(userId, quality_to_request)

        # Fairness in EADAS
        segment_size = float(segment_size_ladder[quality_to_request])
        EdgeAssisted.set_last_segment_size_requested(userId, segment_size)

        # Savings bitrates
        size = Utils.getSegmentSize(userId, segment_number, quality_to_request, segment_duration)
        duration = Utils.mapIndexToSegDuration(segment_duration)
        bitrate_next_segment = float(size) * 8 / duration  # KB/s to kbps
        Utils.set_historical_bitrate(userId, bitrate_next_segment)
        Utils.set_historical_qindex(userId, quality_to_request)

        last_bitrate = User.get_bitrate(userId)
        User.add_bitrate_changes(userId, abs(last_bitrate - bitrate_next_segment))
        User.set_bitrate(userId, bitrate_next_segment)

        seg_size_megabits = request_size * 0.008
        tx_time_sec = seg_size_megabits / self.BW_Edge_Server  # seconds
        end_time = time + tx_time_sec * 1000  # ms
        end_time = end_time + self.latency_Edge_Server

        return end_time


class Client:
    # Send request client to edge
    def sendRequest(self, time, userId):
        buffer_in_sec = float(User.get_buffer(userId)) / 1000
        segment_duration = User.get_segment_duration(userId)
        segment_duration_in_sec = Utils.mapIndexToSegDuration(segment_duration)

        # To send a request, the buffer should have space for another segment
        if buffer_in_sec + segment_duration_in_sec > max_buffer:
            # player wait until it has space
            time = time + (buffer_in_sec + segment_duration_in_sec - max_buffer) * 1000
            User.set_buffer(userId, (max_buffer-segment_duration_in_sec)*1000)

        Utils.set_req_time(userId, time)

        last_segment_number = User.get_segment_number(userId)
        User.set_segment_number(userId, last_segment_number + 1)
        segment_quality = User.get_segment_quality(userId)
        segment_duration = User.get_segment_duration(userId)

        Utils.set_historical_e2e_throughput(userId, Radio.throughputGivenTime(userId,time))

        # ABR algorithm
        segment_size = Utils.getSegmentSize(userId, last_segment_number + 1, segment_quality, segment_duration)

        Utils.set_historical_buffer(userId, buffer_in_sec)

        estimated_throughput = Utils.get_estimated_throughput(userId)  # Mbps
        # future_throughput_Client_Edge = radio.predict_next_throughput(userId, time, utils.mapIndexToSegDuration(segment_duration))
        # predicted_throughput = utils.get_predicted_throughput(userId, future_throughput_Client_Edge, throughput_Client_Edge, estimated_throughput)

        # Future throughput prediction, we predict future radio throughput and adjust the estimated throughput
        # We calculate the time that would have taken to sent the last segment with future network conditions,
        # Then use that factor to modify the estimated throughput
        # if mode == 'EADAS':
        #    est_throughput = predicted_throughput
        # else:
        #    est_throughput = estimated_throughput

        est_throughput = estimated_throughput

        ABR.set_harmonic_mean_values(userId, segment_size, est_throughput)

        if ABR_Algorithm == "SARA":
            next_segment_quality = ABR.SARA(userId, n_qualities, segment_duration, segment_quality, buffer_in_sec,
                                            last_segment_number + 1)
        elif ABR_Algorithm == "BBA":
            next_segment_quality = ABR.BBA(segment_quality, buffer_in_sec, n_qualities)
        elif ABR_Algorithm == "TBA":
            next_segment_quality = ABR.TBA(userId, n_qualities, est_throughput)

        # Markov
        r = User.get_segment_quality(userId)
        c = next_segment_quality
        User.add_markov(userId,r,c)

        User.set_segment_quality(userId, next_segment_quality)
        User.set_time_last_request(userId, time)
        # end ABR algorithm


        if trace_mode == 1:
            print("------------------------------------------")
            print("Client " + str(userId) + " send request at " + str(time))
            print("Segment number " + str(last_segment_number + 1) + " Segment quality " + str(
                next_segment_quality) + " Segment length " + str(segment_duration))
            print("------------------------------------------")

        # We measure the time to send the file through the radio interface
        RAN_tx_time = Radio.calculateTxTimeRAN(userId, time, request_size)
        end_time = time + RAN_tx_time * 1000  # ms
        end_time = end_time + User.get_latency(userId)

        return end_time

    # Control the buffer
    def receiveSegment(self, time, userId):

        Utils.set_rec_time(userId, time)
        Utils.estimateThroughput(userId, wasPrefetched[userId], latency_Server_Edge, User.get_latency(userId),
                                 BW_Server_Edge)
        Utils.set_historical_predicted_throughput(userId, Utils.get_estimated_throughput(userId))

        segment_number = User.get_segment_number(userId)
        segment_quality = User.get_segment_quality(userId)
        segment_duration = User.get_segment_duration(userId)
        segment_duration_sec = Utils.mapIndexToSegDuration(segment_duration)

        seg_size = float(Utils.getSegmentSize(userId, User.get_segment_number(userId), segment_quality,
                                              User.get_segment_duration(userId)))
        User.set_last_segment_throughput(userId, time, seg_size * 8)

        # Packet timer
        time_requested = User.get_time_last_request(userId)
        timer = segment_duration_sec * 1000 * 2

        # Timer expired and "no packet arrived"
        if time > (time_requested + timer):
            # If user is NOT in the buffer filling, he plays the video
            if User.get_buffer_filling(userId) == 0:
                # Buffer and stalls management
                buffer_value = User.get_buffer(userId) - timer

                if buffer_value < 0:
                    # Stall occur
                    print("Stalling event, n request: " + str(segment_number))
                    print("User and trace id: " + str(userId))
                    stall_duration = -buffer_value

                    User.add_rebuffering_acumulated(userId, stall_duration)
                    User.add_n_stalls(userId)

                    P1203_Evaluation.addStall(time, stall_duration,
                                              userId)  # Store stall durations and time for QoE ITU

                    User.set_buffer(userId, 0)
                    if trace_mode == 1:
                        print("[Packet received. Timer expired. Stall] Player rebuffering for " + str(
                            stall_duration) + " milliseconds. New value of buffer is: " + str(User.get_buffer(userId)))
                else:
                    # No stall
                    User.set_buffer(userId, buffer_value)
                    if trace_mode == 1:
                        print("[Packet received. Timer expired. No stall. Packet of " + str(
                            segment_duration_sec) + " seconds received, now buffer value is: " + str(
                            User.get_buffer(userId)))
            else:
                print("[Packet received at " + str(time) + ". Timer expired. Buffer filling phase")

            # We request again the same packet
            User.set_segment_number(userId, segment_number - 1)

            # timer expired at this time
            time = time_requested + timer

        # Timer no expired, normal functioning:
        else:
            # If user is NOT in the buffer filling, he plays the video
            if User.get_buffer_filling(userId) == 0:
                # Buffer and stalls management
                stall = (User.get_time_last_request(userId) + User.get_buffer(userId)) - time

                if stall < 0:
                    # Stall occur
                    stall_duration = -stall
                    # print("Stalling event, n request: " + str(segment_number))

                    User.add_n_stalls(userId)
                    User.add_rebuffering_acumulated(userId, stall_duration)

                    P1203_Evaluation.addStall(time, stall_duration, userId)
                    # Here we have to store stall durations and time for QoE ITU
                    User.set_buffer(userId, segment_duration_sec * 1000)
                    if trace_mode == 1:
                        print("[Packet received. Stall] Player rebuffering for " + str(
                            stall_duration) + " milliseconds. New value of buffer is: " + str(
                            User.get_buffer(userId)))
                else:
                    User.set_buffer(userId, (User.get_time_last_request(userId) + User.get_buffer(
                        userId)) - time + segment_duration_sec * 1000)
                    if trace_mode == 1:
                        print("[Packet received. No stall] Packet of " + str(
                            segment_duration_sec) + " seconds received, now buffer value is: " + str(
                            User.get_buffer(userId)))
            else:
                # If user is in the buffer filling, he do NOT plays the video
                User.set_buffer(userId, User.get_buffer(userId) + segment_duration_sec * 1000)
                if trace_mode == 1:
                    print(
                        "[Packet received. Buffer filling] Time last request was: " + str(
                            int(User.get_time_last_request(userId))) + " time receiced: " + str(
                            int(time)))
                    print("Packet of " + str(segment_duration_sec) + " seconds received, now buffer value is: " + str(
                        User.get_buffer(userId)))
                if User.get_buffer(userId) > 2 * segment_duration_sec * 1000:
                    # When there are two segments we start playing the video
                    User.set_buffer_filling(userId, False)
                    User.set_startup_delay(userId, time)

        return time


#####################################################################################

BW_Server_Edge = 100  # Mbps
latency_Server_Edge = 200  # one-way (ms)

request_size = 2  # KB
bitrate_ladder_type = 2  # 0 low 1 high 2 all 3 twenty qualities
if bitrate_ladder_type == 2:
    n_qualities = 6
elif bitrate_ladder_type == 0:
    n_qualities = 4
elif bitrate_ladder_type == 1:
    n_qualities = 4
else:
    n_qualities = 20

n_users = 100
n_videos = n_users
initial_segment = 1
initial_quality = 1
initial_segment_duration = 1  # Segment duration: 3 = 10 sec; 2 = 6 sec; 1 = 2 sec;
User.set_n_users(n_users)
User.init(n_users, initial_quality, initial_segment_duration, n_qualities)
initial_segment_size = Utils.getSegmentSize(0, initial_segment, initial_quality, initial_segment_duration)
max_buffer = 30  # Maximum buffer size in seconds

# Parameters of simulation
P1203_Evaluation.init(n_users)
ABR.init(n_users)
Utils.init(n_users, n_videos, bitrate_ladder_type)

automatic_run = False  # Run multiple simulations from script. Arguments are needed.
select_file = True  # Select directly the radio traces (True) or use a mobility pattern (False)

if automatic_run:
    radiofile_number = int(sys.argv[5])
    select_file = True
else:
    radiofile_number = 1  # In case we want to select the radio trace

# For selecting radio traces with different name (ECAS-ML paper) and write the results of that trace in a file
if radiofile_number < 10:
    payload = "00"
elif radiofile_number < 100:
    payload = "0"
else:
    payload = ""
radiofile = "R_" + payload + str(radiofile_number)
writefile = "TR_" + payload + str(radiofile_number) + ".txt"

if select_file:
    Radio.init(n_users, "LTE", "DATASET_file", radiofile_number, latency_Server_Edge, BW_Server_Edge)  # We select a specific trace
else:
    Radio.init(n_users, "LTE", "DATASET",
               "pedestrian", latency_Server_Edge, BW_Server_Edge)  # We select where the radio traces comes from "DATASET" or "NS3" and the mobility pattern

mode = 'Classic'  # 'Classic', 'EADAS', 'ECAS', 'korean'
EADAS_startup_time = 10  # When (s) EADAS will start working?
ABR_Algorithm = "SARA"  # BBA, TBA, SARA,
trace_mode = 2  # 0 all, 1 only when segment received, 2 no traces
clustering_policy = ""  # "Par" "Video" or "Thr"
clusters = 1  # number of clusters
premium_users = "All"  # All/Par Who is premium user and enjoy prefetching?

#  Prefetching policy:  0 no prefetching (default), 1 prefetching all, 2 prefetching the same, 3 prefetching same +1 and -1, 4 transrating, 5 markov, 6 ml, 7 SuperResolution
for userId in range(n_users):
    if premium_users == "Par":
        if userId + 1 % 2 == 0:
            User.set_prefetching_policy(userId, 0)
            User.set_subscription(userId, 1)
            # Only premium users has segment prefetching
            # Par users (0, 2, 4, 6...) are premium
    elif premium_users == "All":
        User.set_prefetching_policy(userId, 6)
        User.set_subscription(userId, 1)

# ECAS-ML parameters
if automatic_run:
    ss_penalty = float(sys.argv[1]) * 0.01  # We are facing some issues to pass 0.05, 0.1... so we sent 5, 10...
    stalls_penalty = float(sys.argv[2]) * 0.01
    threshold1 = int(sys.argv[3])
    threshold2 = int(sys.argv[4])  # + int(sys.argv[3])
else:
    ss_penalty = 0.2
    stalls_penalty = 0.2
    threshold1 = 2
    threshold2 = 4
EdgeAssisted.init(n_users, initial_segment_size, ss_penalty, stalls_penalty, threshold1, threshold2)

if clustering_policy == "Thr":
    for userId in range(n_users):
        step = n_users / clusters
        clusterId = int(userId / step)
        User.set_clusterId(userId, clusterId)
elif clustering_policy == "Video":
    for userId in range(n_users):
        User.set_clusterId(userId, userId % 3)
elif clustering_policy == "Par":
    for userId in range(n_users):
        if userId % 2 == 0:
            User.set_clusterId(userId, 1)

for userId in range(n_users):
    videoId = Utils.getRandomVideoId(n_videos)
    User.set_videoId(userId, videoId)  # 0 = Big Buck Bunny, 1 = Elephants Dream, 2 = Tears of Steel, 3 = fixed dataset
    resolutions = ["426x240", "640x360", "850x480", "1280x720"]  # 1920x1080, 3840x2160
    screen_resolution = resolutions[userId % 4]
    screen_resolution = "1920x1080"  # fixed in 4K 2048x1080 "3840x2160"
    User.set_screen_resolution(userId, screen_resolution)

Utils.showVideoPopularity(n_users)
backhaul_data_sent_KB = 0
needed_data_sent_KB = 0
prefetched_hits = 0
caching_hits = 0
possible_hits = 0

## Cache
n_seg = 250
Utils.createCache(n_users, n_videos, n_seg, n_qualities)
cachingPolicy = "LFU"

# Start
print("---------------------------------------")
print("----------WELCOME TO ANGELA------------")
print("Simulation with mode: " + str(mode))
print("ABR algorithm: " + ABR_Algorithm)
print("---------------------------------------")


# List of bitrates requested per client
historical_throughput = []
prefetch_time = [0] * n_users  # We save here time when last segment arrived at the edge for do prefetching
wasPrefetched = [False] * n_users  # Save if last segment of one user was prefetched or not
first_segment = [True] * n_users
for x in range(n_users):
    historical_throughput.append([])

server = Server(BW_Server_Edge, latency_Server_Edge)
edge = Edge(BW_Server_Edge, latency_Server_Edge)
client = Client()

#  Initialize data
time = [0] * n_users  # in ms
if initial_segment_duration == 1:
    n_segments = 227  # Limited by OfForestAndMen video
    # n_segments = 285  # big buck bunny
elif initial_segment_duration == 2:
    n_segments = 76
elif initial_segment_duration == 3:
    n_segments = 46

QoEmetadata = [""] * n_users
for x in range(n_users):
    QoEmetadata[x] = P1203_Evaluation.generateInitialInfo()
    P1203_Evaluation.addStall(0, 0, x)
metadata_start = [0] * n_users

# Main loop
for x in range(n_segments - 1):
    for userId in range(n_users):
        time[userId] = client.sendRequest(time[userId], userId)
    for userId in range(n_users):
        time[userId] = edge.sendRequest2Server(time[userId], userId)
    for userId in range(n_users):
        time[userId] = server.sendSegment2Edge(time[userId], userId)
    for userId in range(n_users):
        time[userId] = edge.sendPacket2Client(time[userId], userId)
    for userId in range(n_users):
        time[userId] = client.receiveSegment(time[userId], userId)

# P1203 mode 0 generation
for x in range(n_users):
    last_info = P1203_Evaluation.generateFinalInfo(x)
    QoEmetadata[x] = QoEmetadata[x] + last_info
    filename = P1203_Evaluation.writefile(QoEmetadata[x], x)
    P1203_Evaluation.executeEvaluation(str(filename), x)

# Utils.plotGraph(0)
Utils.CalculateFairness(n_users, clustering_policy)
Utils.stats(n_users)
#print("Stored data in KB: " + str(backhaul_data_sent_KB))
#print("Needed data in KB: " + str(needed_data_sent_KB))
#print("Wasted data in KB: " + str(backhaul_data_sent_KB - needed_data_sent_KB))

possible_hits -= 1
#print("Prefetched hits: " + str(prefetched_hits))
#print("Caching hits: " + str(caching_hits))
print("Wasted data in KB (%): " + str((backhaul_data_sent_KB-needed_data_sent_KB)*100/(backhaul_data_sent_KB) ))
print("Prefetched hits (%): " + str(prefetched_hits*100/possible_hits))
print("Caching hits (%): " + str(caching_hits*100/possible_hits))
qoe = Utils.parseQoE(n_users)
Utils.write_data(ss_penalty, stalls_penalty, threshold1, threshold2, qoe, writefile)
# Utils.ECAS_stats(n_users)
#Utils.writeDataset(n_users)
#Utils.writeMLdata(n_users)