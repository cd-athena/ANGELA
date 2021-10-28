import Radio
import random
import User
import Utils
import math
import csv

#  Simulation data
devices_support_4k = 100  # percentage

last_segment_size_requested = []
n_users = 0

repeat_quality = []

# ECAS parameters
switches_penalty_factor = 0
buffer_penalty_factor = 0
threshold1 = 0  # number of segments
threshold2 = 0  # number of segments


def init(nusers, bitrate_zero, ss_penalty, stalls_penalty, th1, th2):
    global n_users
    n_users = nusers
    global last_segment_size_requested
    global repeat_quality
    repeat_quality = [False] * n_users
    for x in range(n_users):
        last_segment_size_requested = [bitrate_zero] * n_users
    global switches_penalty_factor
    switches_penalty_factor = ss_penalty
    global buffer_penalty_factor
    buffer_penalty_factor = stalls_penalty
    global threshold1
    threshold1 = th1  # number of segments
    global threshold2
    threshold2 = th2  # number of segments


def set_last_segment_size_requested(userId, value):
    last_segment_size_requested[userId] = value  # Last segment_size requested in KB
    # Only for fairness


# This function normalizes values in a scale 0 to 10, for example (2.5, 0, 5) will return 5
def normalize(value, min_value, max_value):
    dif = min_value - 0
    max_value = max_value - dif
    value = value - dif
    norm_value = value * 10 / max_value

    return norm_value


def JainFairnessIndex(userId, quality_index, segment_size_ladder, clusterId):
    segment_size_to_evaluate = float(segment_size_ladder[quality_index])  # Segment size to evaluate KB
    set_last_segment_size_requested(userId, segment_size_to_evaluate)

    # Jain fairness index
    sum_xi = 0
    sum_square_xi = 0
    nusers_in_cluster = 0

    for x in range(n_users):
        if User.get_clusterId(x) == clusterId:
            nusers_in_cluster += 1
            sum_xi = sum_xi + (last_segment_size_requested[x])
            sum_square_xi = sum_square_xi + (last_segment_size_requested[x]) ** 2

    fairness_index = (sum_xi ** 2) / (nusers_in_cluster * sum_square_xi)

    return fairness_index


def EADAS(time, userId, n_qualities, ABR_quality_index, segment_size_ladder):
    alpha = 0.5
    chosen_quality_score = 0
    chosen_quality = 0

    max_quality_to_evaluate = n_qualities
    #  max_quality_to_evaluate = ABR_quality_index + 1

    if max_quality_to_evaluate >= n_qualities:
        max_quality_to_evaluate = n_qualities

    for x in range(max_quality_to_evaluate):

        total_throughput = Radio.throughputGivenTime(userId, time)

        # First test: avoid stallings
        estimated_download = (segment_size_ladder[x] * 8 / 1000) / radio_throughput
        current_buffer = User.get_buffer(userId)
        # If we don't go under safe buffer value
        if (current_buffer - estimated_download * 1000) > (User.get_buffer_target(userId) / 2):

            step = 1 / max_quality_to_evaluate
            n_steps = abs(ABR_quality_index - x)

            QoE_score = 1 - (n_steps * step)

            clusterId = User.get_clusterId(userId)

            fairness_index = JainFairnessIndex(userId, x, segment_size_ladder, clusterId)

            final_score = QoE_score * alpha + fairness_index * (1 - alpha)

            if final_score > chosen_quality_score:
                chosen_quality = x
                chosen_quality_score = final_score

    return chosen_quality


def get_ECAS_parameters(time, userId):

    tracenumber = User.get_tracenumber(userId)

    filename = "PredTrace_" + str(tracenumber) + ".txt"
    csv_file = open(filename, 'r')
    l = [0,0,0,0]
    with csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            time_to_length = int(time*0.001)
            if time_to_length == int(row[0]):
                switches_penalty = row[1]
                stalls_penalty = row[2]
                th1 = row[3]
                th2 = row[4]

                l[0] = switches_penalty
                l[1] = stalls_penalty
                l[2] = th1
                l[3] = th2

    csv_file.close()
    return l


def ECAS(userId, time, n_qualities, CN_BW, CN_latency):

    param_over_time = False
    if time > 5000 and param_over_time:
        l = get_ECAS_parameters(time, userId)
        switches_penalty_factor = float(l[0])
        buffer_penalty_factor = float(l[1])
        threshold1 = float(l[2])  # number of segments
        threshold2 = float(l[3])  # number of segments
    else:
        switches_penalty_factor = 0.9
        buffer_penalty_factor = 0.2
        threshold1 = 4  # number of segments
        threshold2 = 7  # number of segments

    next_quality = 0  # Quality index to return
    n_qualities_window = 9

    # Summatory bitrate last n_qualities_window qualities (or less if there are less requests)
    mean_bitrate_last = 0
    n_seg_req = Utils.get_n_segments_requested(userId)
    n_seg = min(n_qualities_window, n_seg_req)
    for x in range(n_seg):
        mean_bitrate_last += Utils.get_historical_bitrate(userId, n_seg_req - 1 - x)
    if n_seg == 0:
        mean_bitrate_last = 0
    else:
        mean_bitrate_last = mean_bitrate_last / n_seg

    #  Evaluate all the qualities
    for qid in range(n_qualities):
        seg_size = float(
            Utils.getSegmentSize(userId, User.get_segment_number(userId), qid, User.get_segment_duration(userId)))  # KB
        bitrate = (seg_size * 8) / Utils.mapIndexToSegDuration(User.get_segment_duration(userId))

        # Screen properties
        screen_resolution = User.get_screen_resolution(userId)
        if screen_resolution == "426x240":
            beta = 8.17
        elif screen_resolution == "640x360":
            beta = 3.73
        elif screen_resolution == "850x480":
            beta = 2.75
        elif screen_resolution == "1280x720":
            beta = 1.89
        elif screen_resolution == "1920x1080":
            beta = 0.78
        elif screen_resolution == "3840x2160":
            beta = 0.5

        QoEm = bitrate
        x = bitrate * 0.001
        screen_factor = 1 - math.exp(-beta * x)
        QoEs = QoEm * screen_factor

        # Mean bitrate including the quality we are evaluating
        if n_seg == 0:
            mean_bitrate = 0
        else:
            mean_bitrate = ((mean_bitrate_last * n_seg) + bitrate) / n_seg + 1

        # Switches penalty calculation
        # switches_penalty = abs(mean_bitrate_last - bitrate) * switches_penalty_factor
        switches_penalty = abs(mean_bitrate - bitrate) * switches_penalty_factor

        # Delivery time calculation
        # this could be easily done with estimated throughput, but you want to use the future throughput prediction
        tx_time_request_CE = Radio.calculateTxTimeRAN(userId, time, 1*8, CN_latency, CN_BW)  # Assuming 1 KB request
        tx_time_request_ES = 1 * 8 / CN_BW  # Assuming 1 KB request
        tx_time_segment_CE = Radio.calculateTxTimeRAN(userId, time, seg_size * 8, CN_latency, CN_BW)
        tx_time_segment_ES = seg_size * 8 / CN_BW

        # If the quality is gonna be prefetched, the delivery time is lower
        last_q = Utils.get_last_historical_qindex(userId)
        prefetching_policy = User.get_prefetching_policy(userId)
        if prefetching_policy == 0:
            isPrefetched = False
        elif prefetching_policy == 1:
            isPrefetched = True
        elif prefetching_policy == 2:
            if qid == last_q:
                isPrefetched = True
            else:
                isPrefetched = False
        elif prefetching_policy == 3:
            if qid == last_q or qid == last_q + 1 or qid == last_q - 1:
                isPrefetched = True
            else:
                isPrefetched = False

        if isPrefetched:
            delivery_time = 2 * User.get_latency(userId) + tx_time_request_CE + tx_time_segment_CE
        else:
            delivery_time = 2 * User.get_latency(
                userId) + 2 * CN_latency + tx_time_request_CE + tx_time_request_ES + tx_time_segment_CE + tx_time_segment_ES

        # Predicted buffer size (ms) when the segment arrives
        seg_duration_ms = Utils.mapIndexToSegDuration(User.get_segment_duration(userId)) * 1000
        predicted_buffer = User.get_buffer(userId) - delivery_time + seg_duration_ms

        # Buffer zone 1: high risk of stall
        if predicted_buffer < seg_duration_ms * threshold1:
            QoE_score = -999
        # Buffer zone 2: medium risk of stall
        elif predicted_buffer < seg_duration_ms * threshold2:
            buffer_difference = (seg_duration_ms * threshold2 - predicted_buffer) * 0.001  # in s
            buffer_penalty = buffer_difference * mean_bitrate * buffer_penalty_factor
            QoE_score = QoEs - (switches_penalty + buffer_penalty)
        # Buffer zone 3: low risk of stall
        else:
            QoE_score = QoEs - switches_penalty

        if qid == 0:
            best_score = QoE_score

        if QoE_score > best_score:
            next_quality = qid
            best_score = QoE_score
            best_score_predicted_buffer = predicted_buffer

    # print("Quality is: " + str(next_quality) + " predicted buffer is: " + str(best_score_predicted_buffer))
    Utils.set_historical_QoE_score(userId, best_score)
    return next_quality


#  Yan QoE is defined in the paper: "A Control-theoretic Approach for Dynamic Adaptive Video Streaming over HTTP"
def YanQoE(userId, bitrate, radio_throughput):
    QoE = 0
    average_video_quality = 0
    average_quality_variations = 0
    rebuffering = 0
    startup_delay = 0

    quality_variations_weight = 0  # does not make sense, I have less switches and more bitrate with a value of 0 than a value of 1.. [to review]
    rebuffering_weight = 3000
    startup_delay_weight = 3000

    n_seg = Utils.get_n_segments_requested(userId)

    # Average Video Quality
    average_video_quality = 0
    for x in range(n_seg):
        average_video_quality += Utils.get_historical_bitrate(userId, x)
    # average_video_quality = average_video_quality / n_seg
    average_video_quality += bitrate  # Evaluation next quality

    # Average Quality Variations
    for x in range(n_seg - 1):
        average_quality_variations += abs(
            Utils.get_historical_bitrate(userId, x) - Utils.get_historical_bitrate(userId, x + 1))
    # average_quality_variations = average_quality_variations / (n_seg-1)
    if (n_seg - 1) >= 0:
        average_quality_variations += abs(
            bitrate - Utils.get_historical_bitrate(userId, n_seg - 1))  # Evaluation next quality

    # Total rebuffering time
    rebuffering = User.get_rebuffering_acumulated(userId) / 1000  # ms to s

    # Evaluation next quality
    current_buffer = User.get_buffer(userId)
    segment_size_Mbps = bitrate * Utils.mapIndexToSegDuration(User.get_segment_duration(userId)) / 1000
    estimated_download = segment_size_Mbps / radio_throughput  # seconds
    # we consider "rebuffering" if it goes below the safe level
    buffer_status = (current_buffer + User.get_buffer_target(userId)) - estimated_download * 1000
    if buffer_status < 0:
        added_rebuffering = -buffer_status
        rebuffering += (added_rebuffering / 1000)  # ms to s

    # Startup delay
    startup_delay = User.get_startup_delay(userId) / 1000  # ms to s

    QoE = average_video_quality - quality_variations_weight * average_quality_variations - rebuffering_weight * rebuffering - startup_delay_weight * startup_delay

    return QoE


# https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9308916
# Edge Computing Assisted Adaptive Streaming Scheme for Mobile Networks - Minsu Kim and Kwangsue Chung
def KoreanEdgeBasedAlgorithm(userId, time, n_qualities):
    # r_avg: average bitrate for the client i up to the kth segment
    mean_bitrate_last = 0
    n_seg_req = Utils.get_n_segments_requested(userId)
    for x in range(n_seg_req):
        mean_bitrate_last += Utils.get_historical_bitrate(userId, n_seg_req - 1 - x)
    r_avg = mean_bitrate_last / n_seg_req  # kbps

    # Thrseg: segment throughput for the client i after receiving the kth segment
    thr_seg = User.get_last_segment_throughput(userId)  # kbps

    # r_thr_max: is the maximum bitrate which is lower than the segment throughput
    r_thr_max = 0  # kbps
    # r_buf_max is the maximum bitrate which does not cause the stalling events
    r_buf_max = 0  # kbps

    for qid in range(n_qualities):
        seg_size = float(Utils.getSegmentSize(userId, User.get_segment_number(userId), qid, User.get_segment_duration(userId))) * 8 # kb
        r = (seg_size) / Utils.mapIndexToSegDuration(User.get_segment_duration(userId))  # kbps
        if r < thr_seg:
            r_thr_max = r

        buffer_size = User.get_buffer(userId) * 0.001  # s
        if (seg_size/thr_seg) < buffer_size:
            r_buf_max = r

    # delta_s: switching threshold
    delta_s = abs(Utils.get_historical_bitrate(userId, n_seg_req - 1) - r_buf_max)

    # delta_uf: unfairness threshold
    r_avg_sus = (r_thr_max + r_buf_max)/2

    r_avg_users = 0
    for uid in range(n_users):
        r_avg_users += Utils.get_historical_bitrate(uid, Utils.get_n_segments_requested(uid) - 1)
    r_avg_users = r_avg_users / n_users

    seg_size = float(Utils.getSegmentSize(userId, User.get_segment_number(userId), n_qualities - 1, User.get_segment_duration(userId)))  # KB
    r_max = (seg_size * 8) / Utils.mapIndexToSegDuration(User.get_segment_duration(userId))
    seg_size = float(Utils.getSegmentSize(userId, User.get_segment_number(userId), 0, User.get_segment_duration(userId)))  # KB
    r_min = (seg_size * 8) / Utils.mapIndexToSegDuration(User.get_segment_duration(userId))

    delta_uf = abs(r_avg_sus - r_avg_users) / (r_max - r_min)

    # last bitrate
    r_last = Utils.get_historical_bitrate(userId, Utils.get_n_segments_requested(userId) - 1)
    q_last = Utils.get_last_historical_qindex(userId)
    q_next = 0
    # For each bitrate r - R in decreasing order
    for qid in range(n_qualities-1, -1, -1):
        seg_size = float(Utils.getSegmentSize(userId, User.get_segment_number(userId), qid, User.get_segment_duration(userId)))  # KB
        r = (seg_size * 8) / Utils.mapIndexToSegDuration(User.get_segment_duration(userId))  # kbps
        # If allocation of r satifes the resource constraint. (Lower than the radio throughput)
        radio_throughput_kbps = Radio.radioThroughputGivenTime(userId, time) * 1000
        if r < radio_throughput_kbps and r <= max(r_thr_max, r_buf_max):
            if abs(r - r_last) <= delta_s:
                if qid > q_last:
                #if r > r_last:
                    # Determine delta_uf to the fixed value
                    delta_uf = 0.5
                    if abs(r - r_avg_users)/(r_max-r_min) <= delta_uf:
                        q_next = qid
                        break
                if qid == q_last and abs(r - r_avg_users)/(r_max-r_min) <= delta_uf:
                #if r == r_last and abs(r - r_avg_users) / (r_max - r_min) <= delta_uf:
                    q_next = qid
                    break
                if qid < q_last:
                #if r < r_last:
                    if r > r_avg:
                        q_next = qid
                        break
                    else:
                        # I add these two lines...
                        q_last = Utils.get_last_historical_qindex(userId)
                        if q_last <= qid:
                            q_next = q_last
                        break
    if q_next == 0:
        for qid in range(n_qualities-1, -1, -1):
            seg_size = float(Utils.getSegmentSize(userId, User.get_segment_number(userId), qid,
                                                  User.get_segment_duration(userId)))  # KB
            r = (seg_size * 8) / Utils.mapIndexToSegDuration(User.get_segment_duration(userId))
            radio_throughput_kbps = Radio.radioThroughputGivenTime(userId, time) * 1000
            if r < radio_throughput_kbps and r <= max(r_thr_max, r_buf_max):
                q_next = qid
                break
    if q_next == 0:
        for qid in range(n_qualities-1, -1, -1):
            seg_size = float(Utils.getSegmentSize(userId, User.get_segment_number(userId), qid,
                                                  User.get_segment_duration(userId)))  # KB
            r = (seg_size * 8) / Utils.mapIndexToSegDuration(User.get_segment_duration(userId))
            if r < thr_seg:
                q_next = qid
                break
    return q_next


