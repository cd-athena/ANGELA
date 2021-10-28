user = []
n_users = 0


def create_list(n_users, initial_quality, initial_segment_duration):
    segment_number = 0
    segment_quality = initial_quality
    segment_duration = initial_segment_duration
    latency = 50  # ms
    bitrate = 52  # kbps
    time_last_request = 0  # ms
    buffer = 0  # ms
    buffer_target = 8000  # ms
    buffer_filling_phase = True
    clusterId = 0
    videoId = 0  # 0 = Big Buck Bunny, 1 = Elephants Dream, 2 = Tears of Steel
    rebuffering_accumulated = 0
    startup_delay = 0
    n_stalls = 0
    bitrate_changes = 0
    subscription = 0  # 0 normal, 1 premium
    prefetching_policy = 0
    screen_resolution = ""
    last_segment_thoughput = 0
    tracenumber = 0

    for x in range(n_users):
        user.append([segment_number, segment_quality, segment_duration, latency, bitrate, time_last_request, buffer,
                     buffer_target, buffer_filling_phase, clusterId, videoId, rebuffering_accumulated, startup_delay,
                     n_stalls, bitrate_changes, subscription, prefetching_policy, screen_resolution, last_segment_thoughput, tracenumber])


def get_n_users():
    return n_users


def set_n_users(value):
    global n_users
    n_users = value


def get_segment_number(userId):
    return user[userId][0]


def set_segment_number(userId, value):
    user[userId][0] = value


def get_segment_quality(userId):
    return user[userId][1]


def set_segment_quality(userId, value):
    user[userId][1] = value


def get_segment_duration(userId):
    return user[userId][2]


def set_segment_duration(userId, value):
    user[userId][2] = value


def get_latency(userId):
    return user[userId][3]


def get_bitrate(userId):
    return user[userId][4]


def set_bitrate(userId, value):
    user[userId][4] = value


def get_time_last_request(userId):
    return user[userId][5]


def set_time_last_request(userId, value):
    user[userId][5] = value


def get_buffer(userId):
    return user[userId][6]


def set_buffer(userId, value):
    user[userId][6] = value


def get_buffer_target(userId):
    return user[userId][7]


def set_buffer_target(userId, value):
    user[userId][7] = value


def get_buffer_filling(userId):
    return user[userId][8]


def set_buffer_filling(userId, value):
    user[userId][8] = value


def get_clusterId(userId):
    return user[userId][9]


def set_clusterId(userId, value):
    user[userId][9] = value


def get_videoId(userId):
    return user[userId][10]


def set_videoId(userId, value):
    user[userId][10] = value


def get_rebuffering_acumulated(userId):
    return user[userId][11]


def add_rebuffering_acumulated(userId, value):
    user[userId][11] += value


def get_startup_delay(userId):
    return user[userId][12]


def set_startup_delay(userId, value):
    user[userId][12] += value


def get_n_stalls(userId):
    return user[userId][13]


def add_n_stalls(userId):
    user[userId][13] += 1


def get_bitrate_changes(userId):
    return user[userId][14]


def add_bitrate_changes(userId, value):
    user[userId][14] += value


def get_subscription(userId):
    return user[userId][15]


def set_subscription(userId, value):
    user[userId][15] = value


def get_prefetching_policy(userId):
    return user[userId][16]


def set_prefetching_policy(userId, value):
    user[userId][16] = value


def get_screen_resolution(userId):
    return user[userId][17]


def set_screen_resolution(userId, value):
    user[userId][17] = value


def get_last_segment_throughput(userId):
    return user[userId][18]


def set_last_segment_throughput(userId, time_arrived, seg_size):
    value = seg_size / ((time_arrived-get_time_last_request(userId))*0.001)

    user[userId][18] = value


def get_tracenumber(userId):
    return user[userId][19]


def set_tracenumber(userId, value):
    user[userId][19] = value