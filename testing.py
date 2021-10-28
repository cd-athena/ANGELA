from matplotlib import pyplot as plt
import Radio
from statistics import mean

historical_bitrates_requested = []
historical_mean_rate_kbps = []

def init(nusers):
    for x in range(nusers):
        historical_bitrates_requested.append([])  # Plot purposes # kbps
        historical_mean_rate_kbps.append([])  # Plot purposes

def set_historical_bitrate(userId, bitrate):
    historical_bitrates_requested[userId].append(bitrate)

# Plot purposes only
def set_historical_mean_rate_kbps(userId, data_rate):
    historical_mean_rate_kbps[userId].append(data_rate)

def CalculateFairness(n_users):
    n_requests = len(historical_bitrates_requested[0])
    fairness_over_time = [0] * n_requests

    for requestId in range(n_requests):
        user_counter = 0
        sum_xi = 0
        sum_square_xi = 0

        for userId in range(n_users):
            Ti = historical_bitrates_requested[userId][requestId]
            Oi = historical_mean_rate_kbps[userId][requestId]
            Xi = Ti / Oi

            if Xi != 0:
                user_counter = user_counter + 1
                sum_xi = sum_xi + Xi
                sum_square_xi = sum_square_xi + Xi ** 2

        if sum_xi == 0 or user_counter == 0:
            fairness_over_time[requestId] = 0
        else:
            fairness_over_time[requestId] = (sum_xi ** 2) / (user_counter * sum_square_xi)

    # Mean fairness score for the session
    print("Mean fairness index for the whole session" + str(mean(fairness_over_time)))

    #plt.plot(fairness_over_time)
    plt.show()

init(3)

set_historical_bitrate(0, 1200)
set_historical_bitrate(1, 1200)
set_historical_bitrate(2, 1200)

set_historical_mean_rate_kbps(0, 2000)
set_historical_mean_rate_kbps(1, 4000)
set_historical_mean_rate_kbps(2, 6000)

CalculateFairness(3)