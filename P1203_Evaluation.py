from datetime import date
from datetime import datetime
import subprocess
from subprocess import Popen
import os

import User

stallings = []
first_Stall = []


def init(n_users):
    global stallings
    global first_Stall
    stallings = [""] * n_users
    first_Stall = [True] * n_users


def addStall(time, duration, userId):
    global first_Stall
    global stallings

    time_s = time / 1000
    duration_s = duration / 1000

    if first_Stall[userId]:
        # do not add ", "
        stallings[userId] = stallings[userId] + "[" + str(time_s) + ", " + str(duration_s) + "]"
    else:
        stallings[userId] = stallings[userId] + ", [" + str(time_s) + ", " + str(duration_s) + "]"

    first_Stall[userId] = False


def generateInitialInfo():
    text = "{" + '\n' \
           + "    \"I11\": {" + '\n' \
           + "     \"_comment\": \" This is for audio\"," + '\n' \
           + "        \"segments\": []," + '\n' \
           + "        \"streamID\": 42 " + '\n' \
           + "    }," + '\n' \
           + "    \"I13\": {" + '\n' \
           + "     \"_comment\": \" This is for video\"," + '\n' \
           + "        \"segments\": [" + '\n'
    return text


def segmentGenerator(bitrate, codec, duration, fps, resolution, start, first_segment):
    if first_segment:
        text = "            {" + '\n' + "              \"bitrate\": " + bitrate + "," + '\n' + "              \"codec\": " + codec + "," + '\n' + "              \"duration\": " + duration + "," + '\n' + "              \"fps\": " + fps + "," + '\n' + "              \"resolution\": " + resolution + "," + '\n' + "              \"start\": " + start + '\n' + "          }" + '\n'
    else:
        text = "            ,{" + '\n' \
               + "              \"bitrate\": " + bitrate + "," + '\n' \
               + "              \"codec\": " + codec + "," + '\n' \
               + "              \"duration\": " + duration + "," + '\n' \
               + "              \"fps\": " + fps + "," + '\n' \
               + "              \"resolution\": " + resolution + "," + '\n' \
               + "              \"start\": " + start + '\n' \
               + "          }" + '\n'
    return text


def generateFinalInfo(userId):
    global stallings
    text = "        ]," + '\n' \
           + "        \"streamId\": 42" + '\n' \
           + "    }," + '\n' \
           + "    \"I23\": {" + '\n' \
           + "        \"stalling\": [" + '\n' \
           + stallings[userId] \
           + "]," + '\n' \
           + "        \"streamId\": 42" + '\n' \
           + "    }," + '\n' \
           + "    \"IGen\": {" + '\n' \
           + "        \"device\": \"mobile\"," + '\n' \
           + "        \"displaySize\": \"" \
           + User.get_screen_resolution(userId) + "\"," + '\n' \
           + "        \"viewingDistance\": \"50cm\"" + '\n' \
           + "    }" + '\n' \
           + "}" + '\n'
    return text


def writefile(mainText, userId):
    today = date.today()
    name_date = today.strftime("%b-%d-%Y")
    now = datetime.now()
    current_time = now.strftime("%H-%M-%S")

    # filename = "TestbedEvaluation-" + name_date + "-" + current_time + ".json"
    filename = "TestbedEvaluation-" + str(userId) + ".json"

    f = open(filename, "w")
    f.write(mainText)
    f.close()
    return filename


def executeEvaluation(filename, userId):
    """
    Evaluation ITU P1203
    go to ITU P1203 file
    execute: ./calculate.py TestbedEvaluation-Jan-01-2021-18-50-04.json >> output.txt
    or get-ouput.sh


    ./itu-p1203-codecextension/calculate.py TestbedEvaluation-Jan-01-2021-17-49-32.json >> output.txt
    """
    print("---")
    print(filename)
    print("---")
    output = "output" + str(userId) + ".txt"
    subprocess.call(['sh', './QoEEvaluation.sh', filename, output])
    #subprocess.call(['sh', './itu-p1203-codecextension/get-output.sh'])
    # subprocess.call(["./get-output.sh", filename], shell=True)

    # os.system('./get-output.sh {}'.format(filename))
    #os.system("python3 /home/ubuntu/itu-p1203-codecextension/calculate.py TestbedEvaluation.json >> output.txt")

    # Parameters for segment generation

"""
bitrate = [str(18000), str(9000), str(5400), str(1800)]
codec = "\"hevc\""
duration = str(1)
fps = str(30.0)
resolution = ["\"3840x2160\"", "\"1920x1080\"", "\"1280x720\"", "\"720x480\""]

initial_text = generateInitialInfo()
segment1 = segmentGenerator(bitrate[1], codec, duration, fps, resolution[1], str(1), False)
segment2 = segmentGenerator(bitrate[1], codec, duration, fps, resolution[1], str(0), True)
last_info = generateFinalInfo()

mainText = initial_text + segment1 + segment2 + last_info

filename = writefile(mainText)

executeEvaluation(str(filename))
"""
