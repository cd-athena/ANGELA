#!/bin/bash

# Move file to itu-p1203-codecextension
#mv $1 /home/ubuntu/itu-p1203-codecextension

# First argument, name of the file like "TestbedEvaluation-Jan-01-2021-18-54-30.json "
#cd ..
#cd ..
#cd itu-p1203-codecextension
#./calculate.py TestbedEvaluation-Jan-01-2021-18-54-30.json >> output.txt
#./calculate.py $1 >> P1203_Evaluation_Output.txt
#python calculate.py $1 >> P1203_Evaluation_Output.txt

# get-output.sh TestbedEvaluation-Jan-11-2021-16-32-52.json

#./calculate.py TestbedEvaluation.json >> output.txt


mv $1 /home/ubuntu/PycharmProjects/Testbed/itu-p1203-codecextension

cd itu-p1203-codecextension

rm $2

./calculate.py $1 >> $2



