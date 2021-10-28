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

cd itu-p1203-codecextension

./calculate.py TestbedEvaluation-0.json >> output0.txt
./calculate.py TestbedEvaluation-1.json >> output1.txt
./calculate.py TestbedEvaluation-2.json >> output2.txt
./calculate.py TestbedEvaluation-3.json >> output3.txt
./calculate.py TestbedEvaluation-4.json >> output4.txt
./calculate.py TestbedEvaluation-5.json >> output5.txt
./calculate.py TestbedEvaluation-6.json >> output6.txt
./calculate.py TestbedEvaluation-7.json >> output7.txt
./calculate.py TestbedEvaluation-8.json >> output8.txt
./calculate.py TestbedEvaluation-9.json >> output9.txt
./calculate.py TestbedEvaluation-10.json >> output10.txt
./calculate.py TestbedEvaluation-11.json >> output11.txt
./calculate.py TestbedEvaluation-12.json >> output12.txt
./calculate.py TestbedEvaluation-13.json >> output13.txt
./calculate.py TestbedEvaluation-14.json >> output14.txt
./calculate.py TestbedEvaluation-15.json >> output15.txt
./calculate.py TestbedEvaluation-16.json >> output16.txt
./calculate.py TestbedEvaluation-17.json >> output17.txt
./calculate.py TestbedEvaluation-18.json >> output18.txt
./calculate.py TestbedEvaluation-19.json >> output19.txt
./calculate.py TestbedEvaluation-20.json >> output20.txt
./calculate.py TestbedEvaluation-21.json >> output21.txt
./calculate.py TestbedEvaluation-22.json >> output22.txt
./calculate.py TestbedEvaluation-23.json >> output23.txt
./calculate.py TestbedEvaluation-24.json >> output24.txt

