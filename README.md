# eecs545_artificial_curiosity
UofM EECS 545 Fall 2020 Project: Domain Exploration Through Artificial Curiosity

# Modifying the dockerfile
The original docker image can be accessed with a `docker pull tsender/tensorflow-gpu` (note, you may need to login to pull the image). If you  make any changes to the dockerfile to run it properly on your own system, you maydo so. After you make your changes, just build the dockerfile on your system with your docker username and the tag `latest-gpu`

    cd dockerfiles/
    docker build -t <username>/tensorflow:latest-gpu .
    
Note 1: This dockerfile includes openCV and you may want to lower the number of threads to use so it doesn't use all of your CPU power when building.
Note 2: I (Ted) will be running this exact dockerfile. If you need additional dependencies, let me know so I can add them.

# How to run the code with the shell script
Place the provided sample shell script in the SAME level as this project folder on your system. Then, run the docker file with `sh run_tf_docker_sample.sh eecs545_artificial_curiosity`. The shell script handles volume mapping for various directories. Depending on your system, you may need to remove/add some volumes.
