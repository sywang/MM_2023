bsub \
-J 'CPU jupyter server' \
-R rusage[mem=10000] \
-q interactive \
-Is /bin/bash


