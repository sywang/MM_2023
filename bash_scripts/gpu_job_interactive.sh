bsub \
-J 'GPU jupyter server' \
-R rusage[mem=20000] \
-gpu num=1:j_exclusive=yes \
-q gpu-interactive \
-Is /bin/bash


