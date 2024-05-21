
module load miniconda/22.11.1_environmentally && conda activate CLL_2024
cd repos/MM_2023/ && python infer_annotaion.py

###### gpu
bsub -q gpu-medium -gpu num=1:j_exclusive=yes:gmem=30000 -R "rusage[mem=210000]" "module load miniconda/22.11.1_environmentally; conda activate CLL_2024;cd repos/MM_2023/; python infer_annotaion.py"
###### cpu
bsub -q new-medium -n 10 -R "rusage[mem=2000] span[hosts=1]" "module load miniconda/22.11.1_environmentally; conda activate CLL_2024;cd repos/MM_2023/; export JUPYTER_ALLOW_INSECURE_WRITES=True; jupyter notebook --no-browser --port 28889 --ip='0.0.0.0'"
bsub -q new-medium -n 10 -R "rusage[mem=2000] span[hosts=1]" "module load miniconda/22.11.1_environmentally; conda activate CLL_2024;cd repos/MM_2023/;  python infer_annotaion.py"
