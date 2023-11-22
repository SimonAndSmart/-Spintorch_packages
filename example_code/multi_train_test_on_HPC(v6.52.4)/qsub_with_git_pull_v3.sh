#!/bin/bash

mkdir job_outs
mkdir job_outs/job_err
chmod +x copy_result*
# Git operations
if [ -d "Spintorch_packages" ]; then
    cd Spintorch_packages
    git pull
    cd ..
else
    git clone https://github.com/SimonAndSmart/Spintorch_packages.git --depth 1 --branch main --single-branch --no-tags
    echo "Done!"
fi

# Submit the job array
pbs_file=$(ls *.pbs)
qsub $pbs_file

