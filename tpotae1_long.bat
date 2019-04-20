#!/bin/bash
#BSUB -J test_autoencoder1
#BSUB -o /home/monicai/tpot-ae/log/test1.log
#BSUB -e /home/monicai/tpot-ae/log/test1.err
#BSUB -q moore_long
#BSUB -M 60000
#BSUB -R "span[hosts=1]"
#BSUB -n 8 
#export PATH="/home/monicai/anaconda3/bin:$PATH"
cd /home/monicai/tpot-ae
source activate tpotae

python run_simple_ae_aml.py