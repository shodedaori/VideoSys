#!/bin/bash

python vbench/run_vbench.py \
    --video_path samples/latte_pab \
    --save_path results/latte_pab

python vbench/run_vbench.py \
    --video_path samples/latte_stu \
    --save_path results/latte_stu