#!/bin/bash

OPTIMS=("Glister" "ContextualDiversity" "GraNd" "Cal" "Uniform" "Submodular")
FRACS=(0.1 0.5)
LRate=(0.1 0.01 0.001)

for optim in "${OPTIMS[@]}"; do
    for frac in "${FRACS[@]}"; do
        for lr in "${LRate[@]}"; do
            outfile="mobilenet_${optim}_${frac}.txt"
            errfile="error_mobilenet_${optim}_${frac}.txt"
            if [[ "$optim" == "Submodular" ]]; then
                submodular_arg="--submodular GraphCut"
            fi
            python -u main.py \
                --fraction "$frac" \
                --dataset CIFAR10 \
                --data_path ~/datasets \
                --num_exp 5 \
                --workers 10 \
                --optimizer Adam \
                -se 10 \
                --selection "$optim" \
                $submodular_arg \
                --model MobileNetV3Small \
                --lr "$lr" \
                -sp ./result \
                --batch 128 \
                > "$outfile" \
                2> "$errfile"
    done
done