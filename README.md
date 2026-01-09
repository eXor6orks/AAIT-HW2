# Task 1

## Pseudo labelling

Train warm-up :
python T1_WU.py --dataset task1

Generate pseudo label :
python T1_generate_pseudo_label.py --dataset task1 --model checkpoints/retrained_student.pth

Retrain on label + pseudo label
python T1_train_PL.py --pseudo_label pseudo_labels.csv

## Meta pseudo labelling

Train Meta pseudo labelling :
python T1_train_mpl.py --dataset task1 --wu_checkpoints checkpoints

If you don't have checkpoints, a new model will be create

# Task 2

## Co-Teaching 

python T2_CoTeaching.py --dataset task2 --noisy_rate 0.25
