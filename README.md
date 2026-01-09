# Task 1

## Pseudo labelling

Train warm-up :
```
python T1_WU.py --dataset task1
```

Generate pseudo label :
```
python T1_generate_pseudo_label.py --dataset task1 --model checkpoints/retrained_student.pth
```

Retrain on label + pseudo label
```
python T1_train_PL.py --pseudo_label pseudo_labels.csv
```

## Meta pseudo labelling

Train Meta pseudo labelling :
```
python T1_train_mpl.py --dataset task1 --wu_checkpoints checkpoints
```

If you don't have checkpoints, a new model will be create

# Task 2

## Co-Teaching 

Train Co-Teaching :
```
python T2_CoTeaching.py --dataset task2 --noisy_rate 0.25
```
## Loss-based data cleaning

Train Warm-Up :
```
python T2_WU.py --dataset task2
```
Create clean images CSV files :
```
python T2_clean_images.py --dataset task2 --noise_rate 0.20 --checkpoint checkpoints/task2/best_resnet.pth
```

Train with clean images files :
```
python T2_train_CI.py --dataset task2 --clean_file task2_clean.csv
```

## Other parameters

For each training, you can modify the number of Epochs and the batch size with --epochs and --batch_size