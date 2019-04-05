run_ucf =  python3 main.py -d ucf-cc-50
run_shanghai = python3 main.py -d shanghai-tech

all: ucf shanghai

#UCF-CC-50 DATASET
ucf: augment-ucf train-ucf-fold1 train-ucf-fold2 train-ucf-fold3 train-ucf-fold4 train-ucf-fold5

augment-ucf: #augment data of ucf-cc-50, do this just once
	$(run_ucf) --augment-only --force-augment --force-den-maps

train-ucf-fold1: #train fold 1 of ucf-cc-50 dataset
	$(run_ucf) --units ucf-fold1 --train-batch 32 --save-dir log/multi-stream
train-ucf-fold2: #train fold 2 of ucf-cc-50 dataset
	$(run_ucf) --units ucf-fold2 --train-batch 32 --save-dir log/multi-stream
train-ucf-fold3: #train fold 3 of ucf-cc-50 dataset
	$(run_ucf) --units ucf-fold3 --train-batch 32 --save-dir log/multi-stream
train-ucf-fold4: #train fold 4 of ucf-cc-50 dataset
	$(run_ucf) --units ucf-fold4 --train-batch 32 --save-dir log/multi-stream
train-ucf-fold5: #train fold 5 of ucf-cc-50 dataset
	$(run_ucf) --units ucf-fold5 --train-batch 32 --save-dir log/multi-stream

test-ucf-fold1: #test fold 1 of ucf-cc-50 dataset
	$(run_ucf) --units ucf-fold1 --save-dir log/multi-stream --resume log/multi-stream/ucf-cc-50_people_thr_0_gt_mode_same/ --evaluate-only --save-plots
test-ucf-fold2: #test fold 2 of ucf-cc-50 dataset
	$(run_ucf) --units ucf-fold2 --save-dir log/multi-stream --resume log/multi-stream/ucf-cc-50_people_thr_0_gt_mode_same/ --evaluate-only --save-plots
test-ucf-fold3: #test fold 3 of ucf-cc-50 dataset
	$(run_ucf) --units ucf-fold3 --save-dir log/multi-stream --resume log/multi-stream/ucf-cc-50_people_thr_0_gt_mode_same/ --evaluate-only --save-plots
test-ucf-fold4: #test fold 4 of ucf-cc-50 dataset
	$(run_ucf) --units ucf-fold4 --save-dir log/multi-stream --resume log/multi-stream/ucf-cc-50_people_thr_0_gt_mode_same/ --evaluate-only --save-plots
test-ucf-fold5: #test fold 5 of ucf-cc-50 dataset
	$(run_ucf) --units ucf-fold5 --save-dir log/multi-stream --resume log/multi-stream/ucf-cc-50_people_thr_0_gt_mode_same/ --evaluate-only --save-plots

#SHANGHAI-TECH DATASET
shanghai: augment-shanghai train-shanghai-partA train-shanghai-partB test-shanghai-partA test-shanghai-partB

augment-shanghai: #augment data of Shanghai Tech, do this just once
	$(run_shanghai) --augment-only

train-shanghai-partA: #train part A of Shanghai Tech dataset
	$(run_shanghai) --units shanghai-partA --train-batch 32 --save-dir log/multi-stream
train-shanghai-partB: #train part B of Shanghai Tech dataset 
	$(run_shanghai) --units shanghai-partB --train-batch 32 --save-dir log/multi-stream

test-shanghai-partA: #test part A of Shanghai Tech dataset
	$(run_shanghai) --units shanghai-partA --save-dir log/multi-stream --resume log/multi-stream/shanghai-tech_people_thr_0_gt_mode_same/ --evaluate-only --save-plots
test-shanghai-partB: #test part B of Shanghai Tech dataset 
	$(run_shanghai) --units shanghai-partB --save-dir log/multi-stream --resume log/multi-stream/shanghai-tech_people_thr_0_gt_mode_same/ --evaluate-only --save-plots

