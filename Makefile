#UCF-CC-50 DATASET
augment-ucf: #augment data of ucf-cc-50, do this just once
	python3 main.py -d ucf-cc-50 --force-den-maps --force-augment --augment-only

train-ucf-fold1: #train fold 1 of ucf-cc-50 dataset
	python3 main.py -d ucf-cc-50 --units ucf-fold1 --train-batch 32 --save-dir log/multi-stream
train-ucf-fold2: #train fold 2 of ucf-cc-50 dataset
	python3 main.py -d ucf-cc-50 --units ucf-fold2 --train-batch 32 --save-dir log/multi-stream
train-ucf-fold3: #train fold 3 of ucf-cc-50 dataset
	python3 main.py -d ucf-cc-50 --units ucf-fold3 --train-batch 32 --save-dir log/multi-stream
train-ucf-fold4: #train fold 4 of ucf-cc-50 dataset
	python3 main.py -d ucf-cc-50 --units ucf-fold4 --train-batch 32 --save-dir log/multi-stream
train-ucf-fold5: #train fold 5 of ucf-cc-50 dataset
	python3 main.py -d ucf-cc-50 --units ucf-fold5 --train-batch 32 --save-dir log/multi-stream

test-ucf-fold1: #test fold 1 of ucf-cc-50 dataset
	python3 main.py -d ucf-cc-50 --units ucf-fold1 --save-dir log/multi-stream --resume log/multi-stream --evaluate-only 
test-ucf-fold2: #test fold 2 of ucf-cc-50 dataset
	python3 main.py -d ucf-cc-50 --units ucf-fold2 --save-dir log/multi-stream --resume log/multi-stream --evaluate-only 
test-ucf-fold3: #test fold 3 of ucf-cc-50 dataset
	python3 main.py -d ucf-cc-50 --units ucf-fold3 --save-dir log/multi-stream --resume log/multi-stream --evaluate-only 
test-ucf-fold4: #test fold 4 of ucf-cc-50 dataset
	python3 main.py -d ucf-cc-50 --units ucf-fold4 --save-dir log/multi-stream --resume log/multi-stream --evaluate-only 
test-ucf-fold5: #test fold 5 of ucf-cc-50 dataset
	python3 main.py -d ucf-cc-50 --units ucf-fold5 --save-dir log/multi-stream --resume log/multi-stream --evaluate-only

#SHANGHAI-TECH DATASET
augment-shanghai: #augment data of Shanghai Tech, do this just once
	python3 main.py -d shanghai-tech --force-den-maps --force-augment --augment-only

train-shanghai-partA: #train part A of Shanghai Tech dataset
	python3 main.py -d shanghai-tech --units shanghai-partA --train-batch 32 --save-dir log/multi-stream
train-shanghai-partB: #train part B of Shanghai Tech dataset 
	python3 main.py -d shanghai-tech --units shanghai-partB --train-batch 32 --save-dir log/multi-stream

test-shanghai-partA: #test part A of Shanghai Tech dataset
	python3 main.py -d shanghai-tech --units shanghai-partA --save-dir log/multi-stream --resume log/multi-stream --evaluate-only
test-shanghai-partB: #test part B of Shanghai Tech dataset 
	python3 main.py -d shanghai-tech --units shanghai-partB --save-dir log/multi-stream --resume log/multi-stream --evaluate-only

