# RAIL
Searching in Branch-and-Bound Algorithms via Generative Adversarial Imitation Learning <br>
# Download the ground truth
```
bash download.sh
```
# Begin training (e.g. TSP with 20 nodes)
```
python train.py --TSP20
```
Applying reinforcement learning to training:
```
python train.py --TSP20 --pass_range 3 --rl
```
Applying imitation learning to training:
```
python train.py --TSP20 --pass_range 3 --il
```
# Testing
```
python train.py --tag --pass_range 3 --rl --eval 
```
If you want to use reinforcement learning and generative adversarial imitation learning separately, then you can use the following instructions <br>
## RL
```
python train.py --tag --pass_range 3 --rl --eval --load RAIL/pretrained/rl.pb --seed 1
```
## GAIL
```
python train.py --tag --pass_range 3 --eval --load RAIL/pretrained/il.pb --seed 1
```
