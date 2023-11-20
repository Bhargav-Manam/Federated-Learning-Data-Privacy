# Federated Learning & Data Privacy, 2022-2023

## General info
This is the lab course for the main course: 
[Federated Learning & Data Privacy, 2022-2023](http://www-sop.inria.fr/members/Giovanni.Neglia/federatedlearning/).  
Teachers: [Angelo Rodio](http://www-sop.inria.fr/members/Angelo.Rodio/), 
[Chuan Xu](https://sites.google.com/view/chuanxu).    
During these practical sessions students will have the opportunity to train ML models 
in a distributed way on Inria scientific cluster.  
Participation to the labs will be graded by the teachers.

## Prerequites
If not already done, students need to carry out the following administrative/configuration steps 
before the start of the labs:
* [Tutorial for the usage of NEF cluster](https://gitlab.inria.fr/arodio/FedCourse23/-/blob/main/NEF.pdf).

## Usage of Nef cluster 
1) log in account: run `ssh user@nef-frontal.inria.fr` (for people using Windows, you may need to use PuTTy)
2) reserve computing resources; for example a mode with one core. run `oarsub -l /nodes=1/core=1,walltime=3 -I`
3) run `module load conda/5.0.1-python3.6`
4) run `conda create -n fedCourse`
5) run `conda activate fedCourse`. If it does not work, run `source activate fedCourse`
6) run `module load cuda/11.0`  
7) run `conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11.0 -c pytorch`

## Class schedule
* TP1: 17/01/2023. Time: 09:00-12.00. Room: Lucioles Campus, 281.  
* TP2: 08/02/2023. Time: 13:30-16.30. Room: Lucioles Campus, 281.  
* TP3: 28/02/2023. Time: 09:00-12.00. Room: Lucioles Campus, 281.  
* TP4: 21/03/2023. Time: 09:00-12.00. Room: Lucioles Campus, 281.  

## Course material
[GitLab page of the course](https://gitlab.inria.fr/arodio/FedCourse23).

