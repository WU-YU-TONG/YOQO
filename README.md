# You Only Query Once: An Efficient Label-Only Membership Inference Attack
This is the official code for paper _You Only Query Once: An Efficient Label-Only Membership Inference Attack_

The code only requires the pytorch library to run.

Get the dataset we use during the experiment for a quick start via the [link]([https://www.openai.com](https://entuedu-my.sharepoint.com/:u:/g/personal/yutong002_e_ntu_edu_sg/EYSqHo56ZWhJtjlCq4ttDVgBKO9WryXPz8trQfIB-2O21A?e=Le1Ry1) download the zip file and run

```
cd shadow_training
cd ..
unzip shadow_dataset.zip -d shadow_dataset
```

Get the victim models and the attack samples we used via the [link](https://entuedu-my.sharepoint.com/:u:/g/personal/yutong002_e_ntu_edu_sg/EZruADtZVb9AuWQmPmNg0eIBXXokHg2ykZPeLinaZA0dQg?e=nNkqEN) and run

```
cd shadow_training
unzip samples_n_checkpoints.zip -d ./
```

This is to put the 'checkpoints' and 'samples' under the './shadow_training', Then run

```
cd samples
mv ./ ../
```
.
to move the 'AE_Online' and 'AE_Offline' folders to './shadow_training'. In other word, the paths under 'shadow_training' should be like:

```
shadow_traning
      ├── AE_Offline
      ├── AE_Online
      ├── atk_utils.py
      ├── attack.py
      ├── attack_evaluation.py
      ├── cert_radius
      ├── checkpoint
      ├── yoqo_atk.py
      ├── other files...
```

To conduct the online attack from scratch, run:
```
cd shadow_training
python yoqo_atk.py -online -alpha 2 -loss_threshold 4 -dataset CIFAR10
```

To conduct the offline attack from scratch, run:
```
python yoqo_atk.py -alpha 5 -loss_threshold 8 -dataset CIFAR10
```

To directly use the attack samples we provide, run
```
python attack_evaluation.py -defence ' ' (for no defence setting) -atk_type 'online' (options: 'online'/'offline') -dataset 'CIFAR10' -net 'CNN7' -test_net 'CNN7' -data_size 2500 (train set size for shadow model) -test_data_size 2500 (train set size for victim model)
```
Note: make sure the file stucture is the same as what is narrated above.
