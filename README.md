<h1 align="center">
  <i>GLoMOT+: Enhancing Low-Frame-Rate Multi-Object Tracking via Dual-Space Discriminative Learning</i>
</h1>

## :fire: News

- <span style="font-variant-numeric: tabular-nums;">**2026.01.22**</span>: We have submitted this paper. The expanded version code will be released after revision.

# 1. Introduction

We propose an Online Graph Neural Network framework GLoMOT+ for tracking Low-Frame-Rate videos, including a Dynamic Node Buffer Pool to bridge long temporal intervals via a long-term memory mechanism; an Adaptive Context-aware Gating Module to dynamically handle feature uncertainty; a novel pseudo-depth feature calculation method to alleviate occlusions and a Dual-Space Discriminative Learning Strategy to distinguish visually similar objects.

# 2. Comparison with the Conference Version

* **Extended Edge Feature Computation:** We propose a method to compute implicit features by projecting visual embeddings onto a discriminative subspace for extended GNN edge feature computation.
* **Dual-Space Discriminative Learning:** We introduce a strategy to fuse explicit and implicit features during message passing, effectively addressing the visual ambiguity that the original GLoMOT struggled with in crowded, similar-appearance scenarios.
* **Extensive Experiments and Analysis:** We provide a more comprehensive evaluation, expanding from MOT17, DanceTrack, and VisDrone to include SportsMOT and MOT20. We also provide more in-depth analysis and visualizations to validate the effectiveness of the new modules.

# 3. Install Requirements

```
pip3 install -r requirements.txt
```

# 4. Dataset File Structure

```
/datasets/ 
|-- MOT17/
|	|--  train/
|	|--  test/
|	|-- val/
|	     |-- MOT17-XX_FRCNN-N-2
|	     |-- MOT17-XX_FRCNN-N-5
|	     |-- MOT17-XX_FRCNN-N-10
|-- DacneTrack/
|	|--  train/
|	|--  val/
|	|--  test/
|	|-- val_lfr/
|	     |--  dancetrack00XX-N-2
|	     |--  dancetrack00XX-N-5
|	     |--  dancetrack00XX-N-10
|-- SportsMOT/
|	|--  train/
|	|--  val/
|	|--  test/
|-- MOT20
|	|--  train/
|	|--  test/
|-- Visdrone-MOT
	|-- train
	|--  val
	|--  test-dev
```

# 5. Data Preparation

This section covers how to prepare the datasets for training and evaluation.

**5.1 Split MOT17 Dataset (train_half / val_half)**

For training and validation on MOT17, you can split the official training set into two halves (train_half and val_half). Use the ```tool/get_mot17_valhalf.py ``` script to automatically perform this split. This script will create a val directory and move the second half of each sequence from the train directory into it, re-indexing the frames and IDs accordingly.

**Usage:**

```
change the dataset ROOT_PATH

python tool/get_mot17_valhalf.py
```

**5.2 Generate Low-Frame-Rate Datasets (MOT17-lfr & DanceTrack-lfr)**

To evaluate the tracker's performance in Low-Frame-Rate scenarios, you first need to generate the corresponding datasets from the original high-frame-rate videos. Use the provided ```tool/get_low_frame_rate_data.py``` script. This script will sample frames at a given interval (--interval) and create new sequence folders with an -N-n suffix.

**Example for MOT17 (frame gap = 2):**

```
python tool/get_low_frame_rate_dataset.py  --root_dir /path/to/your/datasets/mot/val/  --interval 2  --dataset mot17
```

**Example for DanceTrack（frame gap =10）:**

```
python tool/get_low_frame_rate_dataset.py  --root_dir /path/to/your/datasets/DanceTrack/val/  --interval 10  --dataset dancetrack
```

# 6. Model Preparation

**6.1 ReID Model**

This project uses the `sbs-s50` model from the **fast-reid** library to extract appearance features. You can download the pre-trained weights from the official repository or other sources.

​**Download Link**
fast_reid model zoo can download at [here.](https://github.com/Kroery/DiffMOT)

Place the downloaded model weights in `/pretrained` directory and change the model name as `mot17_sbs_S50.pth`and`dancetrack_sbs_S50.pth`.

**6.2 Evaluation Model**

We provide a pre-trained evaluation model specifically for validating tracking performance on Low-Frame-Rate videos.

* ​**Location**​: The model is located in the `/models` directory.
* ​**Size**​: The model size is approximately 1.1MB.**
* You can directly use this model with the `--model_path` argument in your evaluation commands.
# 7. Low-Frame-Rate Evaluation

To evaluate the tracker on the generated Low-Frame-Rate datasets, run the main script in `test` mode. Make sure to point the `--data_path` to the directory containing the Low-Frame-Rate sequences (e.g., `dancetrack/val_lfr`).

**Evaluate MOT17 in Low-Frame-Rate**

```
python GLoMOT+_main.py --experiment_mode test --cuda --test_splits mot17-val-f2 --use_node_buffer True --buffer_len 20 --det_file byte065 --data_path "E:/python/GLoMOT/datasets/" --reid_embeddings_dir reid_sbs_s50 --node_embeddings_dir node_sbs_s50 --reid_arch fastreid_mot17_sbs_S50
```

**Evaluate DanceTrack in Low-Frame-Rate**

```
python GLoMOT+_main.py --experiment_mode test --cuda --test_splits dancetrack-val-f2 --use_node_buffer True --buffer_len 10 --det_file byte065 --data_path "E:/python/GLoMOT/datasets/" --reid_embeddings_dir reid_sbs_s50 --node_embeddings_dir node_sbs_s50 --reid_arch fastreid_dancetrack_sbs_S50 --use_smpnet True
```

We use [TrackEval ](https://github.com/JonathonLuiten/TrackEval)to evaluate the results.

# 8. Benchmark Evaluation

* **Tracking MOT17 test set**

```
python GLoMOT+_main.py --experiment_mode test --cuda --test_splits mot17-test --use_node_buffer True --buffer_len 30 --det_file byte065 --data_path "E:/python/GLoMOT/datasets/" --reid_embeddings_dir reid_sbs_s50 --node_embeddings_dir node_sbs_s50 --reid_arch fastreid_mot17_sbs_S50  --use_smpnet True
```

* **Tracking DanceTrack test set**

```
python GLoMOT+_main.py --experiment_mode test --cuda --test_splits dancetrack-test --use_node_buffer True --buffer_len 20 --det_file byte065 --data_path "E:/python/GLoMOT/datasets/" --reid_embeddings_dir reid_sbs_s50 --node_embeddings_dir node_sbs_s50 --reid_arch fastreid_dancetrack_sbs_S50 --use_smpnet True
```

* **Tracking SporstMOT test set**

```
python GLoMOT+_main.py --experiment_mode test --cuda --test_splits SportsMOT-test --use_node_buffer True --buffer_len 60 --det_file byte065 --data_path "E:/python/GLoMOT/datasets/" --reid_embeddings_dir reid_sbs_s50 --node_embeddings_dir node_sbs_s50 --reid_arch fastreid_sportsmot_sbs_S50 --use_smpnet True
```

* **Tracking MOT20 test set**

```
python GLoMOT+_main.py --experiment_mode test --cuda --test_splits mot20-test --use_node_buffer True --buffer_len 30 --det_file byte065 --data_path "E:/python/GLoMOT/datasets/" --reid_embeddings_dir reid_sbs_s50 --node_embeddings_dir node_sbs_s50 --reid_arch fastreid_mot20_sbs_S50 --use_smpnet True
```

* **Tracking Visdrone test-dev set**

```
python GLoMOT+_main.py --experiment_mode test --cuda --test_splits visdrone-test --use_node_buffer True --buffer_len 60 --det_file byte065 --data_path "E:/python/GLoMOT/datasets/" --reid_embeddings_dir reid_sbs_s50 --node_embeddings_dir node_sbs_s50 --reid_arch fastreid_visdrone_sbs_S50 --use_smpnet True
```

# 9. Model Training

```
python GLoMOT+_main.py --experiment_mode train --cuda --train_splits dancetrack-train --val_splits dancetrack val --det_file byte065 --data_path "E:/python/GLoMOT/datasets/" --reid_embeddings_dir reid_sbs_s50 --node_embeddings_dir node_sbs_s50 --reid_arch fastreid_dancetrack_sbs_S50 --use_smpnet True
```

# 10. Acknowledgement

A large part of the code is borrowed from [SUSHI](https://github.com/dvl-tum/SUSHI) and [HAT](https://github.com/HELLORPG/HATReID-MOT). Our conference version code is in [GLoMOT](https://github.com/sad123-yx/GLoMOT). Thanks for their wonderful works!
