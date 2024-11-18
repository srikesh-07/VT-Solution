# Visual Taxonomy Challgenge - Predicting Multi-Attributes by Meesho
### Solution proposed by [Neural-Retro](https://www.kaggle.com/neuralretr0) ([Srikeshram B](mailto:srikeshram05@gmail.com))

## Setup
### 1. Install Miniconda3 or Anaconda3
### 2. Create the Environment
```
conda env create -f environment.yml
```
### 3. Activate the Environment
```
conda activate vt_challenge
```
### 4. Download the Pre-Trained Backbone weights for Training
This step downloads the pre-trained CLIP backbone on GS10M dataset. Refer [GCL](https://github.com/marqo-ai/GCL) repository for more detatils. Execute the following command to download the pre-trained checkpoints
```
bash scripts/download.sh
```

## Training
### 1. Edit the `scripts/train.sh` with the following parameters:
1. `IMG_DIR` - Train Image Directory
2. `CSV_PATH` - Train CSV Path
3. `SAVE_DIR` - Directory to save all the Checkpoints

If WandB is needed, then
  - Remove `--no-wandb` arguement
  - Add `WANDB_API_KEY`
### 2. Start the Training
```
bash scripts/train.sh
```
#### Note: If any issues arise with reproducing the results, use the train/validation split employed during model creation by uncommenting lines `197` and `198` in `train/trainer.py`.

### 3. Post Training
After training the model, You can see the following stucture in the output directory,
```
output_dir
|___________ best_both.pt
|___________ best_score.pt 
|___________ best_loss.pt
|___________ both_detailed_metrics.jsonn
|___________ score_detailed_metrics.json
|___________ loss_detailed_metrics.json

```

- `best_both.pt` - Best Checkpoint with **Minimal Loss and Maximum F1-Score**.
- `best_score.pt` - Best Checkpoint with **Maximum F1-Score** and no cares on loss.
- `best_loss.pt` - Best Checkpoint with **Minimal Loss** and no cares on F1-Score.
- `both_detailed_metrics.json` - Contains the Harmonic Mean, Micro and Macro scores Overall F1-Metrics, Category-Level F1-Metrics and Attribute-Level F1-metrics of `best_both.pt`.
- `score_detailed_metrics.json` - Contains the Harmonic Mean, Micro and Macro scores Overall F1-Metrics, Category-Level F1-Metrics and Attribute-Level F1-metrics of `best_score.pt`.
- `loss_detailed_metrics.json` - Contains the Harmonic Mean, Micro and Macro scores Overall F1-Metrics, Category-Level F1-Metrics and Attribute-Level F1-metrics of `best_loss.pt`.

## Inference
### 1. Get the Checkpoint Path
- Download the given best Checkpoint from the provided Google Drive link given in Google Form and copy the checkpoint path.
- If you have trained your own model, Just note down the path to the checkpoint
### 1. Edit the `scripts/inference.sh` with the following parameters:
1. `TEST_IMG_DIR` - Test Image Directory
2. `TEST_CSV_PATH` - Test CSV Path
3. `CKPT_PATH` - Path to the Checkpoint

### 2. Start the Inference
```
bash scripts/inference.sh
```
### 3. Post Inference
The user can find the CSV in the same directory where the checkpoint is the present in the name of the checkpoint with suffix as `.csv`



# Credits
This codebase is built with the help of the following repositories and some parts of the code are incorporated from the following repositories. **Heartful Thanks to the respective authors for their amazing work**.
1. GCL - [GitHUb](https://github.com/marqo-ai/GCL)
2. LibMTL - [GitHub](https://github.com/median-research-group/LibMTL)
3. CBLoss - [GitHub](https://github.com/wildoctopus/cbloss)
4. IBLoss - [GitHub](https://github.com/pseulki/IB-Loss)
5. VSLoss - [Github](https://github.com/orparask/VS-Loss)

# TO DO
1. Documentation of Code
2. Explanation of All Command Line Arugements
3. Explanation of List of Supported Losses
4. Explanation of List of Training Rules
5. Explanation of List of Loss Weighting Methods
