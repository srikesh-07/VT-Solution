<h1 align="center">📊 Visual Taxonomy Challenge - Predicting Multi-Attributes by Meesho</h1>
<h2 align="center">A Framework Consisting of Dynamic Shared ViT with Class Re-weighting Algorithms, Loss-Reweighting Algorithms, and Specialized Losses for Long-Tail Imbalance to predict Multi-Attributes of Clothes</h2>
<h3 align="center">Solution Proposed by <a href="https://www.kaggle.com/neuralretr0">Neural-Retro</a> (<a href="mailto:srikeshram05@gmail.com">Srikeshram B</a>)</h3>

# 📖 Introduction
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Clothing attribute prediction from images is inherently challenging due to noisy, imbalanced datasets and domain adaptation issues. Traditional models often struggle to balance learning across categories, especially minority classes, and are prone to biased predictions and catastrophic forgetting. This work presents a robust solution that leverages a domain-adapted Vision Transformer (ViT-B/32) backbone pre-trained on clothing data, minimizing the need for adaptation while effectively extracting relevant features. A customized data pipeline efficiently handles missing values, implements category-specific batch sampling, and applies augmentation techniques to improve model generalization. The model incorporates state-of-the-art loss weighting methods, such as Class Balanced Loss combined with Focal Loss, to ensure balanced learning and prevent conflicts between categories. This strategy mitigates class imbalance, particularly for underrepresented categories, and avoids biased learning toward a dominant category. The training process, utilizing AdamW optimizer, gradient clipping, and CosineLR scheduling with warmup, ensures stable and gradual convergence. The proposed approach achieves competitive results, effectively handling class imbalance and outperforming baseline methods on public benchmarks.

# 🛠️ Setup
### 1. Install Miniconda3 or Anaconda3
Install Miniconda3 or Anaconda3 to manage the environment.
### 2. Create the Environment
Run the following command to create the environment:
```
conda env create -f environment.yml
```
### 3. Activate the Environment
Activate the environment with:
```
conda activate vt_challenge
```
### 4. Download the Pre-Trained Backbone weights for Training
This step downloads the pre-trained CLIP backbone on GS10M dataset. Refer [GCL](https://github.com/marqo-ai/GCL) repository for more detatils. Execute the following command to download the pre-trained checkpoints
```
bash scripts/download.sh
```
# ⚙️ Training
### 1. Edit the `scripts/train.sh` with the following parameters:
1. `IMG_DIR` - Train Image Directory
2. `CSV_PATH` - Train CSV Path
3. `SAVE_DIR` - Directory to save all the Checkpoints

If WandB is needed, then
  - Remove `--no-wandb` arguement
  - Add `WANDB_API_KEY`

**The explanation of other Training Arguments can be found [here](utils/README.md).**

### 2. Start the Training
To begin the training, run:
```
bash scripts/train.sh
```
**Note**: If any issues arise with reproducing the results, use the train/validation split employed during model creation by uncommenting lines `197` and `198` in `train/trainer.py`.

# 🏆 Post Training
After training, you'll see the following structure in the output directory:
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
- `best_score.pt` - Best Checkpoint with **Maximum F1-Score** (ignoring Loss).
- `best_loss.pt` - Best Checkpoint with **Minimal Loss** (ignoring F1-Score).
- `both_detailed_metrics.json` - Contains the Harmonic Mean, Micro, and Macro scores for Overall, Category-Level, and Attribute-Level F1-metrics of `best_both.pt`.
- `score_detailed_metrics.json` - Contains the Harmonic Mean, Micro, and Macro scores Overall, Category-Level, and Attribute-Level F1-metrics of `best_score.pt`.
- `loss_detailed_metrics.json` - Contains the Harmonic Mean, Micro, and Macro scores Overall, Category-Level, and Attribute-Level F1-metrics of `best_loss.pt`.

# 🔍 Inference
### 1. Get the Checkpoint Path
- Download the best checkpoint from the provided Google Drive link and copy its path.
- Alternatively, use your own trained model and note the path to the checkpoint.
### 2. Edit the `scripts/inference.sh` with the following parameters:
1. `TEST_IMG_DIR` - Test Image Directory
2. `TEST_CSV_PATH` - Test CSV Path
3. `CKPT_PATH` - Path to the Checkpoint
   
**The explanation of other Inference Arguments can be found [here](utils/README.md).**


### 3. Start the Inference
To start inference, run:
```
bash scripts/inference.sh
```
### 4. Post Inference
After inference, you'll find the CSV in the same directory as the checkpoint, named according to the checkpoint with a `.csv `suffix.

# 💥 List of Supported Losses
1. Cross Entropy (`CE`) - [arXiv](https://arxiv.org/abs/1512.00567)
2. Label-Distribution-Aware Margin (`LDAM`) - [arXiv](https://arxiv.org/abs/1906.07413)
3. Focal Loss (`Focal`) - [arXiv](https://arxiv.org/abs/1708.02002)
4. Influence-Balanced Loss (`IB`) - [arXiv](https://arxiv.org/abs/2110.02444)
5. Influence-Balanced with Focal Loss (`IBFocal`) - [arXiv](https://arxiv.org/abs/2110.02444)
6. Class Balanced Loss with Cross Entropy (`CE-CB`) - [arXiv](https://arxiv.org/abs/1901.05555)
7. Class Balanced Loss with Focal Loss (`F-CB`) - [arXiv](https://arxiv.org/abs/1901.05555)
8. Vector Scaling Loss (`VS`) - [arXiv](https://arxiv.org/abs/2103.01550)
9. Vector Scaling Loss with Class-Dependent Temperature (`CDT`) - [arXiv](https://arxiv.org/abs/2103.01550)
10. Vector Scaling Loss with Logit Adjustment (`LA`) - [arXiv](https://arxiv.org/abs/2103.01550)
    
To use any of the above losses, pass the respective short name as an argument on `--loss_type`.

# 🔄 List of Supported Training Schedules
1. Deferred Re-weighting (`DRW`) - [arXiv](https://arxiv.org/abs/1906.07413)
2. Class Balanced Re-weighting (`CBReweight`) - [arXiv](https://arxiv.org/abs/1901.05555)
3. Influence Balanced Re-weighting (`IBReweight`) - [arXiv](https://arxiv.org/abs/2110.02444)
   
To use any of the above schedules, pass the respective short name as an argument on `--train_rule` .

# ⚖️ List of Supported Loss Weighting Algorithms
1. Equal Weighting (`EW`)
2. Gradient Normalization (`GradNorm`) - [MLR](http://proceedings.mlr.press/v80/chen18a/chen18a.pdf)
3. Uncertainty Weights (`UW`) - [CVPR](https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf)
4. MGDA (`MGDA`) - [NeurIPS](https://papers.nips.cc/paper/2018/hash/432aca3a1e345e339f35a30c8f65edce-Abstract.html)
5. Dynamic Weight Average (`DWA`) - [CVPR](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.pdf)
6. Geometric Loss Strategy (`GLS`) - [CVPRW 2019](https://openaccess.thecvf.com/content_CVPRW_2019/papers/WAD/Chennupati_MultiNet_Multi-Stream_Feature_Aggregation_and_Geometric_Loss_Strategy_for_Multi-Task_CVPRW_2019_paper.pdf)
7. Projecting Conflicting Gradient (`PCGrad`) - [NeurIPS 2020](https://papers.nips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)
8. Gradient Sign Dropout (`GradDrop`) - [NeurIPS 2020](https://papers.nips.cc/paper/2020/hash/16002f7a455a94aa4e91cc34ebdb9f2d-Abstract.html)
9. Impartial Multi-Task Learning (`IMTL`) - [ICLR 2021](https://openreview.net/forum?id=IMPnRXEWpvr)
10. Gradient Vaccine (`GradVac`) - [ICLR 2021](https://openreview.net/forum?id=F1vEjWK-lH_)
11. Conflict-Averse Gradient Descent (`CAGrad`) - [NeurIPS 2021](https://openreview.net/forum?id=_61Qh8tULj_)
12. Nash-MTL (`Nash_MTL`) - [ICML 2022](https://proceedings.mlr.press/v162/navon22a/navon22a.pdf)
13. Random Loss Weighting (`RLW`) - [TMLR 2022](https://openreview.net/forum?id=jjtFD8A1Wx)
14. MoCo (`MoCo`) - [ICLR 2023](https://openreview.net/forum?id=dLAYGdKTi2)
15. Aligned-MTL (`Aligned_MTL`) - [CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Senushkin_Independent_Component_Alignment_for_Multi-Task_Learning_CVPR_2023_paper.html)
16. STCH (`STCH`) - [ICML 2024](https://openreview.net/forum?id=m4dO5L6eCp)
17. ExcessMTL (`ExcessMTL`) - [ICML 2024](https://openreview.net/forum?id=JzWFmMySpn)
18. FairGrad (`FairGrad`) - [ICML 2024](https://openreview.net/forum?id=KLmWRMg6nL)
19. DB-MTL (`DB_MTL`) - [arXiv](https://arxiv.org/abs/2308.12029)
    
To use any of the above loss weighting algorithms, pass the respective short name as an argument on  `--method` .

# ❓ FAQ

1. **Why is my training not reproducing results?**\
Ensure you use the same train/validation split employed during model creation. Uncomment lines `197` and `198` in `train/trainer.py`.

2.  **Where are my training checkpoints stored?**\
All model checkpoints and metrics are stored in the directory specified in SAVE_DIR.


# 🤝 Credits
This codebase is built with the help of the following repositories and incorporates parts of code from these amazing repositories. **Heartfelt thanks to the respective authors for their incredible work**:
1. GCL - [GitHub](https://github.com/marqo-ai/GCL)
2. LibMTL - [GitHub](https://github.com/median-research-group/LibMTL)
3. CBLoss - [GitHub](https://github.com/wildoctopus/cbloss)
4. IBLoss - [GitHub](https://github.com/pseulki/IB-Loss)
5. VSLoss - [Github](https://github.com/orparask/VS-Loss)

# 📝 TO DO
1. Documentation of Code
