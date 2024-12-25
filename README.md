
<h1 align="center"><span style="font-weight:normal">GaitPT: Skeletons Are All You Need For Gait Recognition</h1>
<h3 align="center"><span style="font-weight:normal">Accepted at: The 18th IEEE International Conference on Automatic Face and Gesture Recognition </h3>

<p align="center"> <a href="https://arxiv.org/pdf/2308.10623"> 📘 Paper PDF </a> | <a href="https://docs.google.com/presentation/d/1Vz1RStFqZxcaMGtB25U5OJ-VdmRMLnA6qfQFS7-WL2E/edit?usp=sharing"> 🪧 Poster </a> |  <a href="https://docs.google.com/presentation/d/15DsTQxjnWf7NA47emtmEin47dzYhMM9CK1eBhoKK9EA/edit?usp=sharing"> 🛝🛝 Slides </a> </p>

<div align="center">
<strong> Authors </strong>: <a href="https://scholar.google.com/citations?user=ct7ju7EAAAAJ&hl=en&oi=ao"> Andy Catruna </a>, <a href="https://scholar.google.com/citations?user=cdYk_RUAAAAJ&hl=en"> Adrian Cosma</a>, <a href="https://scholar.google.com/citations?user=yjtWIf8AAAAJ&hl=en">Emilian Radoi </a>
</div>

<div></div>

<img src="images/arch.png">

## <a name="abstract"></a> 📘 Abstract
*The analysis of patterns of walking is an important area of research that has numerous applications in security, healthcare, sports and human-computer interaction. Lately, walking patterns have been regarded as a unique fingerprinting method for automatic person identification at a distance.  In this work, we propose a novel gait recognition architecture called Gait Pyramid Transformer (GaitPT) that leverages pose estimation skeletons to capture unique walking patterns, without relying on appearance information. GaitPT adopts a hierarchical transformer architecture that effectively extracts both spatial and temporal features of movement in an anatomically consistent manner, guided by the structure of the human skeleton. Our results show that GaitPT achieves state-of-the-art performance compared to other skeleton-based gait recognition works, in both controlled and in-the-wild scenarios. GaitPT obtains 82.6% average accuracy on CASIA-B, surpassing other works by a margin of 6%. Moreover, it obtains 52.16% Rank-1 accuracy on GREW, outperforming both skeleton-based and appearance-based approaches.*

## <a name="getting-started"></a> 📖 Getting Started

### Prerequisites
Install all dependencies with the following command:
```pip install -r requirements.txt```

### Data
Download the preprocessed CASIA-B [1] skeletons from [here](https://drive.google.com/drive/u/2/folders/1QzpM7aj5tU0QhiiqaXM2R1dxXh4lt4tE) and extract them in the ```data/``` directory. For GREW [2] and Gait3D [3], contact the authors for the 2D skeletons.

[1] Shiqi Yu, Daoliang Tan and Tieniu Tan, "A Framework for Evaluating the Effect of View Angle, Clothing and Carrying Condition on Gait Recognition," 18th International Conference on Pattern Recognition (ICPR'06), Hong Kong, China, 2006, pp. 441-444, doi: 10.1109/ICPR.2006.67.

[2] Zhu, Zheng, et al. "Gait recognition in the wild: A benchmark." Proceedings of the IEEE/CVF international conference on computer vision. 2021.

[3] Jinkai Zheng, Xinchen Liu, Wu Liu, Lingxiao He, Chenggang Yan, Tao Mei, "Gait Recognition in the Wild with Dense 3D Representations and A Benchmark." (2022). IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

### Training from scratch

To train on CASIA-B run:
```scripts/train_casia.sh```

To train on GREW run:
```scripts/train_grew.sh```

To train on Gait3D run:
```scripts/train_gait3d.sh```

### Evaluating pretrained models:
Download the pretrained weights from [here](https://drive.google.com/drive/folders/1gHouvtwEI7h0gylb5JjEwWZyu9iVMuNR?usp=sharing) and place them in the ```checkpoints/``` directory.

To evaluate pretrained model on CASIA-B run:
```scripts/evaluate_casia.sh```

To evaluate pretrained model on GREW run:
```scripts/evaluate_grew.sh```

To evaluate pretrained model on Gait3D run:
```scripts/evaluate_gait3d.sh```

## <a name="results"></a> 📖 Results

<div>
<img width="300px" src="images/casia-results.png">
<img width="300px" src="images/grew-results.png">
<img width="300px" src="images/gait3d-results.png">

</div>

## <a name="citation"></a> 📖 Citation
If you found our work useful or use our dataset in any way, please cite our paper:

```
@inproceedings{catruna24gaitpt,
  author       = {Andy Catruna and
                  Adrian Cosma and
                  Emilian Radoi},
  title        = {GaitPT: Skeletons are All You Need for Gait Recognition},
  booktitle    = {18th {IEEE} International Conference on Automatic Face and Gesture
                  Recognition, {FG} 2024, Istanbul, Turkey, May 27-31, 2024},
  pages        = {1--10},
  publisher    = {{IEEE}},
  year         = {2024},
  url          = {https://doi.org/10.1109/FG59268.2024.10581947},
  doi          = {10.1109/FG59268.2024.10581947},
  timestamp    = {Wed, 31 Jul 2024 14:28:05 +0200},
  biburl       = {https://dblp.org/rec/conf/fgr/CatrunaCR24b.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

## <a name="license"></a> 📝 License

This work is protected by [CC BY-NC-ND 4.0 License (Non-Commercial & No Derivatives)](LICENSE).
