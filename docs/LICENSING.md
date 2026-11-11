# Licensing

The code in this repository is released under the **GNU Affero General Public License v3.0** (see [LICENSE](../LICENSE)).

AGPL-3.0 is required, not just preferred. The V4 detector stage imports [Ultralytics](https://github.com/ultralytics/ultralytics) and fine-tunes its YOLO11m-seg weights. Both are distributed under AGPL-3.0, or under a separately purchased enterprise licence, which this project does not hold. A permissive licence such as MIT would not meet those obligations. See the [Ultralytics licence explanation](https://www.ultralytics.com/license).

Separately licensed material:

- **Pretrained weights.** Generic ImageNet initialisations (timm `tf_efficientnetv2_s.in21k_ft_in1k` through segmentation-models-pytorch) and Ultralytics `yolo11m-seg.pt` keep their upstream terms. This repository does not bundle any weights.
- **Competition data.** The SFSC 2026 images and annotations are governed by the [competition rules](https://www.kaggle.com/competitions/filament-segmentation-2026/rules): non-commercial competition and academic use only. The code licence does not relicense the data, and none of the data is included here. The evaluation tables in `eval/results/` contain only derived per-object statistics from the training set (areas, lengths, IoUs). They contain no polygons and no images. The figures in `eval/results/` and `report/` show small crops of training images to illustrate results.
