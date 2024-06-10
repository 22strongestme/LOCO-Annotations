# LOCO-Annotations

LOCO-Annotations is a custom dataset specifically designed for evaluating the performance of logical anomaly detection frameworks like LogiCode. This dataset is an extension of the MVTec LOCO dataset and focuses exclusively on logical anomalies in industrial settings.

## Dataset Description

The LOCO-Annotations dataset comprises 2908 meticulously annotated images, divided into 1772 training images and 1136 testing images. It spans various categories, including:
- Breakfast boxes
- Screw bags
- Pushpins
- Splicing connectors
- Juice bottles

Each image in the dataset is accompanied by detailed JSON files that provide pixel-level object segmentation and ground truth annotations. These annotations are formatted as "Anomaly Type: Specific Reason" to clarify the reasoning behind each identified logical anomaly.

### Categories and Anomaly Types

1. **Quantity Anomalies**: These anomalies occur when the number of objects deviates from the expected count.
2. **Size Anomalies**: These anomalies are identified when the size of an object falls outside the normal range.
3. **Position Anomalies**: These occur when objects are misplaced, not adhering to the predefined positional rules.
4. **Matching Anomalies**: These anomalies arise when the characteristics of objects, such as color or type, do not match the expected criteria.

## Dataset Format

| Name  | Annotation Content  | Type               | Quantity |File Format |
|-------|----------------------|--------------------|----------|-------------|
| LOCO  | Logical Anomalies    | Various Categories | 1772+1136| JSON        |

<img width="698" alt="1717767114856" src="https://github.com/22strongestme/LOCO-Annotations/assets/55016895/d89eb8bf-6206-445d-bb72-039cd68bad7f">

## Usage

### Download

You can download the LOCO-Annotations dataset from the following link:

[Download LOCO-Annotations](https://drive.google.com/file/d/1LjsX6bmpC1IIipQSlcq1RwfNC78R0iU-/view?usp=drive_link)

### Evaluation

The dataset can be used for evaluating logical anomaly detection frameworks by comparing the output of the models with the ground truth annotations provided in the JSON files.


## Citation

If you use this dataset in your research, please cite the following paper:
[Logicode](https://arxiv.org/abs/2406.04687)
@misc{zhang2024logicode,
  title={LogiCode: an LLM-Driven Framework for Logical Anomaly Detection},
  author={Yiheng Zhang and Yunkang Cao and Xiaohao Xu and Weiming Shen},
  year={2024},
  eprint={2406.04687},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}

