# Known Fact Hallucination

Paper (NAACL 2024): [On Large Language Models’ Hallucination with Regard to Known Facts](https://aclanthology.org/2024.naacl-long.60/)

Old Preprint [(arXiv)](https://arxiv.org/abs/2403.20009)

## Abstract

Large language models are successful in answering factoid questions but are also prone to hallucination. We investigate the phenomenon of LLMs possessing correct answer knowledge yet still hallucinating from the perspective of inference dynamics, an area not previously covered in studies on hallucinations. We are able to conduct this analysis via two key ideas. First, we identify the factual questions that query the same triplet knowledge but result in different answers. The difference between the model behaviors on the correct and incorrect outputs hence suggests the patterns when hallucinations happen. Second, to measure the pattern, we utilize mappings from the residual streams to vocabulary space. We reveal the different dynamics of the output token probabilities along the depths of layers between the correct and hallucinated cases. In hallucinated cases, the output token’s information rarely demonstrates abrupt increases and consistent superiority in the later stages of the model. Leveraging the dynamic curve as a feature, we build a classifier capable of accurately detecting hallucinatory predictions with an 88% success rate. Our study shed light on understanding the reasons for LLMs’ hallucinations on their known facts, and more importantly, on accurately predicting when they are hallucinating.

## Files and Structure

**Data Files:**

- `triplet_data.json`: This JSON file contains the dataset used in the paper. It is derived from the CounterFact dataset (source: [CounterFact](https://rome.baulab.info/data/dsets/counterfact.json)), with specific triplet knowledge queries selected and modified for our experiments.

**Utility Scripts:**

- `get_dist.py`: Extracts observation data on model states for various analyses.
- `data_utils.py`: Handles data preprocessing and transformations required by the analysis scripts.
- `plot_utils.py`: Provides plotting functions for visualization.

These utility scripts are used across the codebase to streamline specific functions.

**Main Analysis Scripts:**

- `get_model_output_example_opt.py`: Generates the model’s outputs for the triplet queries in `triplet_data.json`. It produces a JSONL file (with suffix `*hall.jsonl`) where different responses from multiple queries are saved.
- `get_lens_result_example_opt.py`: Processes model hidden states under two lens tools, generating probability change curves for each token in response to the queries.
- `example_plot.ipynb`: A Jupyter notebook for reproducing the probability change curves shown in the paper.
- `example_svm.ipynb`: A Jupyter notebook for performing SVM classification analysis, as shown in the paper’s results.

## Usage Instructions

1. **Generate Model Outputs**: Run `get_model_output_example_opt.py` to obtain model responses for the triplet queries in the dataset. The responses with 'known fact hallucination' will be saved in `*hall.jsonl` files.
2. **Analyze Hidden States with Lens Tools**: Run `get_lens_result_example_opt.py` to extract and analyze the token probability changes in the model’s hidden states, using two different lens tools.
3. **Generate Plots**: Open `example_plot.ipynb` to produce the probability change curves as shown in the paper.
4. **SVM Classification**: Use `example_svm.ipynb` to replicate the SVM classification analysis from the paper.

## Requirements

The code is written in Python, and you may need to install specific libraries (including checkpoints from [Tuned Lens](https://huggingface.co/spaces/AlignmentResearch/tuned-lens/tree/main/lens)) to run the analysis. Please check any additional requirements listed in each script or notebook.

## Citation

```latex
@inproceedings{jiang2024large,
  title={On Large Language Models’ Hallucination with Regard to Known Facts},
  author={Jiang, Che and Qi, Biqing and Hong, Xiangyu and Fu, Dayuan and Cheng, Yang and Meng, Fandong and Yu, Mo and Zhou, Bowen and Zhou, Jie},
  booktitle={Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
  pages={1041--1053},
  year={2024}
}
```