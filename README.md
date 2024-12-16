# Chinese Clinical Named Entity Recognition via Multi-head Self-attention Based BiLSTM-CRF
Welcome to the official code repository for the paper titled *"Chinese Clinical Named Entity Recognition via Multi-head Self-attention Based BiLSTM-CRF"*. This repository contains the implementation of the methods and experiments presented in the paper. The paper introduces a deep learning model/algorithm based on **TensorFlow** to greatly improve the performance of Chinese clinical named entity recognition. 

## Dependencies

Ensure that the following dependencies are installed:

- Python 3.6
- TensorFlow 
- Other dependencies (e.g., NumPy, SciPy, Matplotlib, etc.)

### Setting Up the Environment

1. Create a virtual environment:
   ```bash
   conda create my_env
   ```
2. Activate the virtual environment:
   ```bash
   conda activate my_env
   ```
   Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
### Usage
1. a test example
  - move to project root
  ```cd /project root```
  - run test using next flow
  ```nextflow run main.nf --mode val```
2. prediction example
  -	move to project root
  ```cd /project root```
  -	run using next flow（file.path is the abs path of text_file）
  ```nextflow run main.nf --mode predict –-text_path file.txt```

Paper Citation
This code implementation is based on the following paper:

```bibtex
@article{
author = {  An Ying and     Xia Xianyun and     Chen Xianlai and     Wu Fang-Xiang and Wang Jianxin},
title = {Chinese clinical named entity recognition via multi-head self-attention based BiLSTM-CRF},
journal = {Artificial Intelligence In Medicine},
volume = {127},
pages = {102282-102282},
year = {2022},
issn = {0933-3657},
}    
```
Please update the citation information with your actual paper details.

License
This project is licensed under the Apache-2.0 License. Please refer to the LICENSE file for more details.

Contact
Author: [Ying An]
Email: [anying.csu.edu.cn]
Feel free to submit issues or suggestions via GitHub Issues.
