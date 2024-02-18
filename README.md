# Problem Set #5: Project Outline and Video Pitch

In this problem set, I will be creating a project outline and a video pitch to present my project. This is an essential step to guide my presentation, increase accessibility, and have a draft for my written project report.

## Instructions:

1. Produce a Script: To guide my presentation, increase accessibility, and serve as a draft for my written project report.

2. Include at Least One Illustration: To enhance the visual appeal and clarity of my presentation.

3. Stay Within Time Limit: To maintain audience engagement and adhere to guidelines.

4. Design and Produce a Video: Based on the provided script to enable the audience to learn about:

- Problem definition
- Question
- State-of-the-art
- Nexus that enables the project
- Hypothesis
- Methods
  - Which methods
  - Why this choice
  - What is your data?
- Expected results
- Intellectual merit
- Broader impacts

5. Upload the Video and Script: Once completed, I will upload both the video and script to Canvas for evaluation.

## Operational Instructions:
### Installation of Miniconda:
Download Miniconda installation script:
```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```
Change the script permissions:
```bash
chmod +x Miniconda3-latest-Linux-x86_64.sh
```
Execute the installation script:
```bash
./Miniconda3-latest-Linux-x86_64.sh
```
Environment Setup:
Create the environment using the provided configuration file:
```bash
conda env create -f environment.yaml
```
Activate the created environment:
```bash
conda activate engg-107
```
### Graph Dimension Reduction:
Navigate to the directory containing the code:
```bash
cd ./scratch/f006dg0/mcas-gmra/pymm-gmra/experiments
```
Execute the first Python script to build the cover tree:
```bash
python ./graphs/covertree_build.py ./graphs/results --data_file ./graphs/n2v/{trace_test_32}.txt
```
Run the second Python script to execute the GMRA algorithm and generate low-dimensional embeddings:
```bash
python ./graphs/dram/gmra.py ./graphs/results/{target}_covertree.json
```
### Link Prediction:
Convert the input data to pickle format:
```bash
python provenance/link_prediction/src/convert_darpatc.py
```
Execute the link prediction script:
```bash
python provenance/link_prediction/src/link_pred_node2vec.py
```
### Anomaly Detection:
Note: To be completed