# DVRF

Implementation of the paper **“Delta Velocity Rectified Flow for Text-to-Image Editing”**.


## Installation

```bash
git clone https://github.com/anonymousdvrf/delta-velocity-rectified-flow.git
cd delta-velocity-rectified-flow
pip install torch diffusers transformers accelerate sentencepiece protobuf
```

## Usage

In the exp.yaml file, enter the folder of the images you want to edit, scheduler_strategy, learning rate, number of optimization steps, CFG weights, $\eta$.

```bash
python edit.py --exp_yaml exp.yaml
```
