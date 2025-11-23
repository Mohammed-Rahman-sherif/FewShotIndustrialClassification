# FewShotIndustrialClassification

This repository contains a few-shot industrial classification pipeline built on top of **OpenCLIP**.

## Installation

Install OpenCLIP:

```bash
pip install open_clip_torch
```

## Usage

Run the main script with your custom configuration:

```bash
python main.py --config configs/custom_dataset.yaml
```

Ensure that the config file points to the correct dataset paths and training settings.
