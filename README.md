# Retrospective Learning from Interactions

Website: <https://lil-lab.github.io/respect/>

## Setting up

Prepare conda environment

```bash
conda create -n respect python=3.9.18
pip install -r requirements.txt
pip install -e .
```

Download data

```python
from datasets import load_dataset

ds = load_dataset("lil-lab/respect", name="turn", split="train")
```

Download checkpoints

```python
from transformers import Idefics2ForConditionalGeneration
from peft import PeftModel

checkpoint = "HuggingFaceM4/idefics2-8b"
model_id = 'lil-lab/respect'

model = Idefics2ForConditionalGeneration.from_pretrained(
    checkpoint, torch_dtype=torch.bfloat16)
peft_model = PeftModel.from_pretrained(
    model, model_id, adapter_name="r6_bp", revision="r6_bp")  
```

## Reproducibility

To generate plots from the paper, run `analysis/plots.ipynb`.
