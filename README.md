# Implement of paper "Gaussian-Adaptive Fine-Tuning: Enhancing Large Language Models with Layer-Wise Learning Rate Optimization via Gaussian Functions"

## Draw LPD

```sh
python lpd.py
```

## GFFT

 After model loaded:

```python
from gfft import get_gfft_params

model = AutoModelForCausalLM.from_pretrained('...')
params = get_gfft_params(model, base_lr=lr, min_lr_scale=0.1, ...)
optimizer = Optimizer(params, ...)
```

## GPEFT

 After model loaded:

```python
from gpeft import get_gfft_params

model = AutoModelForCausalLM.from_pretrained('...')
gpeft(model)
optimizer = Optimizer(model.parameters(), lr=lr, ...)
```
