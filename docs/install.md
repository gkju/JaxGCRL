## Environment creation
For CUDA 13 on Linux or WSL2, install JAX's CUDA extra together with JaxGCRL:
```bash
pip install -e ".[train]" "jax[cuda13]"
```

If you prefer conda, the included `environment.yml` delegates the JAX CUDA
runtime to the same pip extra:
```bash
conda env create -f environment.yml
```

To check whether installation worked, run a test experiment using `./scripts/train.sh` file:

```bash
chmod +x ./scripts/train.sh; ./scripts/train.sh
```
