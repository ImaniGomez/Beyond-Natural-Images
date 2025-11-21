# results/

This folder stores model outputs, logs, and metrics.

The benchmark script writes results as:

- `results/<task>/<model>/<image_name>_*.png`

For example, after running:

```bash
python -m src.benchmark \
    --task sr \
    --model bicubic_x4 \
    --image_path data/sample.jpeg \
    --save_output
