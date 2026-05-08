# Colab Training Workflow

Use this workflow for deep-generator experiments so local CPU-only or older hardware is not used for expensive training.

## Runtime

- Platform: Google Colab
- Runtime type: GPU
- Local use: tests, CLI reports, and notebook inspection only
- Artifact path in Colab: `artifacts/models/`

## Steps

1. Upload or clone this repository in Colab.
2. Install the project with the optional deep dependency:

   ```bash
   pip install -e ".[deep]"
   ```

3. Mount Google Drive if you want persistent artifacts.
4. Run `notebooks/04_colab_deep_generator_training.ipynb`.
5. Export model artifacts and evaluation summaries from Colab.
6. Download the trained artifacts to your local repo only after training is complete.

## Stop Conditions

Stop and reassess before paying for Colab Pro if:

- Free Colab cannot allocate a GPU.
- GPU memory is insufficient for the selected window size or batch size.
- Runtime disconnects before one experiment can finish.
- Training time exceeds the experiment budget without useful diagnostics.

## Guardrails

- Do not use the generated model to produce buy/sell recommendations.
- Keep evaluation focused on distribution similarity, volatility clustering, autocorrelation, drawdown behavior, and regime conditioning.
- Treat the deep model as research evidence, not the trusted source for product-grade stress testing.
