
# ChestAI

A minimal Streamlit app to classify chest X-rays as **Pneumonia: Positive** or **Pneumonia: Negative** using a PyTorch model.

## Quick start (local)

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Put your model file (e.g., `ChestAI_final_project_esnet_gan.pkl`) in the same folder as `app.py`.

3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Streamlit Community Cloud

- Upload `app.py`, `requirements.txt`, and your model checkpoint file to the same repo.
- Set the app entry point to `app.py`.

## Notes

- The app tries to load either a full PyTorch model object or a `state_dict`. For a `state_dict`, it will build a reasonable backbone (ResNet18 by default, or inferred from filename) and attach a 1-unit head for binary classification.
- For 2-logit classifiers, it assumes class index 1 corresponds to "Pneumonia". For single-logit models, it uses a sigmoid and applies a 0.5 threshold.
