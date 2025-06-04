# RL

This repository demonstrates a minimal reinforcement learning loop with a simple
generative reward model. The training script writes logs that can be visualized
with a lightweight Streamlit app.

## Usage

Install dependencies (PyTorch and Streamlit)::

    pip install torch streamlit

Run training to generate `training.log`::

    python train.py

Launch the visualization web app::

    streamlit run streamlit_app.py

Both scripts are intentionally small and well commented for future expansion.
