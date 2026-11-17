# 🚀 Google Colab Training Guide

Training a Chess AI with Monte Carlo Tree Search (MCTS) and Neural Networks is incredibly computationally expensive. Using Google Colab is a **fantastic idea** because it gives you access to a free NVIDIA GPU (like a T4) which will speed up your training by 10x to 100x compared to a standard laptop CPU!

Here is the exact step-by-step process to train your AI on Google Colab and then bring the trained brain back to your local web application.

---

### Step 1: Open Google Colab and Enable the GPU
1. Go to [Google Colab](https://colab.research.google.com/) and click **New Notebook**.
2. In the top menu, go to **Runtime > Change runtime type**.
3. Under **Hardware accelerator**, select **T4 GPU** (or any available GPU) and click Save.

### Step 2: Upload Your Backend Files to Colab
On the left sidebar of Colab, click the **Folder icon (📁)** to open the Files panel. You need to drag and drop the following files from your local computer into that Colab file panel:

- `intelligent_training_pipeline.py`
- `neural_network.py`
- `self_play_training.py`
- `smart_database_parser.py`
- `training_monitor.py`
- `small_dataset.pgn` (or your full 2GB PGN if you're ready for the real deal!)

*(Note: If you are uploading the 2GB file, it's better to upload it to Google Drive first, and then mount your Drive in Colab).*

### Step 3: Install Required Libraries
Create a new code cell in Colab, paste the following command, and hit the **Play** button to install the required Python chess library:

```python
!pip install chess
```
*(PyTorch and Numpy are already pre-installed on Colab!)*

### Step 4: Start the Training
Create another code cell, paste the following command, and hit **Play**:

```python
!python intelligent_training_pipeline.py
```

The script will automatically detect the Colab GPU (CUDA) and start training incredibly fast. You will see the logs printing right there in your browser.

### Step 5: Download the Trained Model Back Home
Once the training finishes (or whenever you decide to stop it), the AI will have saved its "brain" into a new folder called `models/checkpoints/` inside Colab.

1. In the Colab Files panel (📁), refresh the folder view.
2. Open the `models` folder.
3. Download the latest `.pt` file (e.g., `best_model.pt` or `checkpoint_epoch_5.pt`).
4. Place that downloaded `.pt` file into the `models/` folder on your local Windows machine.

### Step 6: Test it Locally!
Now you can run your local Flask application, and your web-based Chess AI will play using the massive brain power it gained from the Google Colab supercomputers!
