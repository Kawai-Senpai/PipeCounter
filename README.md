# YOLO-Based Pipe Counter

This project leverages YOLOv8, a state-of-the-art object detection and segmentation algorithm, to automatically detect and count pipes from images developed by [*Ranit Bhowmick*](https://www.linkedin.com/in/ranitbhowmick/) & [*Sayanti Chatterjee*](https://www.linkedin.com/in/sayantichatterjee/). The system is trained to detect various types of pipes in industrial settings and can be used for monitoring, inspection, and inventory management.

### Results

<div style="display: flex; gap: 10px; flex-wrap: wrap;">
<img src="https://github.com/Kawai-Senpai/PipeCounter/blob/0ccc3ee19f419823defbd01c807a620ad004f868/results/train_batch0.jpg" alt="train_batch0" width="200"/>
<img src="https://github.com/Kawai-Senpai/PipeCounter/blob/0ccc3ee19f419823defbd01c807a620ad004f868/results/train_batch1.jpg" alt="train_batch1" width="200"/>
<img src="https://github.com/Kawai-Senpai/PipeCounter/blob/0ccc3ee19f419823defbd01c807a620ad004f868/results/train_batch2.jpg" alt="train_batch2" width="200"/>
<img src="https://github.com/Kawai-Senpai/PipeCounter/blob/0ccc3ee19f419823defbd01c807a620ad004f868/results/train_batch360.jpg" alt="train_batch360" width="200"/>
<img src="https://github.com/Kawai-Senpai/PipeCounter/blob/0ccc3ee19f419823defbd01c807a620ad004f868/results/train_batch361.jpg" alt="train_batch361" width="200"/>
<img src="https://github.com/Kawai-Senpai/PipeCounter/blob/0ccc3ee19f419823defbd01c807a620ad004f868/results/train_batch362.jpg" alt="train_batch362" width="200"/>
<img src="https://github.com/Kawai-Senpai/PipeCounter/blob/0ccc3ee19f419823defbd01c807a620ad004f868/results/val_batch1_labels.jpg" alt="Batch1 Lables" width="200"/>
<img src="https://github.com/Kawai-Senpai/PipeCounter/blob/0ccc3ee19f419823defbd01c807a620ad004f868/results/val_batch1_pred.jpg" 
alt="Batch1 Predictions" width="200"/>
</div>

### Statistics

<div style="display: flex; gap: 10px; flex-wrap: wrap;">
<img src="https://github.com/Kawai-Senpai/PipeCounter/blob/2555e6317e61398ff6885838317ee34ba109a195/results/confusion_matrix_normalized.png" alt="confusion_matrix_normalized" width="200"/>
<img src="https://github.com/Kawai-Senpai/PipeCounter/blob/2555e6317e61398ff6885838317ee34ba109a195/results/labels.jpg" 
alt="labels" width="200"/>
<img src="https://github.com/Kawai-Senpai/PipeCounter/blob/2555e6317e61398ff6885838317ee34ba109a195/results/labels_correlogram.jpg" alt="labels_correlogram" width="200"/>
<img src="https://github.com/Kawai-Senpai/PipeCounter/blob/2555e6317e61398ff6885838317ee34ba109a195/results/results.png" 
alt="results" width="200"/>
<img src="https://github.com/Kawai-Senpai/PipeCounter/blob/2555e6317e61398ff6885838317ee34ba109a195/results/F1_curve.png" 
alt="F1_curve" width="200"/>
<img src="https://github.com/Kawai-Senpai/PipeCounter/blob/2555e6317e61398ff6885838317ee34ba109a195/results/PR_curve.png" 
alt="PR_curve" width="200"/>
<img src="https://github.com/Kawai-Senpai/PipeCounter/blob/2555e6317e61398ff6885838317ee34ba109a195/results/P_curve.png" 
alt="P_curve" width="200"/>
<img src="https://github.com/Kawai-Senpai/PipeCounter/blob/2555e6317e61398ff6885838317ee34ba109a195/results/R_curve.png" 
alt="R_curve" width="200"/>
</div>

## Project Overview

- **YOLOv8 Model**: Utilizes a pretrained YOLOv8 model for high-precision detection.
- **Training Dataset**: Custom dataset (`pipe.yaml`) used for training the model to detect pipes.
- **Batched Inference**: Perform inference on batches of images to efficiently detect pipes in real time.

## Features

- **Pipe Detection**: Detects pipes and returns bounding boxes, masks, and keypoints for each instance.
- **Real-Time Inference**: Capable of processing multiple images in parallel.
- **Custom Training**: Trained on a custom dataset to improve detection accuracy in specific environments.
- **Save & Display Results**: Saves detection results and allows visualization of the bounding boxes and masks.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Kawai-Senpai/PipeCounter.git
   ```

2. Install dependencies:
   ```bash
   pip install ultralytics
   ```

3. Download the pretrained model weights:
   - Place the `yolov8n.pt` file in the working directory.

4. (Optional) If training from scratch, update the `pipe.yaml` file to point to your dataset.

## How to Run

### Inference
To run inference on a list of images:

```python
from ultralytics import YOLO

# Load the pretrained model
model = YOLO('yolov8n.pt')

# Perform inference on a batch of images
results = model(['pipe/train/images/DJI_20240220122723_0022_D.JPG'], device=0)

# Display and save results
for result in results:
    result.show()  # Display bounding boxes and masks
    result.save('result.jpg')  # Save results to disk
```

### Training
To train the model on the custom pipe dataset:

```python
# Train for 100 epochs with a batch size of 5
model.train(data='pipe.yaml', epochs=100, imgsz=640, batch=5, device=0)
```

### Load a Trained Model
To load and test a model trained from a previous session:

```python
# Load trained model weights
model = YOLO('runs/detect/train3/weights/best.pt')

# Perform inference
results = model(['path_to_image.jpg'])
results.show()
```

## Training Metrics

- **Epochs**: 100
- **Batch Size**: 5
- **Precision**: Achieved a precision of 0.934 on the validation set.
- **mAP (Mean Average Precision)**: Reached 0.924 for mAP50 and 0.633 for mAP50-95.

## Dataset

The dataset consists of images of pipes in various environments, labeled with bounding boxes for training the model to detect and count pipes. The dataset is defined in the `pipe.yaml` file, with separate training and validation splits.

## Results

After training, the model achieved high accuracy in detecting pipes across a variety of test images. The final model shows a precision of **0.934** and mAP of **0.924**, making it suitable for industrial applications.


### Results

<div style="display: flex; gap: 10px; flex-wrap: wrap;">
<img src="https://github.com/Kawai-Senpai/PipeCounter/blob/0ccc3ee19f419823defbd01c807a620ad004f868/results/train_batch0.jpg" alt="train_batch0" width="200"/>
<img src="https://github.com/Kawai-Senpai/PipeCounter/blob/0ccc3ee19f419823defbd01c807a620ad004f868/results/train_batch1.jpg" alt="train_batch1" width="200"/>
<img src="https://github.com/Kawai-Senpai/PipeCounter/blob/0ccc3ee19f419823defbd01c807a620ad004f868/results/train_batch2.jpg" alt="train_batch2" width="200"/>
<img src="https://github.com/Kawai-Senpai/PipeCounter/blob/0ccc3ee19f419823defbd01c807a620ad004f868/results/train_batch360.jpg" alt="train_batch360" width="200"/>
<img src="https://github.com/Kawai-Senpai/PipeCounter/blob/0ccc3ee19f419823defbd01c807a620ad004f868/results/train_batch361.jpg" alt="train_batch361" width="200"/>
<img src="https://github.com/Kawai-Senpai/PipeCounter/blob/0ccc3ee19f419823defbd01c807a620ad004f868/results/train_batch362.jpg" alt="train_batch362" width="200"/>
<img src="https://github.com/Kawai-Senpai/PipeCounter/blob/0ccc3ee19f419823defbd01c807a620ad004f868/results/val_batch1_labels.jpg" alt="Batch1 Lables" width="200"/>
<img src="https://github.com/Kawai-Senpai/PipeCounter/blob/0ccc3ee19f419823defbd01c807a620ad004f868/results/val_batch1_pred.jpg" 
alt="Batch1 Predictions" width="200"/>
</div>

### Statistics

<div style="display: flex; gap: 10px; flex-wrap: wrap;">
<img src="https://github.com/Kawai-Senpai/PipeCounter/blob/2555e6317e61398ff6885838317ee34ba109a195/results/confusion_matrix_normalized.png" alt="confusion_matrix_normalized" width="200"/>
<img src="https://github.com/Kawai-Senpai/PipeCounter/blob/2555e6317e61398ff6885838317ee34ba109a195/results/labels.jpg" 
alt="labels" width="200"/>
<img src="https://github.com/Kawai-Senpai/PipeCounter/blob/2555e6317e61398ff6885838317ee34ba109a195/results/labels_correlogram.jpg" alt="labels_correlogram" width="200"/>
<img src="https://github.com/Kawai-Senpai/PipeCounter/blob/2555e6317e61398ff6885838317ee34ba109a195/results/results.png" 
alt="results" width="200"/>
<img src="https://github.com/Kawai-Senpai/PipeCounter/blob/2555e6317e61398ff6885838317ee34ba109a195/results/F1_curve.png" 
alt="F1_curve" width="200"/>
<img src="https://github.com/Kawai-Senpai/PipeCounter/blob/2555e6317e61398ff6885838317ee34ba109a195/results/PR_curve.png" 
alt="PR_curve" width="200"/>
<img src="https://github.com/Kawai-Senpai/PipeCounter/blob/2555e6317e61398ff6885838317ee34ba109a195/results/P_curve.png" 
alt="P_curve" width="200"/>
<img src="https://github.com/Kawai-Senpai/PipeCounter/blob/2555e6317e61398ff6885838317ee34ba109a195/results/R_curve.png" 
alt="R_curve" width="200"/>
</div>



## Future Improvements

- **Fine-tuning**: Further fine-tuning on a more diverse dataset to improve robustness.
- **Video Processing**: Extend the project to perform real-time video detection.
- **Deployment**: Package the model for deployment in embedded systems or cloud environments.