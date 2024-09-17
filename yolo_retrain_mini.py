
from ultralytics import YOLO

# Load a model
#model = YOLO('yolov8n.yaml')  # build a new model from YAML
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
#model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights


# Train the model
results = model.train(data='pipe.yaml',epochs=100, imgsz=640, device=0, batch = 5)  # train a new model for 100 epochs


