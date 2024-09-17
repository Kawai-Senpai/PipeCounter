# %%
from ultralytics import YOLO

# Load a model
#model = YOLO('yolov8n.yaml')  # build a new model from YAML
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
#model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# %%
# Run batched inference on a list of images
results = model(['pipe/train/images/DJI_20240220122723_0022_D.JPG'], device = 0)  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    result.show()  # display to screen
    result.save(filename='result.jpg')  # save to disk
    # Print the class labels and the bounding box coordinates
    labels = boxes.cls  # a tensor of shape (N,)
    coords = boxes.xyxy  # a tensor of shape (N, 4)
    print(labels)
    print(coords)

# %%
# Train the model
results = model.train(data='pipe.yaml',epochs=100, imgsz=640, device=0, batch = 5)  # train a new model for 100 epochs

# %%
#load the model from runs\detect\train6
model = YOLO('runs/detect/train3/weights/best.pt')  # load a pretrained model (recommended for training)


