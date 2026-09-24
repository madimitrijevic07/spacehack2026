from ultralytics import YOLO

model = YOLO(
    "/Users/sam/Documents/VS CODE/SPACEHACK2026/pipeline-detection-algorithm.pt"
)

image_folder = "test-non-training-dataset/not-a-pipeline" ## Images are inputed here
results = model.predict(image_folder, device="mps")

for res in results:
    top_class = res.names[res.probs.top1]
    confidence = res.probs.top1conf.item()
    filename = res.path.split("/")[-1]

    print(f"File: {filename}")
    print(f"Result: {top_class} ({confidence * 100:.2f}% confidence)\n")