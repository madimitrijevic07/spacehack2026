from pathlib import Path
from string import ascii_uppercase

from PIL import Image
from ultralytics import YOLO

# ---------- Settings ----------
MODEL_PATH = "/Users/sam/Documents/VS CODE/SPACEHACK2026/pipeline-detection-algorithm.pt"
INPUT_FOLDER = Path("test-non-training-dataset/is-a-pipeline")  # images go in here
OUTPUT_FOLDER = Path("pipeline-tiles")                          # kept tiles are saved here
DEVICE = "mps"
GRID_SIZE = 5                     # 5x5 grid
PIPELINE_CLASS = "pipeline"       # the model's other class is "non-pipeline"
MIN_CONFIDENCE = 0.5              # ignore "pipeline" predictions below this confidence
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def has_pipeline(model, images):
    """Run the model on a list of PIL images and return a list of True/False,
    one per image, saying whether a pipeline was detected."""
    results = model.predict(images, device=DEVICE, verbose=False)
    flags = []
    for res in results:
        top_class = res.names[res.probs.top1]
        confidence = res.probs.top1conf.item()
        flags.append(top_class == PIPELINE_CLASS and confidence >= MIN_CONFIDENCE)
    return flags


def split_into_grid(image, n=GRID_SIZE):
    """Split a PIL image into an n x n grid.
    Returns a list of (label, tile) where label is like 'A1' (row letter + column number).
    Edges are computed with rounding so no pixels are lost if the size isn't divisible by n."""
    width, height = image.size
    xs = [round(i * width / n) for i in range(n + 1)]
    ys = [round(i * height / n) for i in range(n + 1)]

    tiles = []
    for row in range(n):
        for col in range(n):
            box = (xs[col], ys[row], xs[col + 1], ys[row + 1])
            label = f"{ascii_uppercase[row]}{col + 1}"
            tiles.append((label, image.crop(box)))
    return tiles


def process_image(model, path):
    """Returns the number of tiles saved for this image."""
    image = Image.open(path).convert("RGB")

    # Step 1: whole-image check
    if not has_pipeline(model, [image])[0]:
        print(f"{path.name}: no pipeline, skipped")
        return 0

    # Step 2: split, then check every tile in one batch
    tiles = split_into_grid(image)
    flags = has_pipeline(model, [tile for _, tile in tiles])

    # Step 3: save only the tiles with a pipeline
    saved = 0
    for (label, tile), keep in zip(tiles, flags):
        if keep:
            tile.save(OUTPUT_FOLDER / f"{path.stem}_{label}.png")
            saved += 1

    print(f"{path.name}: pipeline found, saved {saved}/{len(tiles)} tiles")
    return saved


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_PATH)
    print(f"Model classes: {model.names}")  # confirm PIPELINE_CLASS matches one of these

    image_paths = sorted(
        p for p in INPUT_FOLDER.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    total_saved = 0
    for path in image_paths:
        total_saved += process_image(model, path)

    print(f"\nDone. {len(image_paths)} images checked, {total_saved} tiles saved to {OUTPUT_FOLDER}/")


if __name__ == "__main__":
    main()
