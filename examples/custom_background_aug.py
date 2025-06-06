import torch
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from datasets import load_dataset
import albumentations as A
from albumentations.core.transforms_interface import DualTransform
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.transforms import functional as F

# Augmentation transform operating on images and bounding boxes
class BackgroundBlur(DualTransform):
    """Blur image background outside bounding boxes."""

    def __init__(self, blur_limit=5, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.blur_limit = blur_limit

    def apply(self, image, bboxes=None, **params):
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        for box in bboxes or []:
            x, y, w, h = box
            draw.rectangle([int(x), int(y), int(x + w), int(y + h)], fill=255)
        blurred = image.filter(ImageFilter.GaussianBlur(self.blur_limit))
        background = Image.composite(blurred, image, ImageOps.invert(mask))
        image.paste(background, mask=ImageOps.invert(mask))
        return image

    def apply_to_bbox(self, bbox, **params):
        # bounding boxes remain unchanged
        return bbox

    def get_transform_init_args_names(self):
        return ("blur_limit",)


def build_dataset(img_folder, ann_file, transform):
    """Load COCO dataset with the datasets library and apply augmentation."""
    ds = load_dataset(
        "coco",
        "2017",
        data_dir=img_folder,
        annotation_file=ann_file,
        split="train",
    )

    aug = A.Compose(
        [transform],
        bbox_params=A.BboxParams(format="coco", label_fields=["labels"]),
    )

    def _apply(example):
        bboxes = example["objects"]["bbox"]
        labels = example["objects"]["category_id"]
        out = aug(image=example["image"], bboxes=bboxes, labels=labels)
        example["image"] = F.to_tensor(out["image"])
        example["boxes"] = torch.tensor(out["bboxes"], dtype=torch.float32)
        example["labels"] = torch.tensor(out["labels"])
        return example

    return ds.map(_apply)

def get_model(num_classes):
    model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torch.nn.Linear(in_features, num_classes)
    return model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--annotations", required=True)
    args = parser.parse_args()

    dataset = build_dataset(
        img_folder=args.images,
        ann_file=args.annotations,
        transform=BackgroundBlur(blur_limit=5),
    )
    print(f"Loaded {len(dataset)} augmented samples")
    model = get_model(num_classes=91)
    sample = dataset[0]
    print("Sample image tensor shape:", sample["image"].shape)
    print("Model ready for training")

