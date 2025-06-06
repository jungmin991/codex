# Codex

This repository includes example code for using Faster R-CNN with custom data
augmentation on the COCO dataset.

See `examples/custom_background_aug.py` for a script that uses the
`datasets` library together with an Albumentations `DualTransform` to blur the
background outside COCO bounding boxes before training with a lightweight
Faster R-CNN model.
