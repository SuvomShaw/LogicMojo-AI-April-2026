# Object Detection and YOLO 

## What is object detection?

Object detection finds objects inside an image.

For every object, it gives:

```text
WHAT it is       -> class name, for example person or bus
WHERE it is      -> bounding box
HOW SURE it is   -> confidence score
```

Example:

```text
person | [x1, y1, x2, y2] | confidence 0.91
bus    | [x1, y1, x2, y2] | confidence 0.88
```

### Diagram: one image, many detections

```text
Street image
┌──────────────────────────────────────────────┐
│  ┌──────────┐                                │
│  │ person   │  0.91                           │
│  └──────────┘                                │
│                         ┌──────────────────┐ │
│                         │       bus        │ │
│                         │       0.88       │ │
│                         └──────────────────┘ │
└──────────────────────────────────────────────┘

One image -> two separate objects -> two boxes -> two confidence scores
```

- The model does not return only the word `bus`.
- It returns one result for every detected object.

## Object detection vs classification

```text
Classification
Image -> one label
Example: "This image contains a bus."

Object detection
Image -> many objects, each with a box and label
Example: "There are three people and one bus."
```

## Important words

| Word | Full form | Simple meaning |
|---|---|---|
| AI | Artificial Intelligence | Making computers perform intelligent tasks |
| ML | Machine Learning | Learning patterns from examples |
| DL | Deep Learning | ML with neural networks |
| CV | Computer Vision | Understanding images and video |
| YOLO | You Only Look Once | A fast object-detection model |
| CNN | Convolutional Neural Network | A network that learns image features |
| IoU | Intersection over Union | How much two boxes overlap |
| NMS | Non-Maximum Suppression | Removes duplicate boxes |
| COCO | Common Objects in Context | A well-known object-detection dataset |
| mAP | mean Average Precision | A detection-quality metric |

## Bounding boxes

A bounding box is a rectangle around one object.

Two common formats are:

```text
[x1, y1, x2, y2]
x1, y1 = top-left corner
x2, y2 = bottom-right corner

[cx, cy, w, h]
cx, cy = centre of the box
w, h   = width and height
```

### Diagram: the two box formats describe the same object

```text
Pixel image

            y
            ↓
        (x1,y1) ┌──────────────┐
                │    object    │
                │      ●       │  ● = centre (cx, cy)
                │              │
                └──────────────┘ (x2,y2)

corner format      = [x1, y1, x2, y2]
centre-size format = [cx, cy, width, height]
```

- Corner format is easy for drawing a rectangle.
- Centre-size format is common in YOLO label files.

YOLO labels use normalized centre-size format:

```text
class_id  centre_x  centre_y  width  height
0         0.50      0.40      0.20   0.30
```

Normalized values are between 0 and 1. They work even when image sizes are different.

## IoU: checking box overlap

IoU compares a predicted box with the true box.

```text
IoU = overlap area / total area covered by both boxes

IoU = 1.0 -> boxes are exactly the same
IoU = 0.0 -> boxes do not overlap
```

### Diagram: IoU compares two boxes

```text
True box                    Predicted box
┌──────────────┐            ┌──────────────┐
│              │            │              │
│     ┌────────┼──────┐     │              │
│     │ overlap│      │     │              │
└─────┼────────┘      │     └──────────────┘
      │               │
      └───────────────┘

large overlap -> high IoU
small overlap -> low IoU
```

- IoU checks location quality, not class-name quality.
- Two boxes with the same class can still have poor IoU if one is in the wrong place.

A correct detection normally needs:

1. the correct class; and
2. sufficient IoU, such as IoU at least 0.50.

## How YOLO works

```text
┌─────────────┐
│ Input image │
└──────┬──────┘
       │ pixels
       ▼
┌──────────────────────────────────────────┐
│ BACKBONE                                 │
│ Finds edges, shapes and object features  │
└──────┬───────────────────────────────────┘
       │ feature maps at different sizes
       ▼
┌──────────────────────────────────────────┐
│ NECK                                     │
│ Mixes detail with wider image context    │
└──────┬───────────────────────────────────┘
       │ improved features
       ▼
┌──────────────────────────────────────────┐
│ HEAD                                     │
│ Predicts box + object score + class      │
└──────┬───────────────────────────────────┘
       │ many candidate boxes
       ▼
 confidence filter + NMS
       │
       ▼
 final detections
```

- Backbone = see useful image patterns.
- Neck = share information between different feature-map sizes.
- Head = turn features into boxes, scores, and class names.

### Backbone

The Backbone changes image pixels into useful feature maps.

Example: it may first notice edges, then a wheel shape, then features that suggest a bus.

### Neck

The Neck shares information between small-detail and large-context feature maps.

Example: a small object needs detailed features, while a large bus needs a wider view of the image.

### Head

The Head makes the final predictions.

For many locations in the image, it predicts a possible box, an object score, and class scores.

## Grids and many objects

YOLO uses feature grids inside the neural network. A grid is not a set of separate photos. It is a map of image features.

```text
Fine grid   -> helps find small objects
Medium grid -> helps find medium objects
Coarse grid -> helps find large objects
```

### Diagram: different grids help with different object sizes

```text
Same image: a bolt, a tool, and a large box

Fine feature grid              Medium feature grid          Coarse feature grid
many small cells               medium cells                 fewer large cells

┌─┬─┬─┬─┬─┬─┬─┬─┐              ┌──┬──┬──┬──┐                ┌────┬────┐
│ │●│ │ │ │ │ │ │              │  │ ●│  │  │                │    │ ●  │
├─┼─┼─┼─┼─┼─┼─┼─┤              ├──┼──┼──┼──┤                ├────┼────┤
│ │ │ │ │ │ │ │ │              │  │  │  │  │                │    │    │
└─┴─┴─┴─┴─┴─┴─┴─┘              └──┴──┴──┴──┘                └────┴────┘
small bolt                     medium tool                  large box
```

- Fine grids keep more location detail for small objects.
- Coarse grids see a wider part of the image for large objects.

Each grid produces candidate boxes. All candidates are collected into one list.

```text
small-object candidates
medium-object candidates
large-object candidates
            ↓
one candidate list
            ↓
remove weak boxes
            ↓
remove duplicate boxes with NMS
            ↓
final separate objects
```

The Neck combines feature information. It does not combine two real objects into one object.

## Confidence and NMS

### Confidence threshold

Confidence is the model's score for a prediction.

```text
Low threshold  -> more boxes, including more weak guesses
High threshold -> fewer boxes, but some real objects may be missed
```

### NMS

Several boxes can point to the same object. NMS keeps the highest-score box and removes close duplicates.

```text
person box A | score 0.92
person box B | score 0.76
high overlap -> keep A, remove B
```

### Diagram: NMS keeps one strong box

```text
Before NMS                         After NMS

person 0.92                        person 0.92
┌───────────────┐                  ┌───────────────┐
│ ┌───────────┐ │                  │               │
│ │ person    │ │       ---->      │    person     │
│ │ 0.76      │ │                  │    0.92       │
│ └───────────┘ │                  │               │
└───────────────┘                  └───────────────┘

two close boxes                    one final box
```

- The weak or duplicate box is removed.
- Boxes for different classes are normally checked separately.

## Real data used in the lab

The notebook uses **COCO8**, a very small real-data sample from COCO.

| Split | Images | Labelled objects |
|---|---:|---:|
| Train | 4 | 13 |
| Validation | 4 | 17 |
| Total | 8 | 30 |

The train images contain these labelled classes:

```text
zebra, giraffe, bowl, orange, broccoli, potted plant, vase
```

The model supports the full 80-class COCO list, but COCO8 is only a small pipeline demonstration. It is not enough to train a production model.

### Diagram: real-data workflow in this lab

```text
COCO8 images + label files
             │
             ▼
      pretrained YOLO model
             │
             ▼
   train split: 4 images, 13 objects
             │
             ▼
       short training run
             │
             ▼
validation split: 4 images, 17 objects
             │
             ▼
losses, mAP values and prediction images
```

- Training images update model weights.
- Validation images check how the model behaves on different images.

## The YOLO model used

```python
model = YOLO("yolo26n.pt")
```

| Part | Meaning |
|---|---|
| `YOLO` | YOLO model interface from Ultralytics |
| `yolo26` | YOLO model family used in this lab |
| `n` | Nano: small and lightweight |
| `.pt` | PyTorch weight file |

The model is pretrained. This means it already learned useful image patterns before the short COCO8 training run.

## Training settings in the lab

```python
model.train(
    data="coco8.yaml",
    epochs=3,
    imgsz=320,
    batch=4,
    device="cpu",
)
```

| Setting | Meaning |
|---|---|
| `data` | Dataset configuration file |
| `epochs=3` | The model sees training images three times |
| `imgsz=320` | Images are resized to 320 × 320 during training |
| `batch=4` | Four images are processed together |
| `device="cpu"` | Training runs on CPU |

COCO8 has four train images and batch size is four. Therefore one epoch has one training batch. Three epochs give about three model-update steps.

The training library automatically handles reading labels, batches, loss calculation, backpropagation, validation, checkpoint saving, and NMS.

## Losses during training

The training log may show:

| Loss | Meaning |
|---|---|
| Box loss | Error in box location and size |
| Class loss | Error in object class prediction |
| L1 loss | Absolute box-coordinate error |

Loss going down is useful, but it is not enough. Always inspect predicted images and validation metrics too.

## Optimizer

The notebook lets Ultralytics choose an optimizer automatically. In the verified run, it selected **AdamW**.

AdamW updates model weights to reduce losses. The exact learning rate can change with the installed library and training setup.

## Reading prediction results

After prediction, YOLO returns a `Results` object.

```text
result.boxes.xyxy -> box corners
result.boxes.conf -> confidence scores
result.boxes.cls  -> class IDs
result.names      -> class names
result.plot()     -> image with boxes and labels
```

## Precision, recall and mAP

```text
Precision = correct reported detections / all reported detections
Recall    = correct detected objects / all real objects
```

- High precision: reported boxes are usually correct.
- High recall: the model finds most real objects.
- mAP: summarizes detection quality across classes and IoU thresholds.

## Important limitations

- COCO8 has only eight images, so its metrics are unstable.
- Three epochs are useful for learning the workflow, not proving accuracy.
- A real project needs representative train, validation and test images.
- Good labels are as important as the model.
- Real testing should include difficult lighting, blur, crowding and object-size changes.

## Quick recap

```text
Object detection = class + box + confidence
YOLO             = fast one-pass object detection
Backbone         = finds features
Neck             = mixes features from different sizes
Head             = predicts boxes and classes
Confidence       = removes weak guesses
NMS              = removes duplicate boxes
COCO8            = small real-data pipeline sample
```
