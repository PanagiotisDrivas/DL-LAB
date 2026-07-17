Here is the final `README.md` updated for your complete pipeline:

* Generate **instance reference bank**
* Generate **semantic reference bank**
* Run INSID3 inference
* Select reference type (`semantic` / `instance`)
* Single image or batch inference
* Generate polygon JSON
* Generate masks from JSON

```markdown
# INSID3 Reference-Based Segmentation Pipeline

This project uses **INSID3 reference-based segmentation**.

The pipeline supports two types of reference banks:

1. **Instance Reference Bank**
2. **Semantic Reference Bank**

The inference pipeline:

```

Reference Bank
|
|
v
INSID3 inference
|
|
v
Class masks
|
|
v
Polygon JSON annotations
|
|
v
Binary masks

```

---

# Directory Structure

Recommended project structure:

```

project/

├── infer.py
├── create_instance_reference_bank.py
├── create_semantic_reference_bank.py
├── create_mask.py
│
├── input/
│   ├── images/
│   │   ├── image1.png
│   │   └── image2.png
│   │
│   ├── annotations/
│   └── masks/
│
├── reference_bank/
│
│   ├── instance/
│   │
│   └── semantic/
│
└── aggregated_masks/

```

---

# 1. Generate Reference Banks

Reference banks are created from Cityscapes polygons.

The reference contains:

```

reference image

*

binary mask

```

Example:

```

car/
car_000_img.png
car_000_mask.png

```

---

# 2. Instance Reference Bank

## Description

Each reference contains **one object instance**.

Example:

A Cityscapes image:

```

car
car
car

```

creates:

```

car/

car_000_img.png
car_000_mask.png

car_001_img.png
car_001_mask.png

car_002_img.png
car_002_mask.png

````

Each mask contains only one car.

---

## Generate Instance Bank

Run:

```bash
python create_instance_reference_bank.py
````

Output:

```
reference_bank_instance/

├── car/
│
│── car_000_img.png
│── car_000_mask.png
│
├── person/
│
└── building/
```

---

# 3. Semantic Reference Bank

## Description

A semantic reference contains **all instances of the same class in one image**.

Example:

Input image:

```
car
car
car
car
```

creates:

```
car_000_img.png

car_000_mask.png
```

where the mask contains all cars:

```
      ███

███        ███


       ███
```

---

## Generate Semantic Bank

Run:

```bash
python create_semantic_reference_bank.py
```

Default:

```python
min_instances=3
```

Meaning:

Only images containing at least 3 objects of the same class are selected.

Example:

```
car: 7 objects  -> saved

person: 2 objects -> ignored
```

---

# 4. Prepare Input Images

Place images here:

```
input/images/

image1.png
image2.png
image3.png
```

---

# 5. Run INSID3 Inference

## Batch inference

Default:

```bash
python infer.py
```

This uses:

```
model:
small


reference:
semantic


references per class:
7
```

---

# 6. Select Reference Type

## Semantic reference bank

Recommended for dense scenes:

```bash
python infer.py \
--type semantic
```

Uses:

```
REFRENCE_SEMANTIC_BANK_ROOT
```

---

## Instance reference bank

```bash
python infer.py \
--type instance
```

Uses:

```
REFRENCE_INSTANCE_BANK_ROOT
```

---

# 7. Single Image Inference

Run one image:

```bash
python infer.py \
--image input/images/test.png
```

Example:

```bash
python infer.py \
--image input/images/test.png \
--type semantic
```

---

# 8. Select Number of References

Default:

```
7 references per class
```

Change:

```bash
python infer.py \
--max-instances 3
```

Example:

```
car/

car_000
car_001
car_002
```

Only these references are used.

---

# 9. Change INSID3 Model

Small:

```bash
python infer.py \
--model_size small
```

Base:

```bash
python infer.py \
--model_size base
```

Large:

```bash
python infer.py \
--model_size large
```

---

# 10. Output After Inference

After running:

```
aggregated_masks/
```

contains:

```
aggregated_masks/

└── image1_small/

    ├── car.png

    ├── person.png

    ├── building.png

    └── semantic_mask.png
```

Each class mask:

```
255 = predicted class

0 = background
```

---

# 11. Generate Polygon JSON

The inference automatically creates:

```
input/annotations/
```

Example:

```
input/

├── annotations/

│   └── image1.json
```

JSON format:

```json
{
    "imgHeight":1024,
    "imgWidth":2048,

    "objects":[
        {
            "label":"car",

            "polygon":[
                [100,200],
                [150,250],
                [200,300]
            ]
        }
    ]
}
```

---

# 12. Generate Binary Masks From JSON

Run:

```bash
python create_mask.py \
--json-dir input/annotations \
--output-dir input/masks
```

Output:

```
input/masks/

image1/

    car.png

    person.png

    building.png
```

Mask:

```
white  = object

black  = background
```

---

# 13. Remove Intermediate Masks

Normally:

```
aggregated_masks/
```

is kept.

To remove automatically:

```bash
python infer.py \
--remove-masks
```

Example:

```bash
python infer.py \
--image input/images/test.png \
--type semantic \
--remove-masks
```

After JSON creation:

```
aggregated_masks/image1_small/
```

is deleted.

---

# Complete Workflow

## Step 1

Create instance bank:

```bash
python create_instance_reference_bank.py
```

or semantic bank:

```bash
python create_semantic_reference_bank.py
```

---

## Step 2

Put test images:

```
input/images/
```

---

## Step 3

Run INSID3:

```bash
python infer.py \
--type semantic
```

---

## Step 4

Convert annotations to masks:

```bash
python create_mask.py \
--json-dir input/annotations \
--output-dir input/masks
```

---

# Recommended Settings

For normal street scenes:

```
Reference:
semantic


References/class:
~3


Model:
base
```

For small objects:

```
Reference:
instance


References/class:
~3
```

```
semantic reference bank
        +
INSID3
        +
polygon extraction
```

produces final segmentation annotations automatically.

