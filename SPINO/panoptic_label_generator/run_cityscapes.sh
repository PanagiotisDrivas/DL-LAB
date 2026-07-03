conda activate spino
mkdir -p logs


# Inside panoptic_label_generator
# put dataset inside the panoptic_label_generator
#```
── cityscapes
   ├── camera
   │    └── ...
   ├── gtFine
   │    └── ...
   └── leftImg8bit_sequence
        └── ...
```


# for v2
# dirpath: "checkpoints/"
#   test_save_dir: "results/cityscapes"
python semantic_fine_tuning.py fit --trainer.devices [0] --config configs/semantic_cityscapes.yaml > logs/semantic_cityscapes.txt 2>&1
python boundary_fine_tuning.py fit --trainer.devices [0] --config configs/boundary_cityscapes.yaml > logs/boundary_cityscapes.txt 2>&1
python instance_clustering.py test --trainer.devices [0] --config configs/instance_cityscapes.yaml > logs/instance_cityscapes.txt 2>&1


#for v3
# also make changes in config 
# dirpath: "checkpoints_v3/"
#   test_save_dir: "results_v3/cityscapes"
python semantic_fine_tuning.py fit --trainer.devices [0] --config configs/semantic_cityscapes.yaml > logs/semantic_cityscapes_v3.txt 2>&1
python boundary_fine_tuning.py fit --trainer.devices [0] --config configs/boundary_cityscapes.yaml > logs/boundary_cityscapes_v3.txt 2>&1
python instance_clustering.py test --trainer.devices [0] --config configs/instance_cityscapes.yaml > logs/instance_cityscapes_v3.txt 2>&1


# Run from root of SPINO
# for v2
python -m panoptic_segmentation_model.scripts.evaluate_labels --dataset_name cityscapes --gpu_id 0 --gt_path panoptic_label_generator/cityscapes/ --pred_path panoptic_label_generator/results_v3/cityscapes/
#for v3
python -m panoptic_segmentation_model.scripts.evaluate_labels --dataset_name cityscapes --gpu_id 0 --gt_path panoptic_label_generator/cityscapes/ --pred_path panoptic_label_generator/results_v3/cityscapes/


# git clone https://github.com/facebookresearch/dinov3/tree/FINO into SPINO/panoptic_label_generator and name it dinov3_dict
# you may need to upgrade python from 3.8 to 3.13 for dinov3_dict to work (i made a new conda environment for it)
# added lines of code on top of semantic_fine_tuning.py and boundary_fine_tuning.py and instance_clustering.py to work with the new python version
# fixed ms_deform to work with the new python version