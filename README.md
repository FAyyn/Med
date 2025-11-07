# Med

The evaluation script has been changed, I added the checkpoint supply, now you could evaluate the model by using the checkpoint. 

I also leaved some comments in the scripts about the dataset path, I hope they would help you.

If there are any other questions you could leave a comment in the issue.

I have stored the model training code in the MMedPO directory, while the model evaluation code resides in the MedEvalKit directory.

All model training scripts are located in the MMedPO/MMedPO/scripts directory. SFT training can be performed using the train_sft.sh script, and DPO training via train_dpo_visual-text.sh. Training parameters such as GPU invocation can be directly modified within the scripts.

SFT training can directly utilize DPO data converted to SFT format. The data format conversion is implemented within the MMedPO/scripts/train_sft.sh script.

Method 1: The script for constructing pairs is MMedPO/scripts/run_inference_visual_indirect.sh. It uses MMedPO/data/slake_dpo_weighted.json as input, performs tie calculations, and constructs pairs. Since the original images and background images are too large to upload to the repository, you can directly use the pre-generated dataset for DPO or SSPO training.

There are three training scripts for SSPO:
MMedPO/scripts/train_sspo_adv.sh: SSPO using weights instead of dynamic w for loss calculation
MMedPO/scripts/train_sspo.sh: Standard SSPO
MMedPO/scripts/train_tie_sspo.sh: SSPO with dynamic w calculation added

SLAKE dataset and the processed occluded background image URLs：https://drive.google.com/file/d/1YJC7KspZohlfGxylWAKc6bVxfMZKGbYS/view?usp=drive_link

Method 1 dataset path: MMedPO/data/tie_dpo_dataset_method1_converted.json

All scripts above allow direct modification of parameters within the script, such as dataset paths, model paths, GPU invocation, etc.

After modifications, run the script using bash, e.g., bash MMedPO/scripts/train_dpo_visual-text.sh

For evaluation, refer directly to the MedEvalKit readme file located in the /workspace/MMedPO/MedEvalKit directory.


Environment requirements are detailed in the respective requirements.txt files.
