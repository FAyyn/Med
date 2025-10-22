# Med

在MMedPO目录下我存放了模型训练的代码，MedEvalKit目录下则是模型评估的代码。

模型训练的脚本均在MMedPO/MMedPO/scripts目录下，可通过train_sft.sh脚本进行SFT训练，通过train_dpo_visual-text.sh进行DPO训练，GPU调用等训练参数可直接在脚本中进行修改。

SFT训练可直接采用DPO数据转换为sft数据，MMedPO/scripts/train_sft.sh脚本中添加了数据格式的转换。

方式一pairs的构建脚本为MMedPO/scripts/run_inference_visual_indirect.sh，使用MMedPO/MMedPO/data/slake_dpo_weighted.json作为输入，进行tie计算后构建pairs，原图片和背景图片过大无法上传到仓库，因此可以直接使用生成好的数据集进行DPO或SSPO训练。

SSPO的训练脚本有三个，分别是
MMedPO/scripts/train_sspo_adv.sh,这是用权重代替动态w计算loss的sspo
MMedPO/scripts/train_sspo.sh,这是标准的sspo
MMedPO/scripts/train_tie_sspo.sh，这是添加了动态w计算的sspo

SLAKE数据集以及处理好的遮挡背景图片地址：https://drive.google.com/file/d/1YJC7KspZohlfGxylWAKc6bVxfMZKGbYS/view?usp=drive_link

方式一数据集路径为：MMedPO/data/tie_dpo_dataset_method1_converted.json

以上的脚本均可以直接修改脚本中的参数，如数据集路径、模型路径、GPU调用等。

修改完成后使用 bash 运行脚本即可，如 bash MMedPO/scripts/train_dpo_visual-text.sh

评估则可以直接参考MedEvalKit的readme文件，在/workspace/MMedPO/MedEvalKit目录下。


环境需求分别参考各自的requirements.txt文件。

# Med

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
