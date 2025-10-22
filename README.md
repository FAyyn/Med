# Med

在MMedPO目录下我存放了模型训练的代码，MedEvalKit目录下则是模型评估的代码。

模型训练的脚本均在/workspace/MMedPO/MMedPO/scripts目录下，可通过train_sft.sh脚本进行SFT训练，通过train_dpo_visual-text.sh进行DPO训练，GPU调用等训练参数可直接在脚本中进行修改。

SFT训练可直接采用DPO数据转换为sft数据，/workspace/MMedPO/MMedPO/scripts/train_sft.sh脚本中添加了数据格式的转换。

方式一pairs的构建脚本为/workspace/MMedPO/scripts/run_inference_visual_indirect.sh，使用/workspace/MMedPO/MMedPO/data/slake_dpo_weighted.json作为输入，进行tie计算后构建pairs，原图片和背景图片过大无法上传到仓库，因此可以直接使用生成好的数据集进行DPO或SSPO训练。

SSPO的训练脚本有三个，分别是
/workspace/MMedPO/MMedPO/scripts/train_sspo_adv.sh,这是用权重代替动态w计算loss的sspo
/workspace/MMedPO/MMedPO/scripts/train_sspo.sh,这是标准的sspo
/workspace/MMedPO/MMedPO/scripts/train_tie_sspo.sh，这是添加了动态w计算的sspo



方式一数据集路径为：/workspace/MMedPO/MMedPO/data/tie_dpo_dataset_method1_converted.json


评估则可以直接参考MedEvalKit的readme文件，在/workspace/MMedPO/MedEvalKit目录下。

/workspace/MMedPO/MMedPO/outputs 目录下存放着一些evaluate的结果以及通过TIE计算得到的pairs数据。

环境需求分别参考各自的requirements.txt文件。