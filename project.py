

# LML-diffusion-sampler
import os

# coco_data_path = "/home/liying/Documents/dataset/coco"
# coco_annotations_path = os.path.join(coco_data_path, "annotations")
# coco_val2014_images_path = os.path.join(coco_data_path, "val2014")

mme_data_path = "/home/liying/Documents/MLLMBenchMark/MME/extracted_data"

amber_data_path = "/home/liying/Documents/MLLMBenchMark/AMBER/data"

# llava_v15_7b_path = "/home/liying/Documents/llava-v1.5-7b"
# clip_vit_large_patch14_path = "/home/liying/Documents/clip-vit-large-patch14"
# anole_7b_v0_1_path = "/home/liying/Documents/Anole-7b-v0.1"
# volcano_7b_path = "/home/liying/Documents/volcano-7b"

data_path = "/home1/cjl/MM_2026/dataset"
coco_data_path = os.path.join(data_path, "ms_coco")
coco_annotations_path = os.path.join(coco_data_path, "annotations")
coco_val2014_images_path = os.path.join(coco_data_path, "val2014")

gqa_bench_path = os.path.join(data_path, "GQA_Bench")
clevr_v1_0_path = os.path.join(data_path, "CLEVR_v1.0/")
llava_instruct_150k_path = os.path.join(data_path, "LLaVA_INSTRUCT_150K/")
MMBench_path = os.path.join(data_path, "MMBench/")
MMBench_data_path = os.path.join(MMBench_path, "data")
vo_cot_path = os.path.join(data_path, "vo-cot", "VoCoT")
vsr_path = os.path.join(data_path, "VSR")
vsr_images_path = os.path.join(vsr_path, "images")
vsr_ramdom_path = os.path.join(vsr_path, "vsr_random")
embspatial_path = os.path.join(data_path, "embspatial")

models_path = "/home1/cjl/models"
llava_v15_7b_path = os.path.join(models_path, "llava-v1.5-7b")
clip_vit_large_patch14_path = os.path.join(models_path, "clip-vit-large-patch14")
# VoCoT 默认用 336；model.load_model 在 vision_encoder 为 openai/clip-* 时会改为此本地目录，避免联网拉 CLIP
clip_vit_large_patch14_336_path = os.path.join(models_path, "clip-vit-large-patch14-336")
anole_7b_v0_1_path = os.path.join(models_path, "anole-7b")
volcano_7b_kaist_ai_path = os.path.join(models_path, "volcano-7b_kaist_ai")
volcano_7b_luoruipu1_path = os.path.join(models_path, "volcano_7b_luoruipu1")

# 推理测试图片目录（可选）：在此目录放 default.jpg 等，run_inference 会优先用作默认图
_project_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(_project_dir, "images")