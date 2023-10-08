echo "9个实验后台执行开始, Compute Node: 212 ....."
#【base-1】--distillation_loss_coef 0.01 :: 
CUDA_VISIBLE_DEVICES=0 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v5_18" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --prefix "v5" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --exp_logit_scale --segmentation_loss --semantic_vhead_type "None" --augment_prompt_type "single" --distillation_loss --distillation_loss_coef 0.01 &
sleep 3
#【base-2】--segmentation_loss_coef 2 --distillation_loss_coef 0.1 :: 
CUDA_VISIBLE_DEVICES=1 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v5_19" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --prefix "v5" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --exp_logit_scale --segmentation_loss --semantic_vhead_type "Conv" --augment_prompt_type "single" --distillation_loss --lr_semantic_head 1e-7 --segmentation_loss_coef 2 --distillation_loss_coef 0.1 &
sleep 3
#【base-2】--segmentation_loss_coef 4 --distillation_loss_coef 0.1 :: 
CUDA_VISIBLE_DEVICES=2 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v5_20" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --prefix "v5" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --exp_logit_scale --segmentation_loss --semantic_vhead_type "Conv" --augment_prompt_type "single" --distillation_loss --lr_semantic_head 1e-7 --segmentation_loss_coef 4 --distillation_loss_coef 0.1 &
sleep 3
#【base-2】--segmentation_loss_coef 5 --distillation_loss_coef 0.1 :: 
CUDA_VISIBLE_DEVICES=3 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v5_21" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --prefix "v5" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --exp_logit_scale --segmentation_loss --semantic_vhead_type "Conv" --augment_prompt_type "single" --distillation_loss --lr_semantic_head 1e-7 --segmentation_loss_coef 5 --distillation_loss_coef 0.1 &
sleep 3
#【base-2】--segmentation_loss_coef 10 --distillation_loss_coef 0.1 :: 
CUDA_VISIBLE_DEVICES=4 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v5_22" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --prefix "v5" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --exp_logit_scale --segmentation_loss --semantic_vhead_type "Conv" --augment_prompt_type "single" --distillation_loss --lr_semantic_head 1e-7 --segmentation_loss_coef 10 --distillation_loss_coef 0.1 &
sleep 3
#【base-2】--segmentation_loss_coef 2 --distillation_loss_coef 0.01 :: 
CUDA_VISIBLE_DEVICES=5 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v5_23" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --prefix "v5" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --exp_logit_scale --segmentation_loss --semantic_vhead_type "Conv" --augment_prompt_type "single" --distillation_loss --lr_semantic_head 1e-7 --segmentation_loss_coef 2 --distillation_loss_coef 0.01 &
sleep 3
#【base-2】--segmentation_loss_coef 4 --distillation_loss_coef 0.01 :: 
CUDA_VISIBLE_DEVICES=6 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v5_24" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --prefix "v5" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --exp_logit_scale --segmentation_loss --semantic_vhead_type "Conv" --augment_prompt_type "single" --distillation_loss --lr_semantic_head 1e-7 --segmentation_loss_coef 4 --distillation_loss_coef 0.01 &
sleep 3
#【base-2】--segmentation_loss_coef 5 --distillation_loss_coef 0.01 :: 
CUDA_VISIBLE_DEVICES=7 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v5_25" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --prefix "v5" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --exp_logit_scale --segmentation_loss --semantic_vhead_type "Conv" --augment_prompt_type "single" --distillation_loss --lr_semantic_head 1e-7 --segmentation_loss_coef 5 --distillation_loss_coef 0.01 &
sleep 3
#【base-2】--segmentation_loss_coef 10 --distillation_loss_coef 0.01 :: 
CUDA_VISIBLE_DEVICES=7 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v5_26" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --prefix "v5" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --exp_logit_scale --segmentation_loss --semantic_vhead_type "Conv" --augment_prompt_type "single" --distillation_loss --lr_semantic_head 1e-7 --segmentation_loss_coef 10 --distillation_loss_coef 0.01 &
wait
echo "9个实验后台执行结束, Compute Node: 212 ....."

