echo "4个实验后台执行开始, Compute Node: 209 ....."
# --enable_sdfusion --fusion_type "non_parameter"
CUDA_VISIBLE_DEVICES=4 python main.py --model_name "prcv_ActivityNet13_base_act_sdfusion_4" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 5 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --actionness_loss --enable_sdfusion --fusion_type "non_parameter"
# --enable_sdfusion --sdfuison_rate 0.6 --fusion_type "non_parameter"
CUDA_VISIBLE_DEVICES=5 python main.py --model_name "prcv_ActivityNet13_base_act_sdfusion_5" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 5 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --actionness_loss --enable_sdfusion --sdfuison_rate 0.6 --fusion_type "non_parameter"
# --enable_sdfusion --sdfuison_rate 0.3 --fusion_type "non_parameter"
CUDA_VISIBLE_DEVICES=6 python main.py --model_name "prcv_ActivityNet13_base_act_sdfusion_6" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 5 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --actionness_loss --enable_sdfusion --sdfuison_rate 0.3 --fusion_type "non_parameter"
# --enable_sdfusion --sdfuison_rate 0.0 --fusion_type "non_parameter"
CUDA_VISIBLE_DEVICES=7 python main.py --model_name "prcv_ActivityNet13_base_act_sdfusion_7" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 5 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --actionness_loss --enable_sdfusion --sdfuison_rate 0.0 --fusion_type "non_parameter"
                    
wait
echo "4个实验后台执行结束, Compute Node: 209 ....."



