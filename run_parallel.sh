echo "12个实验后台执行开始, Compute Node: 209 ....."
# --enable_sdfusion --sdfuison_rate 0.1 --fusion_type "average" :: 
CUDA_VISIBLE_DEVICES=0 python main.py --model_name "prcv_Thumos14_base_act_sdfusion_12" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 40 --enc_layers 2 --dec_layers 5 --actionness_loss --enable_sdfusion --sdfuison_rate 0.1 --fusion_type "average" &
sleep 10
# --enable_sdfusion --sdfuison_rate 0.01 --fusion_type "average" :: 
CUDA_VISIBLE_DEVICES=1 python main.py --model_name "prcv_Thumos14_base_act_sdfusion_13" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 40 --enc_layers 2 --dec_layers 5 --actionness_loss --enable_sdfusion --sdfuison_rate 0.01 --fusion_type "average" &
sleep 10
# --enable_sdfusion --sdfuison_rate 0.1 --fusion_type "non_parameter" :: 
CUDA_VISIBLE_DEVICES=2 python main.py --model_name "prcv_Thumos14_base_act_sdfusion_14" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 40 --enc_layers 2 --dec_layers 5 --actionness_loss --enable_sdfusion --sdfuison_rate 0.1 --fusion_type "non_parameter" &
sleep 10
# --enable_sdfusion --sdfuison_rate 0.01 --fusion_type "non_parameter" :: 
CUDA_VISIBLE_DEVICES=3 python main.py --model_name "prcv_Thumos14_base_act_sdfusion_15" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 40 --enc_layers 2 --dec_layers 5 --actionness_loss --enable_sdfusion --sdfuison_rate 0.01 --fusion_type "non_parameter" &
sleep 10
# --enable_sdfusion --sdfuison_rate 0.1 --fusion_type "average" :: 
CUDA_VISIBLE_DEVICES=0 python main.py --model_name "prcv_ActivityNet13_base_act_sdfusion_12" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 5 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --actionness_loss --enable_sdfusion --sdfuison_rate 0.1 --fusion_type "average" &
sleep 10
# --enable_sdfusion --sdfuison_rate 0.01 --fusion_type "average" :: 
CUDA_VISIBLE_DEVICES=1 python main.py --model_name "prcv_ActivityNet13_base_act_sdfusion_13" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 5 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --actionness_loss --enable_sdfusion --sdfuison_rate 0.01 --fusion_type "average" &
sleep 10
# --enable_sdfusion --sdfuison_rate 0.1 --fusion_type "non_parameter" :: 26.7575
CUDA_VISIBLE_DEVICES=2 python main.py --model_name "prcv_ActivityNet13_base_act_sdfusion_14" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 5 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --actionness_loss --enable_sdfusion --sdfuison_rate 0.1 --fusion_type "non_parameter" &
sleep 10
# --enable_sdfusion --sdfuison_rate 0.01 --fusion_type "non_parameter" :: 26.7575
CUDA_VISIBLE_DEVICES=3 python main.py --model_name "prcv_ActivityNet13_base_act_sdfusion_15" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 5 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --actionness_loss --enable_sdfusion --sdfuison_rate 0.01 --fusion_type "non_parameter" &
sleep 10
# --enable_sdfusion --sdfuison_rate 0.1 --fusion_type "average" :: 20.85071
CUDA_VISIBLE_DEVICES=4 python main.py --model_name "prcv_Thumos14_50_base_act_sdfusion_4" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --save_result --batch_size 16 --lr 5e-5 --epochs 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --split_id 2 --actionness_loss --enable_sdfusion --sdfuison_rate 0.1 --fusion_type "average" &
sleep 10
# --enable_sdfusion --sdfuison_rate 0.01 --fusion_type "average" :: 20.85071
CUDA_VISIBLE_DEVICES=6 python main.py --model_name "prcv_Thumos14_50_base_act_sdfusion_5" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --save_result --batch_size 16 --lr 5e-5 --epochs 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --split_id 2 --actionness_loss --enable_sdfusion --sdfuison_rate 0.01 --fusion_type "average" &
sleep 10
# --enable_sdfusion --fusion_type "average" --sdfuison_rate 0.1 :: 
CUDA_VISIBLE_DEVICES=4 python main.py --model_name "prcv_ActivityNet13_50_base_act_sdfusion_4" --cfg_path "./config/ActivityNet13_CLIP_zs_50.yaml" --save_result --batch_size 16 --lr 5e-5 --epochs 100 --num_queries 5 --enc_layers 2 --dec_layers 3 --enable_backbone --lr_backbone 1e-2 --actionness_loss --enable_sdfusion --fusion_type "average" --sdfuison_rate 0.1 &
sleep 10
# --enable_sdfusion --fusion_type "average" --sdfuison_rate 0.01 :: 
CUDA_VISIBLE_DEVICES=6 python main.py --model_name "prcv_ActivityNet13_50_base_act_sdfusion_5" --cfg_path "./config/ActivityNet13_CLIP_zs_50.yaml" --save_result --batch_size 16 --lr 5e-5 --epochs 100 --num_queries 5 --enc_layers 2 --dec_layers 3 --enable_backbone --lr_backbone 1e-2 --actionness_loss --enable_sdfusion --fusion_type "average" --sdfuison_rate 0.01 &
wait
echo "12个实验后台执行结束, Compute Node: 209 ....."



