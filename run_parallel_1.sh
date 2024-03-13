echo "4个实验后台执行开始, Compute Node: 209 ....."
# --enable_sdfusion --fusion_type "non_parameter"
CUDA_VISIBLE_DEVICES=0 python main.py --model_name "prcv_Thumos14_base_act_sdfusion_4" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 40 --enc_layers 2 --dec_layers 5 --actionness_loss --enable_sdfusion --fusion_type "non_parameter" &
sleep 10
# --enable_sdfusion --sdfuison_rate 0.6
CUDA_VISIBLE_DEVICES=1 python main.py --model_name "prcv_Thumos14_base_act_sdfusion_5" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 40 --enc_layers 2 --dec_layers 5 --actionness_loss --enable_sdfusion --sdfuison_rate 0.6 --fusion_type "non_parameter" &
sleep 10
# --enable_sdfusion --sdfuison_rate 0.3
CUDA_VISIBLE_DEVICES=2 python main.py --model_name "prcv_Thumos14_base_act_sdfusion_6" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 40 --enc_layers 2 --dec_layers 5 --actionness_loss --enable_sdfusion --sdfuison_rate 0.3 --fusion_type "non_parameter" &
sleep 10
# --enable_sdfusion --sdfuison_rate 0
CUDA_VISIBLE_DEVICES=3 python main.py --model_name "prcv_Thumos14_base_act_sdfusion_7" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 40 --enc_layers 2 --dec_layers 5 --actionness_loss --enable_sdfusion --sdfuison_rate 0 --fusion_type "non_parameter" &
wait
echo "4个实验后台执行结束, Compute Node: 209 ....."

