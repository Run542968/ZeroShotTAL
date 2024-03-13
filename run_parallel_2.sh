echo "4个实验后台执行开始, Compute Node: 211 ....."
# --enable_sdfusion
CUDA_VISIBLE_DEVICES=0 python main.py --model_name "prcv_ActivityNet13_base_act_sdfusion" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 5 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --actionness_loss --enable_sdfusion &
sleep 10
# --enable_sdfusion --sdfuison_rate 0.6
CUDA_VISIBLE_DEVICES=1 python main.py --model_name "prcv_ActivityNet13_base_act_sdfusion_1" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 5 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --actionness_loss --enable_sdfusion --sdfuison_rate 0.6 &
sleep 10
# --enable_sdfusion --sdfuison_rate 0.3
CUDA_VISIBLE_DEVICES=2 python main.py --model_name "prcv_ActivityNet13_base_act_sdfusion_2" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 5 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --actionness_loss --enable_sdfusion --sdfuison_rate 0.3 &
sleep 10
# --enable_sdfusion --sdfuison_rate 0.0
CUDA_VISIBLE_DEVICES=3 python main.py --model_name "prcv_ActivityNet13_base_act_sdfusion_3" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 5 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --actionness_loss --enable_sdfusion --sdfuison_rate 0.0 &
wait
echo "4个实验后台执行结束, Compute Node: 211 ....."

