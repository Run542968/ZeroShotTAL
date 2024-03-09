echo "8个实验后台执行开始, Compute Node: 209 ....."
# --enable_element
CUDA_VISIBLE_DEVICES=0 python main.py --model_name "prcv_Thumos14_base_act_dis_sal_dyn" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 40 --enc_layers 2 --dec_layers 5 --actionness_loss --distillation_loss --distillation_loss_coef 0.1 --salient_loss --salient_loss_coef 2 --enable_element &
sleep 10
# --enable_element --element_rate 0.9
CUDA_VISIBLE_DEVICES=1 python main.py --model_name "prcv_Thumos14_base_act_dis_sal_dyn_1" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 40 --enc_layers 2 --dec_layers 5 --actionness_loss --distillation_loss --distillation_loss_coef 0.1 --salient_loss --salient_loss_coef 2 --enable_element --element_rate 0.9 &
sleep 10
# --enable_element --element_rate 0.75
CUDA_VISIBLE_DEVICES=2 python main.py --model_name "prcv_Thumos14_base_act_dis_sal_dyn_2" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 40 --enc_layers 2 --dec_layers 5 --actionness_loss --distillation_loss --distillation_loss_coef 0.1 --salient_loss --salient_loss_coef 2 --enable_element --element_rate 0.75 &
sleep 10
# --enable_element --element_rate 0.6
CUDA_VISIBLE_DEVICES=3 python main.py --model_name "prcv_Thumos14_base_act_dis_sal_dyn_3" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 40 --enc_layers 2 --dec_layers 5 --actionness_loss --distillation_loss --distillation_loss_coef 0.1 --salient_loss --salient_loss_coef 2 --enable_element --element_rate 0.6 &
sleep 10
# --enable_element --element_rate 0.5
CUDA_VISIBLE_DEVICES=4 python main.py --model_name "prcv_Thumos14_base_act_dis_sal_dyn_4" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 40 --enc_layers 2 --dec_layers 5 --actionness_loss --distillation_loss --distillation_loss_coef 0.1 --salient_loss --salient_loss_coef 2 --enable_element --element_rate 0.5 &
sleep 10
# --enable_element --element_rate 0.3
CUDA_VISIBLE_DEVICES=5 python main.py --model_name "prcv_Thumos14_base_act_dis_sal_dyn_5" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 40 --enc_layers 2 --dec_layers 5 --actionness_loss --distillation_loss --distillation_loss_coef 0.1 --salient_loss --salient_loss_coef 2 --enable_element --element_rate 0.3 &
sleep 10
# --enable_element --element_rate 0.1
CUDA_VISIBLE_DEVICES=6 python main.py --model_name "prcv_Thumos14_base_act_dis_sal_dyn_6" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 40 --enc_layers 2 --dec_layers 5 --actionness_loss --distillation_loss --distillation_loss_coef 0.1 --salient_loss --salient_loss_coef 2 --enable_element --element_rate 0.1 &
sleep 10
# --enable_element --element_rate 0.05
CUDA_VISIBLE_DEVICES=7 python main.py --model_name "prcv_Thumos14_base_act_dis_sal_dyn_7" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 40 --enc_layers 2 --dec_layers 5 --actionness_loss --distillation_loss --distillation_loss_coef 0.1 --salient_loss --salient_loss_coef 2 --enable_element --element_rate 0.05 &
wait
echo "8个实验后台执行结束, Compute Node: 209 ....."

