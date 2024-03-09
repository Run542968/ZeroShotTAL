echo "8个实验后台执行开始, Compute Node: 209 ....."
# base
CUDA_VISIBLE_DEVICES=3 python main.py --model_name "prcv_Thumos14_base" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 40 --enc_layers 2 --dec_layers 5 &
sleep 5
# base + actionness
CUDA_VISIBLE_DEVICES=4 python main.py --model_name "prcv_Thumos14_base_act" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 40 --enc_layers 2 --dec_layers 5 --actionness_loss &
sleep 5
# base + actionness + distillation
CUDA_VISIBLE_DEVICES=5 python main.py --model_name "prcv_Thumos14_base_act_dis" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 40 --enc_layers 2 --dec_layers 5 --actionness_loss --distillation_loss --distillation_loss_coef 0.1 &
sleep 5
# base + actionness + distillation + salient
CUDA_VISIBLE_DEVICES=6 python main.py --model_name "prcv_Thumos14_base_act_dis_sal" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 40 --enc_layers 2 --dec_layers 5 --actionness_loss --distillation_loss --distillation_loss_coef 0.1 --salient_loss --salient_loss_coef 2 &
sleep 5
# base
CUDA_VISIBLE_DEVICES=7 python main.py --model_name "prcv_deform_Thumos14_base" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --save_result --detr_architecture "DeformableDetr" --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 40 --enc_layers 2 --dec_layers 5 &
sleep 5
# base + actionness
CUDA_VISIBLE_DEVICES=1 python main.py --model_name "prcv_deform_Thumos14_base_act" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --save_result --detr_architecture "DeformableDetr" --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 40 --enc_layers 2 --dec_layers 5 --actionness_loss &
sleep 5
# base + actionness + distillation 
CUDA_VISIBLE_DEVICES=2 python main.py --model_name "prcv_deform_Thumos14_base_act_dis" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --save_result --detr_architecture "DeformableDetr" --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 40 --enc_layers 2 --dec_layers 5 --actionness_loss --distillation_loss --distillation_loss_coef 0.1 &
sleep 5
# base + actionness + distillation + salient
CUDA_VISIBLE_DEVICES=0 python main.py --model_name "prcv_deform_Thumos14_base_act_dis_sal" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --save_result --detr_architecture "DeformableDetr" --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 40 --enc_layers 2 --dec_layers 5 --actionness_loss --distillation_loss --distillation_loss_coef 0.1 --salient_loss --salient_loss_coef 2 &
wait
echo "8个实验后台执行结束, Compute Node: 209 ....."

