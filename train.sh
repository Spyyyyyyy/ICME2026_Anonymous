cd /data1/sunpengyu/Task_VirtualStain/Code/GAN_virtualstain

# train
python train.py \
--mode train \
--gan_type cond_cpt \
--config_path /data1/sunpengyu/Task_VirtualStain/Code/GAN_virtualstain/config/cond_cpt_config.yml \
--data_path /data6/sunpengyu/Task_VirtualStain/Data/szph_preprocessed_v1_reg_patch_2048-8 \
--split_csv_path /data6/sunpengyu/Task_VirtualStain/Data/szph_preprocessed_v1_reg_patch_2048/split_HE2PR.csv \
--save_path /data1/sunpengyu/Task_VirtualStain/Result/26May2025-2048_8/cond_cpt \
--src_marker HE \
--dst_marker PR \
--cuda True \
--gpu_ids 0 1 2 3 4 5 6 7 \
