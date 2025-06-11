cd /data1/sunpengyu/Task_VirtualStain/Code/GAN

python test_visEnc.py \
--config_path i2i_config.yml \
--i2i_patches_path /data6/sunpengyu/Task_VirtualStain/Data/szph_preprocessed_v1_reg_patch_2048-8 \
--src_data_split_path /data6/sunpengyu/Task_VirtualStain/Data/szph_preprocessed_v1_reg_patch_2048/split_HE2PR.csv \
--save_path /data1/sunpengyu/Task_VirtualStain/Result/26May2025/VisEnc-2048_8 \
--src_marker HE \
--dst_marker PR \
--is_train False \
--is_test True \
--temperature 0.01 \
--timm_model swinv2_base_window16_256.ms_in1k \
--ckpt_path /data1/sunpengyu/Task_VirtualStain/Result/26May2025_2048-8/VisEnc_swinv2b_temp-0.01/PR_VisEnc_epoch_75.pt
# --timm_model swinv2_tiny_window16_256.ms_in1k \
# --ckpt_path /data1/sunpengyu/Task_VirtualStain/Result/26May2025_2048-8/VisEnc_swinv2t_temp-0.1/PR_VisEnc_epoch_175.pt
