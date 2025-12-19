python /home/liyong/code/svpretrain/mae/main_test_baidu.py \
--data_path /home/liyong/code/CityHomogeneity/data/baidu/V3/merged_test_sv.csv \
--output_dir /home/liyong/code/CityHomogeneity/data/baidu/V3/test_loss_feature_ep299_2.csv \
--model_path /home/liyong/code/svpretrain/output_dir/checkpoint/mae_vit_small_bsv1m_v3/checkpoint-299.pth \
--batch_size 256 \
--gpu_id 3

python /home/liyong/code/svpretrain/mae/main_test_baidu.py \
--data_path /data_ssd/BaiduSVs/metadata/merged_metadata.csv \
--output_dir /data_ssd/BaiduSVs/metadata_loss_feature_ep299.csv \
--model_path /home/liyong/code/svpretrain/output_dir/checkpoint/mae_vit_small_bsv1m_v3/checkpoint-299.pth \
--batch_size 256 \
--gpu_id 3

python /home/liyong/code/svpretrain/mae/main_test_baidu.py \
--data_path /home/liyong/code/CityHomogeneity/data/baidu/V3_history/beijing_history_with_year_month.csv \
--output_dir /home/liyong/code/CityHomogeneity/data/baidu/V3_history/beijing_history_mask2_all_loss_feature_ep299.csv \
--model_path /home/liyong/code/svpretrain/output_dir/checkpoint/mae_vit_small_bsv1m_v3/checkpoint-299.pth \
--batch_size 256 \
--gpu_id 3

