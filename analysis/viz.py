import streamlit as st
import pandas as pd
from PIL import Image
import os

# 读取数据
csv_path = "/home/liyong/code/CityHomogeneity/output/classification/qwen_cls_merged_homo.csv"
df = pd.read_csv(csv_path)

st.title("City Homogeneity Image Explorer")
st.set_page_config(layout="wide")  # 添加这一行
# 下拉框筛选
# city = st.selectbox("Select City", sorted(df['city'].unique()))
# category = st.selectbox("Select Category", sorted(df[df['city'] == city]['category'].unique()))

# # 根据筛选结果过滤数据
# filtered_df = df[(df['city'] == city) & (df['category'] == category)].reset_index(drop=True)

# st.write(f"Found {len(filtered_df)} images for city: {city}, category: {category}")

# st.subheader("Category Count")
# city_df = df[df['city'] == city]
# category_counts = city_df['category'].value_counts().sort_index()
# st.bar_chart(category_counts)

# 可选择的标签列（文本列）
label_cols = [c for c in df.columns if df[c].dtype == 'object']
default_idx = label_cols.index('category') if 'category' in label_cols else 0
label_col = st.selectbox("Label column", options=label_cols, index=default_idx)

if label_col not in df.columns:
    st.error(f"Column '{label_col}' not found in DataFrame.")
    st.stop()

# 下拉框筛选
city = st.selectbox("Select City", sorted(df['city'].dropna().unique()))
label_value = st.selectbox(
    "Select Label",
    sorted(df[df['city'] == city][label_col].dropna().unique())
)

# 根据筛选结果过滤数据
filtered_df = df[(df['city'] == city) & (df[label_col] == label_value)].reset_index(drop=True)

st.write(f"Found {len(filtered_df)} images for city: {city}, {label_col}: {label_value}")

st.subheader(f"{label_col} Count")
city_df = df[df['city'] == city]
category_counts = city_df[label_col].value_counts().sort_index()
st.bar_chart(category_counts)

# # 图片切换控件
# st.subheader("Images")
# if len(filtered_df) > 0:
#     img_idx = st.number_input("Image Index", min_value=0, max_value=len(filtered_df)-1, value=0, step=1)
#     row = filtered_df.iloc[img_idx]
#     st.write(f"Image Name: {row['image_name']}")
#     st.write(f"Homogeneity: {row['homogeneity']}")
#     img_path = f"/data_nas/liyong/{row['path']}"
#     if os.path.exists(img_path):
#         img = Image.open(img_path)
#         st.image(img, caption=row['image_name'], use_container_width=True)
#     else:
#         st.warning(f"Image not found: {img_path}")
# else:
#     st.info("No images found for this selection.")

st.subheader("Images")
images_per_page = 9  # 每页显示9张
images_per_row = 3   # 每行3张



if len(filtered_df) > 0:
    sorted_df = filtered_df.sort_values(by="homogeneity", ascending=False).reset_index(drop=True)
    total_pages = (len(sorted_df)-1)//images_per_page + 1
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    st.write(f"Current page: {page} / {total_pages}")
    start_idx = (page-1)*images_per_page
    end_idx = min(start_idx + images_per_page, len(sorted_df))
    page_df = sorted_df.iloc[start_idx:end_idx].reset_index(drop=True)
    n_rows = (len(page_df) + images_per_row - 1) // images_per_row
    for row_idx in range(n_rows):
        cols = st.columns(images_per_row)
        for col_idx in range(images_per_row):
            img_idx = row_idx * images_per_row + col_idx
            if img_idx >= len(page_df):
                break
            row = page_df.iloc[img_idx]
            img_path = f"/data_ssd/{row['path']}"
            with cols[col_idx]:
                st.write(f"Homogeneity: {row['homogeneity']:.4f}")
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    st.image(img, caption=row['image_name'], width='stretch')
                else:
                    st.warning(f"Image not found: {img_path}")
else:
    st.info("No images found for this selection.")