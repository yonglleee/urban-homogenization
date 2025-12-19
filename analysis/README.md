# Homogeneity Index Analysis Scripts


This directory contains scripts for analyzing the homogeneity index of urban regions using geospatial and statistical methods.

## File Descriptions

- `cls_analysis`: Clustering analysis of homogeneity or related features.
- `visual_baidu_v3.ipynb`: Data preprocessing and visualization for Baidu datasets.
- `yhy_new_visual.ipynb`: Visualization and analysis for figures and results used in the paper.
- `viz.py`: Utility tools for data inspection and quick viewing.


## Requirements
- Python 3.8+
- pandas, geopandas, numpy, openpyxl, shapely, torch, tqdm, scikit-learn

Install dependencies:
```bash
pip install pandas geopandas numpy openpyxl shapely torch tqdm scikit-learn
```

## Notes
- The scripts assume a specific directory structure and file naming convention for shapefiles and data files.
- For statistical tests (e.g., pairwise p-values), see the relevant notebook cells.
