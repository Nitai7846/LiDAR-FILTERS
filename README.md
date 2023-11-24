# LiDAR-FILTERS
A collection of filtering Algorithms for LiDAR data 

This repository contains 3 filtering algorithms - 1] Ground Filtering, 2] Building Filtering and 3] Vegetation Filtering 


Basic Functions.py contains functions to read a PLY File, convert it into a dataframe and visualize it. 


Ground Filtering.py makes use of the CSF algorithm to remove ground points. 


Building Filtering.py detects rectangular patches and removes them from the file. 


Vegetation Filtering.py detects high density grids and removes them from the file. 


This code is compatible with (and based on) the DALES Open Source LiDAR Dataset. 

@inproceedings{varney2020dales,
  title={DALES: A Large-scale Aerial LiDAR Data Set for Semantic Segmentation},
  author={Varney, Nina and Asari, Vijayan K and Graehling, Quinn},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  pages={186--187},
  year={2020}
}


