
# Cloud-Classification
-------

## Part 1. Introduction

### 1.1.The Goal of the Project

Modeling clouds and understanding their relationship with climate systems is one of the most important steps to reduce uncertainties in current and future climate forecasts. For this purpose, it is extremely important to correctly classify cloud types with high spatial and temporal resolution. In this project, we will begin these processes by classifying the clouds.

### 1.2.About Dataset
In this project, we used the CUMULO comparative datasets. This dataset consists of one-year 1km resolution MODIS hyperspectral images combined with pixel-width 'traces' of CloudSat cloud tags. Bringing these complementary datasets together is a crucial first step that enables the Machine Learning community to develop innovative new techniques that can greatly benefit the Climate community. Our dataset has cloud layers. Cloud tier types are classified as follows:

* 0 - Cirrus

* 1 - Altostratus

* 2 - Altocumulus

* 3 - Stratus

* 4 - Stratocumulus

* 5 - Cumulus

* 6 - Nimbostratus

* 7 - Deep Convection

You may want to download the CUMULO dataset used here. For you can download the relevant dataset from [Here](https://github.com/FrontierDevelopmentLab/CUMULO)

The dataset we use is CUMULO. The subset of the CUMULO dataset consists of 150 files in NetCDF format according to the CF rule. If you want more detail about NetCDF and CF please click  [Here](http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.html).  The naming of these NetCDF files we downloaded is as follows:

**`file_name`** =  AYYYYDDD.HHMM.nc

**`YYYY`** = Year

**`DDD`**  = Day Since 01.01.2008 

**`HH`**   = Hour of Day

**`MM`**   = Minutes
  


### 1.3.About Algorithm
In this project, LightGBM algorithm, which is a gradient boosting method using tree-based learning algorithms, was used. The most important reason for using this algorithm is that it is computationally efficient and provides high-accuracy models. For more details about the algorithm and dataset, please visit **Part 2.**


# Part 2.Further Resources 
[For CUMULO Github Repository](https://github.com/FrontierDevelopmentLab/CUMULO)

[For CUMULO: A Dataset for Learning Cloud Classification](https://arxiv.org/abs/1911.04227)

[For LightGBM’s Documentation ](https://lightgbm.readthedocs.io/en/latest/index.html)


# Part 3.License
This work is licensed under a MIT License.



      **NOTE :** This project was created using the AI for Earth Monitoring course.
                        void.

  
