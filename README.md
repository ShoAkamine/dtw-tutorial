# Tutorial for using dynamic time warping (DTW) to quantify the degree of gesture form similarity
This tutorial aims to help students and researchers working on gestures and sign languages learn how to use DTW to quantify the degree of gesture (sign) form (dis)similarity. The following topics are covered:

1. Understanding the characteristics of DTW (`dtw_example.ipynb`)
2. Preprocessing the MediaPipe motion tracking time series data
3. Trimming the time series data for the duration of gestures
4. Learning how to improve the predictive accuracy of DTW

## Requirements
Before running the script, you need to have the following installed:
- [Anaconda](https://www.anaconda.com/download/success)
- [Visual Studio Code](https://code.visualstudio.com/download) (or an IDE of your preference).


## Preparation
To run the Python notebooks, please complete the steps below:

1. Download the repository. You can do so in two different ways:
    - Option 1: clone the repository
    - Option 2: click the green "<> Code" button on this page and select "Download ZIP". Make sure to unzip the folder before moving on to the next step. 

2. Set up the conda environment by following the instructions under the "Install packages" section.
    1. Open terminal/anaconda prompt at the "scripts" folder
        - Mac:
            1. go to the "dtw_tutorial" folder
            1. right-click the "scripts" folder
            1. select "open terminal at this folder" (under "Services") <br><br>
        - Windows:
            1. go to the "scripts" folder
            1. copy the path to the folder
            1. open Anaconda Prompt
            1. type cd and paste the path after a space (e.g., cd D:/users/shoakamine/dtw_tutorial/scripts)
            1. if the folder is not in the C drive (e.g., D:/users/...), type the drive initial followed by a colon (e.g., D:):
  
    1. Create Python 3.10 environment on Anaconda prompt/terminal
        ```
        conda create --name dtw python=3.10
        ```
    1. Activate your conda environment
        ```
        conda activate dtw
        ```
    
    1. Install git if not already installed: `conda install -c anaconda git`
    
    1. Run this command to install required packages: `pip install -r requirements.txt`
    
    1. Reinstall the ffmpeg package: `conda install -c conda-forge ffmpeg`

3. Open the `dtw.ipynb` or `dtw_example.ipynb` file (I strongly encourage everyone to go through `dtw_example.ipynb` first to understand the properties of DTW)

4. On VS Code, click "select kernel" on the top right and select "Python Environments..." --> "dtw"

5. Download MediaPipe time series data from [this OSF repository](https://osf.io/4bqys/files).

6. Place the time series data under `data\mediapipe\` in the corresponding folders

The preparation needs to be done only once. Afterwards, you can run the scripts immediately.

