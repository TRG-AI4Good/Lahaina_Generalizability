# RAPID_A

### What is RAPID-A? RAPID-A is a deep learning-based framework for rapid post-event damage assessment from satellite imagery, with the main incentive being effective transferability across urban textures and hazards, as real-world scenarios demand.
* We took the opportunity of the recent 2023 Kahramanmaras earthquake, as a rich damage dataset, and the 2023 Maui wildfire incident to build RAPID-A.
* This repo, on its own, is a complete pipeline for such damage assessments. However, this repo mainly serves to reproduce the research article "Rapid Computer-Vision-based Post-Event Assessment Tools for Natural Disasters: Enhancing Generalizability" (DOI), aimed at adding generalizability to rapid post-event damage assessment through satellite imagery.
* Breaking down RAPID-A compartments, this repo step-by-step goes through how our design and taken measures aid the generalizability concerns, with the Lahaina-Maras case as the case study.


 What is RAPID-A? RAPID-A is deep mearning-based framework for rapid post-event damage assessment from satellite imagery with the main incentive being effective transferrability across urban textures and hazards, as real-world scenarios demand that.

## We took the opporutniy of the recent 2023 Kahramanmaras earthquake, as a rich damage dataset, and the 2023 Maui wildfire incident to build RAPID-A.

## This repo on its own, is a complete piopeline for such damage assessments, yet, this repo mainly serves reproducing reaseacrh article 'Rapid Computer-Vision-based Post-Event Assessment Tools for Natural Disasters: Enhancing Generalizability' (DOI), aimed at adding generalizability to rapid post-event damage assessment through satellite imagery.

###Breaking down RAPID-A compartments, this repo step-by-step goes through how our design and taken measures aid the generalizabilty concernss, with the Lahaina-Maras case as the cvase study.

* Cloning the repo in your system:
```bash 
cd /path/to/directory
git clone https://github.com/TRG-AI4Good/RAPID_A.git
```
* The Run.ipynb notebook will guide you through the whole repository, building data and making inference step-by-step
* Still, making sure the requirements.txt libraries are installed is needed.
* Since for the "general-purpose" CD model we are using building on codes by El. Amin et. al. (https://github.com/vbhavank/Unstructured-change-detection-using-CNN), the Tensorflow librray must be at version 2.13.0; as noted in the requirements file (un-installation of a prior TF version might be needed.

* In case you want to give it a try at Google Colab

Open ROC_Curve_Results.ipynb In Colab: 
<a target="_blank" href="https://colab.research.google.com/github/TRG-AI4Good/RAPID_A/blob/main/Run.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# What is RAPID-A?

## Accordingly, RAPID-A is build upon 1- Channel augemntattion, 2- Deep ensemble learning, and 3- Test-Time augmenttaion strategies to accomplish that.
## The frameowrk has definite inputs, wnd given those, it can be used elsewhere.
##Inputs are: 
* a set of six deep segmenttaion models with various channel augmenttaions; herein, those are provided with models trained on Turkiye earthquake sequence of 2023
* The Building inventory; herein, we use the mixture of FEMA and NSI, but it is not built within this repo and only the completed inventory is inputted
*  The region divided into 640-by-640 pixel images, wtih geometries stored in a parquet file (you can find an example when downlloading the data)
*  The NASA ARIA maps of the incident, always follows shortly after major events
*  If labels are proviusded as wel, it will return the accutrcay too

# RAPID-A hgas three cartegories, fully-partially collapsed buildings (C), buildings with observable aerial damages (D) and no-damage biuildings are the ones which are not identified as C or D
