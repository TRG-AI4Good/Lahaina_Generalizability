# RAPID_A
### This code serve as the re-production tool for the article 'Rapid Computer-Vision-based Post-Event Assessment Tools for Natural Disasters: Enhancing Generalizability' (DOI), aimed at adding generalizability to rapid post-event damage assessment through satellite imagery.

### The code offers step-by-step data generation, channel augmentation, ensemble learning, and test-time augmentation strategies outlined in the paper to evaluate their impacts on adding generalizability to rapid post-event damage assessment through satellite imagery.

* Cloning the repo in your system:
```bash 
cd /path/to/directory
git clone https://github.com/TRG-AI4Good/RAPID_A.git
```
* The Run.ipynb notebook will guide you through the whole repository, building data and making inference step-by-step
* Still, making sure the requirements.txt libraries are installed is needed.
* Since for the "general-purpose" CD model we are using building on codes by El. Amin et. al. (https://github.com/vbhavank/Unstructured-change-detection-using-CNN), the Tensorflow librray must be at version 2.13.0; as noted in the requirements file (un-installation of a prior TF version might be needed.

* In case you want to give it a try at Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]([https://colab.research.google.com/github/karpathy/llama2.c/blob/master/run.ipynb)



