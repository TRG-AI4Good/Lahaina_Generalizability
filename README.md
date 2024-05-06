# RAPID_A

### What is RAPID-A? RAPID-A is a deep learning-based framework for rapid post-event damage assessment from satellite imagery, with the main incentive being effective transferability across urban textures and hazards, as real-world scenarios demand.
* We took the opportunity to use the recent 2023 Kahramanmaras earthquake as a rich damage dataset and the 2023 Maui wildfire incident to build RAPID-A.
* This repo, on its own, is a complete pipeline for such damage assessments. However, this repo mainly serves to reproduce the research article "Rapid Computer-Vision-based Post-Event Assessment Tools for Natural Disasters: Enhancing Generalizability" (DOI), aimed at adding generalizability to rapid post-event damage assessment through satellite imagery.
* To tun RAPID-A, you need to have 5 data: 1- Blocks that is the geometry parquet file of your case study divided after taking 640-by-640 pixel images from it. 2- The inventory: you can mix NSI and FEMA inventories as outlined in the paper to build such inventory for anywhere inside the U.S. 3- Pre- and Post-event images for those blocks 4- (optional) damage labels for each building inside the inventory, and 5- NASA ARIA map for the incident of your choice as a tiff file, the code itself will figure that out across blocks.
* RAPID-A outputs two damage labels, C (fully or partially collapsed) and D (aerially observable non-collapse damages). Any building of inventory not categorized as those is deemed as no damage.
* Breaking down RAPID-A compartments, this repo step-by-step goes through how our design and taken measures aid the generalizability concerns, with the Lahaina-Maras case as the case study.


## How to use the repo:
* 1- Cloning the repo in your system:
```bash 
cd /path/to/directory
git clone https://github.com/TRG-AI4Good/Lahaina_Generalizability
```
* 2- Go to the cloned repo directory, and execute Run.ipynb notebook that will guide you through the whole repository, building data and making inferences step-by-step.
* Still, it is necessary to make sure the requirements.txt libraries are installed.
* Since for the "general-purpose" CD model, we are building on codes by El. Amin et. al. (https://github.com/vbhavank/Unstructured-change-detection-using-CNN), the Tensorflow library must be at version 2.13.0 (you may change it to 2.15.0 in case of persistent package confliction) and Python v 3.10 to support that; as noted in the requirements file (un-installation of a prior TF version might be needed.


