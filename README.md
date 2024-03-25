# RAPID_A
### This code serve as the re-production tool for the article (DOI), aimed at adding generalizability to rapid post-event damage assessment through satellite imagery.

### The code offers step-by-step data generation, channel augmentation, ensemble learning, and test-time augmentation strategies outlined in the paper to evaluate their impacts on adding generalizability to rapid post-event damage assessment through satellite imagery.

* Cloning the repo in your system:
```bash 
cd /path/to/directory
git clone [repository_url](https://github.com/TRG-AI4Good/RAPID_A)
```
*  Then, define a directory (variable "cwd") to save outcomes, and pick a state (variable "State"). Regardless of your area of interest, metadata must be collected state-by-state.  
*  First, import the InventoryV1 class from Biv and create an instance of it. Doing this, two subfolders, "Raw data" and "Transitory data," are formed.  
Then, call Microsoft, NSI, and FEMA functions to store these inventories inside the Raw Data sub-directory, either as a single file (Sace_County=False) or both as a single file and County-wise files (Save_County=True, built as folders labeled by county FIPS codes).


