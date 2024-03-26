import os
import numpy as np
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
import torchvision
get_ipython().system('pip install git+https://github.com/qubvel/segmentation_models.pytorch')
import segmentation_models_pytorch as smp
import albumentations as A  
from albumentations.pytorch import ToTensorV2
import tqdm
try:
    from tqdm.notebook import tqdm
except:
    from tqdm import tqdm
import matplotlib.pyplot as plt
import geopandas as gpd
import fiona
import shapely

import rasterio
from rasterio.plot import reshape_as_image
import tensorflow as tf
import keras
from keras.applications.vgg19 import VGG19
from keras.utils import load_img, img_to_array
from keras.applications.vgg19 import preprocess_input
import sys
from skimage import filters #change to 'import filter' for Python>v2.7
from skimage import exposure
from keras import backend as K
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely import wkt
import pandas as pd
from shapely.validation import make_valid
from shapely.ops import unary_union
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from pyproj import Geod
from scipy.stats import lognorm
import urllib
import patoolib
import shutil
import albumentations as A 
import random
import tabulate
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

class RAPID_A():
    """
    This initialization first download and extract required files to start the analysis
    These files include: Block geometries, Google (post-event) and Bing (pre-event) images
    which were taken from Lahaina Maui fire incident herein.
    
    Parameters:
        width (int): Width of images in pixels
        height (int): Height of images in pixels
        
    Returns:
        Build ./Data folder with the aformentioned files within
    """
    def __init__(self,cwd,width: int,height: int):
        self.cwd=cwd
        self.W=width
        self.H=height
        path='./Data'
        if os.path.exists(path):
            pass
        else:
    # Download required files
            url = 'https://www.dropbox.com/scl/fi/e1vx4wpqo3inz3uid3fz8/RAPID_A.rar?rlkey=ggh17ohxap7vbab8soywdbzmi&id=1&dl=1'
            destination = 'Files.rar'
            import urllib
            #Some functions to handle downloads
            def download_file(url, destination):
                urllib.request.urlretrieve(url, destination)

            def is_download_complete(url, destination):
                # Get the size of the file from the Content-Length header
                with urllib.request.urlopen(url) as response:
                    expected_size = int(response.headers['Content-Length'])

                # Get the actual size of the downloaded file
                actual_size = os.path.getsize(destination)

                # Compare the expected size with the actual size
                return expected_size == actual_size

            download_file(url, destination)

            if is_download_complete(url, destination):
                print("Download complete!")
            else:
                print("Download Failed; Please retry")

            #Building the "Data" folder within the directory to save data
            rar_file = 'Files.rar'
            destination_folder = './'
            patoolib.extract_archive(rar_file, outdir=destination_folder)
            shutil.move('./RAPID_A','./Data')
            os.remove('./Files.rar')


    def get_concat_h(self,im1, im2):
        """
        This function glues two given Pillow images horizontally.
        
        Parameters:
            im1 (Pillow image): Width of images in pixels
            im2 (Pillow image): Height of images in pixels
            
        Returns:
            concat_im: Horizontally concatenated Pillow image
        """
        concat_im = Image.new('RGB', (im1.width + im2.width, im1.height))
        concat_im.paste(im1, (0, 0))
        concat_im.paste(im2, (im1.width, 0))
        return concat_im






# Extract ARIA maps from the main tiff file
# save it solo and also concatenated to the
# Googlw (post-event) image
    def ARIA_maps(self):
        """
        This function extrcats ARIA maps from the main ARIA map, herein "ARIA.tiff" taken from 
        NASA, and then glue them horizontally to the Google (post-event) images and save it within
        the directory folder ./Data/G_Plus_ARIA.
            
        Parameters:
            self
        """
        cwd=self.cwd
        cwd_data=cwd+'/Data/'

        # Load the ARIA tiff file and build a grayscale image from it
        src=rasterio.open(cwd_data+'/ARIA.tif')
        img_array = src.read()
        img_array = reshape_as_image(img_array)
        img = Image.fromarray(img_array)
        img = img.convert('L')

        # Save the image as PNG
        img.save(cwd_data+'ARIAB.png')
        im=Image.open(cwd_data+'ARIAB.png')


        # List the Block boundary parquet files
        # and build some needed directories
        F=os.listdir(cwd_data+'/CSVS/')
        io=0
        try:
            os.mkdir(cwd_data+'/ARIA')
        except:
            pass
        try:
            os.mkdir(cwd+'/Data/G_Plus_ARIA/')
        except:
            pass

        # Extracting ARIA image for each block
        for block in tqdm(range(len(F)),desc='Buiding block-wise ARIA maps'):

            path_to_file = cwd_data+'/CSVS/'+'Blocks_geom'+str(block+1)+'.parquet'
            K = gpd.read_parquet(path_to_file)

        # Get the tiff file boundaris 
            K=K.set_crs(4326)
            Z2=src.bounds
            Z3=[0,0,0,0]
            Z3[0]=Z2[0]
            Z3[1]=Z2[1]
            Z3[2]=Z2[2]
            Z3[3]=Z2[3]
            poly1 =shapely.geometry.box(Z3[0], Z3[1], Z3[2], Z3[3], ccw=True)
            [A,B]=im.size
            stepx=(Z3[2]-Z3[0])/A
            stepy=(Z3[3]-Z3[1])/B

        # find each block pixel boundaries
            GG=K.copy()
            VV=GG['geometry'].iloc[0]
            Z1=(VV).bounds
            PXS=max(0,np.floor(((Z1[0]-Z3[0])/(Z3[2]-Z3[0]))*A))
            PXS_S=((((Z1[0]-Z3[0])/(Z3[2]-Z3[0]))*A))%1

            PXF=min(np.ceil(((Z1[2]-Z3[0])/(Z3[2]-Z3[0]))*A),A)
            PXF_S=1-((((Z1[2]-Z3[0])/(Z3[2]-Z3[0]))*A))%1

            PYS=max(0,np.floor(((Z1[1]-Z3[1])/(Z3[3]-Z3[1]))*B))
            PYS_S=((((Z1[1]-Z3[1])/(Z3[3]-Z3[1]))*B))%1

            PYF=min(np.ceil(((Z1[3]-Z3[1])/(Z3[3]-Z3[1]))*B),B)
            PYF_S=1-(((Z1[3]-Z3[1])/(Z3[3]-Z3[1]))*B)%1
            #print('gg',(((Z1[0]-Z3[0])/(Z3[2]-Z3[0]))))
            AA=PXF-PXS
            BB=PYF-PYS


            im1 = im.crop((PXS, B-PYF, PXF, B-PYS))


            ##########
            width, height=im1.size
            im1=im1.resize((width*500,height*500),resample=0)

        # Since pixel resolution of ARIA images and blocks do not share a common multiplier
        # it occurs occur that some pictures turn into 3500 and some to 2500 pixel wide
        # this code is to account for that issue.
            if im1.size[1]==2500:
                im2 = im1.crop((PXS_S*500, (PYS_S)*500, min(3500,(width)*500+PXF_S*500), min(2500,(height)*500+PYF_S*500)))
            else:
                im2 = im1.crop((PXS_S*500, (1-PYS_S)*500, min(3500,(width)*500+PXF_S*500), min(3000,(height-1)*500+(PYF_S)*500)))

            im2=im2.resize((self.W,self.H))
            im2.save(cwd_data+'/ARIA/'+str(io+1)+'.png')

            # Concat horizontally with Google (post-event) images
            im_g=Image.open(cwd_data+'/M_G_Concat/'+str(io+1)+'.png')
            [width,height]=[im_g.width,im_g.height]
            im_g = im_g.crop((int(width*0.5),height*0, width*1, 1*height))
            self.get_concat_h(im_g,im2).save(cwd_data+'/G_Plus_ARIA/'+str(io+1)+'.png')
            io+=1

    '''
     Several function to get CD maps according to El Amin et al. https://github.com/vbhavank/Unstructured-change-detection-using-CNN.
    '''
    def get_activations(self,model, layer_idx, X_batch):
        get_activations = K.function([model.layers[0].input], [model.layers[layer_idx].output,])
        activations = get_activations([X_batch])[0]
        return activations

    #Function to extract features from intermediate layers
    def extra_feat1(self,img_path):

            #Using a VGG19 as feature extractor
            base_model = VGG19(weights='imagenet',include_top=False)
            img = load_img(img_path, target_size=(224, 224*2))
            img = img.crop((0,0, 224, 224))
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            block1_pool_features=self.get_activations(base_model, 3, x)
            block2_pool_features=self.get_activations(base_model, 6, x)
            block3_pool_features=self.get_activations(base_model, 10, x)
            block4_pool_features=self.get_activations(base_model, 14, x)
            block5_pool_features=self.get_activations(base_model, 18, x)

            x1 = tf.image.resize(block1_pool_features[0],[112,112])
            x2 = tf.image.resize(block2_pool_features[0],[112,112])
            x3 = tf.image.resize(block3_pool_features[0],[112,112])
            x4 = tf.image.resize(block4_pool_features[0],[112,112])
            x5 = tf.image.resize(block5_pool_features[0],[112,112])

            F = tf.concat([x1,x2,x3,x4,x5], axis=2) #Change to only x1, x1+x2,x1+x2+x3..so on, inorder to visualize features from diffetrrnt blocks
            return F

    def extra_feat2(self,img_path):

            #Using a VGG19 as feature extractor
            base_model = VGG19(weights='imagenet',include_top=False)
            img = load_img(img_path, target_size=(224, 224*2))
            img = img.crop((224,0, 448, 224))
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            block1_pool_features=self.get_activations(base_model, 3, x)
            block2_pool_features=self.get_activations(base_model, 6, x)
            block3_pool_features=self.get_activations(base_model, 10, x)
            block4_pool_features=self.get_activations(base_model, 14, x)
            block5_pool_features=self.get_activations(base_model, 18, x)

            x1 = tf.image.resize(block1_pool_features[0],[112,112])
            x2 = tf.image.resize(block2_pool_features[0],[112,112])
            x3 = tf.image.resize(block3_pool_features[0],[112,112])
            x4 = tf.image.resize(block4_pool_features[0],[112,112])
            x5 = tf.image.resize(block5_pool_features[0],[112,112])

            F = tf.concat([x1,x2,x3,x4,x5], axis=2) #Change to only x1, x1+x2,x1+x2+x3..so on, inorder to visualize features from diffetrrnt blocks
            return F


    def CD_maps(self):
            """
            This function extrcats CD maps from Google (post-event) and Bing (pre-event) block satellite iomages
            and then glue them horizontally to the Google (post-event) images and save it within the directory 
            folder ./Data/G_Plus_CD.
            
            Parameters:
                self
                
            """
            cwd=self.cwd
            cwd_data=cwd+'/Data/'
            X=cwd+'/Data/M_G_Concat/'
            try:
                os.mkdir(cwd+'/Data/CDS/')
            except:
                pass
            try:
                os.mkdir(cwd+'/Data/G_Plus_CD/')
            except:
                pass
            files = os.listdir(X)
            
            zet=0
            for file in tqdm(files,desc='Buiding block-wise CD maps'):
                if file[-3:] == "png":
                    basename = file[:-4]
                    F1=self.extra_feat1(X+file) #Features from image patch 1
                    F1=tf.square(F1)
                    F2=self.extra_feat2(X+file) #Features from image patch 2
                    F2=tf.square(F2)
                    d=tf.subtract(F1,F2)
                    d=tf.square(d)
                    d=tf.reduce_sum(d,axis=2)

                    dis=(d.numpy())   #The change map formed showing change at each pixels
                    # dis=np.resize(dis,[640,640])
                    min = np.min(dis)
                    max = np.max(dis)
                    image = (dis - min) / (max - min) * 255
                    im = Image.fromarray(image)
                    im = im.resize((self.W,self.H))
                    im = im.convert("L")
                    im.save(f"{cwd+'/Data/CDS/'}{basename}.png")

                    # concat with Google
                    im_g=Image.open(cwd_data+'/M_G_Concat/'+basename+'.png')
                    [width,height]=[im_g.width,im_g.height]
                    im_g = im_g.crop((int(width*0.5),height*0, width*1, 1*height))
                    self.get_concat_h(im_g,im).save(cwd_data+'/G_Plus_CD/'+basename+'.png')
        

  


# Data loader: for simplicity we always input image bundles to the method
#e.g., Google + CD.
    class SegmentationDataset(Dataset):
        """
        This Data loader Class is to call data during training, etc. while it axccepts 
        the "degree" attribute too to account for the rotational TTA startgey outlined in the 
        paper.
        
        Parameters:
            input_dir (str): Directory to image files.
            W1 (int): Image height in pixels.
            H1 (int): Image width in pixels.

        Returns:

            """
        import random
        def __init__(self, input_dir, is_train,W1,H1,degree=0):
            import albumentations as A 
            self.W=W1
            self.H=H1
            self.input_dir  = input_dir
            self.rotate=degree
            if is_train == True:
                x = round(len(os.listdir(input_dir)) * 1)
                self.images = os.listdir(input_dir)[:x]
            else:
                x = round(len(os.listdir(input_dir)) * 1)
                self.images = os.listdir(input_dir)[x:]
                
                
                                  
            self.transform = A.Compose(
                [
                    A.Resize(height=self.H, width=self.W),
                    ToTensorV2(),
                ],
            )    

        

        def __len__(self):
            return len(self.images)

        def __getitem__(self, index):
            img_path    = os.path.join(self.input_dir, self.images[index])
            img1         = Image.open(img_path).convert("RGB")
            import random
            r=random.uniform(0, 1)
            r1=random.uniform(0, 1)
            img=img1.crop((0, 0, 640, 640))
            cd=img1.crop((640, 0, 1280, 640))
            if self.rotate>0:
                img=img.rotate(self.rotate)
                cd=cd.rotate(self.rotate)
            ####
            img=np.array(img, dtype=np.float32)/255
            cd =np.array((cd).convert('L'), dtype=np.float32)/255

            if self.transform is not None:
                augmentations = self.transform(image=img)
                img   = augmentations["image"]
            return img, cd
        

        


# This function returns masks for a given model or a list of model names.
    def Model_Mask(self,Models: list,damage_classes,rotate=False):
        """
        This Data loader Class is to call data during training, etc. while it axccepts 
        the "degree" attribute too to account for the rotational TTA startgey outlined in the 
        paper.
        
        Parameters:
            Models (list): A list of models.
            damage_classes (list): List of damage classes; herein, ['C','D']
            rotate (bool): whether we have rotation or not (TTA)
            
        Returns:

            """
        cwd=self.cwd
        self.damage_classes=damage_classes
        for Model in Models:
            
            try:
                os.mkdir(os.getcwd()+'/Results')
            except:
                pass
            try:
                os.mkdir(os.getcwd()+'/Results/'+Model)
            except:
                pass
            cwd_re=cwd+'/Results/'
            
            Model=Model
            Source=cwd

            # model names are coded, so it understands the model,
            # augmented channels, etc. from its name
            CD=[]
            if 'CD' in Model or 'ARIA' in Model:
                CD.append(True)
            else:
                CD.append(False)
            if 'CD' in Model:
                CD.append('CD')
            elif 'ARIA' in Model:
                CD.append('ARIA')
            if 'Eff' in Model:
                CD.append('Eff')
            elif 'Tra' in Model:
                CD.append('Tra')
            if rotate==True:
                CD.append('R')
            else:
                CD.append('')        
            try:
                os.mkdir(cwd_re+Model)
                os.mkdir(cwd_re+Model+'/Shape_Files')
            except:
                pass

            device = "cuda" if torch.cuda.is_available() else "cpu"

            # if TTA is asked 
            if rotate:
                Rotaions=[0,90,180,270]
                for rotation in Rotaions:
                    for damage_class in damage_classes:
                        try:
                            os.mkdir(cwd_re+Model+'/Mask_Results_'+damage_class+'_'+str(rotation)+'/')
                        except:
                            pass
                    batch_size=1
                    for damage_class in damage_classes:
                        TRAIN_INP_DIR = Source+'/Data'+'/G_Plus_CD'
                        train_ds=self.SegmentationDataset(TRAIN_INP_DIR,True,self.W,self.H,rotation)
                        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False )
                        try:
                            model=torch.load(Source+'/Data/Deep_Models/'+Model+damage_class+'.pth')['Model']
                        except:
                            model=torch.load(Source+'/Data/Deep_Models/'+Model+damage_class+'.pth')
                        model.eval()

                        DIR = cwd_re+Model+'/Mask_Results_'+damage_class+'_'+str(rotation)+'/'
                        zet=0
                        
                        F=os.listdir(TRAIN_INP_DIR)

                        pep=0
                        for image, cd in tqdm(train_loader,desc=Model+"'s Mask Generatrion for:\t"+damage_class+', rotation= '+str(rotation)):
                            #############
                            #
                            if (CD[0] and CD[2]=='Tra'):
                                img   = image[:,0:2,:,:].to(device=device)
                            else:
                                img   = image[:,:,:,:].to(device=device)
                            imageCD1   = cd.to(device=device).reshape(batch_size,1,640,640)
                            if CD[0]:
                                img=torch.cat((img,imageCD1),axis=1)
                            output= ((torch.sigmoid(model(img.to('cuda')))) >0.5).float()

                            [A,B,C,D]=img.shape
                            for i in range(A):
                                dada=(output[i][0].detach().detach().cpu().numpy()).astype('float32')
                                dada[dada==1]=255
                                img = Image.fromarray(dada, 'F').convert('L')
                                img.save(DIR+F[zet])
                                zet+=1
                            pep+=1
                            del output
                            torch.cuda.empty_cache()
            # If TTA is not asked.
            else:
                for damage_class in damage_classes:
                    try:
                        os.mkdir(cwd_re+Model+'/Mask_Results_'+damage_class+'/')
                    except:
                        pass
                batch_size=1
                for damage_class in damage_classes:
                    TRAIN_INP_DIR = Source+'/Data'+'/G_Plus_CD'
                    train_ds=self.SegmentationDataset(TRAIN_INP_DIR,True,self.W,self.H)
                    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False )
                    try:
                        model=torch.load(Source+'/Data/Deep_Models/'+Model+damage_class+'.pth')['Model']
                    except:
                        model=torch.load(Source+'/Data/Deep_Models/'+Model+damage_class+'.pth')
                    model.eval()

                    DIR = cwd_re+Model+'/Mask_Results_'+damage_class+'/'
                    zet=0
                    
                    F=os.listdir(TRAIN_INP_DIR)

                    pep=0
                    for image, cd in tqdm(train_loader,desc=Model+"'s Mask Generatrion for:\t"+damage_class):
                        #############
                        #
                        if (CD[0] and CD[2]=='Tra'):
                            img   = image[:,0:2,:,:].to(device=device)
                        else:
                            img   = image[:,:,:,:].to(device=device)
                        imageCD1   = cd.to(device=device).reshape(batch_size,1,640,640)
                        if CD[0]:
                            img=torch.cat((img,imageCD1),axis=1)
                        output= ((torch.sigmoid(model(img.to('cuda')))) >0.5).float()

                        [A,B,C,D]=img.shape
                        for i in range(A):
                            dada=(output[i][0].detach().detach().cpu().numpy()).astype('float32')
                            dada[dada==1]=255
                            img = Image.fromarray(dada, 'F').convert('L')
                            img.save(DIR+F[zet])
                            zet+=1
                        pep+=1
                        del output
                        torch.cuda.empty_cache()

    
    def Mask_To_Poly(self,Models,Damage_Cases):
        """
        Given each block has known geometries, this function returns geo-polygon of 
        segmentation patches. This function is rather comples, but to make it short
        it first transforms segmentation pictures into boundaries of shapes within it 
        then it maps those to the known geometruies of that block.

        Parameters:
            Models (list): A list of models.
            damage_classes (list): List of damage classes; herein, ['C','D']
            
        Returns:
            The union of all segmentation patches as a single geometry file inside the folder 
            ./Results/Model/Shape_files/(C or D).parquet

            """
        for Model in Models:
            for Damage_Case in Damage_Cases:
                self.Mask_To_Polygon(Model,Damage_Case);
        
    def Mask_To_Polygon(self,Model,Damage_Cases):
        """
        The logic behind the segmentation patch to geo-polygon transformation is that first
        the boundary pixels for segmentation blobs is identified. Then, by following the closest 
        distance trajectory geo-polygons are formed. Then, if the jump (distance) between two closest point surpass 10 pixles
        (an arbitrary value) then the polygons are seprataed and the next polygon is built by following
        the trajectory.

        Parameters:
            Models (list): A list of models.
            damage_classes (list): List of damage classes; herein, ['C','D']
            
        Returns:
            The union of all segmentation patches as a single geometry file inside the folder 
            ./Results/Model/Shape_files/(C or D).parquet

            """
        cwd=self.cwd
        zigon=0
        l=Point()
        one_patch = gpd.read_parquet(cwd+'/Data/'+'Sample.parquet')
        try:
            os.mkdir(cwd+'/Results/'+Model+'/Shape_Files/')
        except:
            pass
        cc=1
        MASKS=cwd+'/Results/'+Model+'/Mask_Results_'+Damage_Cases+'/'
        F=os.listdir(MASKS)
        Img_Source=cwd+'/Data/CSVS/'
        enum_0=0
        enum_1=0
        for sub_dir in tqdm(F,desc='Masks to polygon for '+Model+' Damage case:\t'+Damage_Cases):
            im_mask=Image.open(MASKS+sub_dir)
            im_mask = ImageOps.grayscale(im_mask)
            im_new=np.array(im_mask)
            im_new[0,0]=0
            #enlarging images by 1 pixel at each side
            im_new=np.concatenate((np.zeros((self.H,1)),im_new,np.zeros((self.H,1))),axis=1)
            im_new=np.concatenate((np.zeros((1,self.W+2)),im_new,np.zeros((1,self.W+2))),axis=0)


            #Segmentation images into boundary pixels
            PX=np.zeros((0,0))
            PY=np.zeros((0,0))
            zet=0
            for i in range(self.W+1):
                for j in range(self.H+1):
                    if im_new[i,j]==im_new[i,j+1]:
                        if (im_new[i-1,j]==0 and im_new[i,j]>0) or (im_new[i+1,j]==0 and im_new[i,j]>0):
                            if zet==0:
                                PX=np.zeros((1))
                                PY=np.zeros((1))
                                PX[0]=j
                                PY[0]=640-i
                                zet+=1
                            else:
                                PX=np.concatenate((PX,np.asarray(j).reshape((-1))))
                                PY=np.concatenate((PY,np.asarray(640-i).reshape((-1))))
                    else:

                        if zet==0:
                            PX=np.zeros((1))
                            PY=np.zeros((1))
                            PX[0]=j
                            PY[0]=640-i
                            zet+=1
                        else:
                            PX=np.concatenate((PX,np.asarray(j).reshape((-1))))
                            PY=np.concatenate((PY,np.asarray(640-i).reshape((-1))))


            # By reading each block geometry, those pixels are now transformed into geometries
            MM=gpd.read_parquet(Img_Source+'Blocks_geom'+sub_dir[0:len(sub_dir)-4]+'.parquet')
            I=(MM['geometry'].iloc[0])
            VB=I.bounds
            #print(VB)
            Step_X=VB[2]-VB[0]
            Step_Y=VB[3]-VB[1]
            PXCO=PX*1
            PYCO=PY*1
            for i1 in range(len(PX)):
                PXCO[i1]=VB[0]+(PX[i1]/50)*Step_X
                PYCO[i1]=VB[1]+(1-PY[i1]/50)*Step_Y

            Z=[]
            ZC=[]
            for i in range(len(PX)):
                Z.append((PX[i],PY[i]))
                ZC.append((PXCO[i],PYCO[i]))



            PXN=np.zeros((len(PX),1))
            PYN=np.zeros((len(PX),1))
            PXN1=np.zeros((len(PX),1))
            PYN1=np.zeros((len(PX),1))
            zet=0

            HJ=pd.DataFrame(Z)
            HJ['Ind']=np.arange(0,len(HJ),1)
            if len(HJ)==0:
                Zgh=0
            else:
                Zgh=1
            if Zgh!=0:
                enum_1+=1
                X=HJ.iloc[0,0]
                Y=HJ.iloc[0,1]
                PXN[0]=X
                PYN[0]=Y
                PXN1[0]=VB[0]+(X/50)*Step_X
                PYN1[0]=VB[1]+(Y/50)*Step_Y
                HJ['D']=0
                HJ= HJ.drop(0)
                zet=0
                while len(HJ)>0:
                    HJ['D']=np.sqrt((HJ.iloc[:,0]-X)**2+(HJ.iloc[:,1]-Y)**2)
                    lp=HJ['Ind'].iloc[np.argmin(HJ['D'])]
                    X=HJ.iloc[np.argmin(HJ['D']),0]
                    Y=HJ.iloc[np.argmin(HJ['D']),1]
                    HJ= HJ.drop(HJ.index[np.argmin(HJ['D'])])
                    HJ['Ind']=np.arange(0,len(HJ),1)
                    zet+=1
                    PXN[zet]=X
                    PYN[zet]=Y
                    PXN1[zet]=VB[0]+(X/self.H)*Step_X
                    PYN1[zet]=VB[1]+(Y/self.W)*Step_Y
                enu_2=0
                enu_3=1
                for i in range(1,len(PXN)-1):
                    Dist=np.sqrt((PXN[i]-PXN[i+1])**2+(PYN[i]-PYN[i+1])**2)

                # A distance of 10 pixels to divide pathces if the closest point in them is greater than
                    if Dist>10 or i==len(PXN)-2:
                        Zr=[]
                        for j in range(enu_3+1,i):
                            Zr.append((PXN1[j],PYN1[j]))
                        enu_3=i+1
                        if len(Zr)>2:
                            polyA = Polygon(Zr)
                            polyA=make_valid(polyA)
        ################################                    
                            lpo=0
                            if "COLLECTION" in str(polyA):                        
                                for geoms in polyA.geoms:
                                    if "POLY" in str(geoms):
                                        if lpo==0:
                                            poly3=geoms
                                            lpo+=1
                                        else:
                                            poly3=poly3.union(geoms) 
                                poly=poly3
                            else:
                                poly=polyA
        ################


                            if enu_2==0:
                                l=poly 
                                enu_2+=1
                            else:
                                l=l.union(poly)
         
                if enu_3==1:
                        Zr=[]
                        for j in range(1,len(PXN1)):
                            Zr.append((PXN1[j],PYN1[j]))
                        if len(Zr)>2:
                            polyA = Polygon(Zr)  
                            polyA=make_valid(polyA)
        ################################                    
                            lpo=0
                            if "COLLECTION" in str(polyA):
                                for geoms in polyA.geoms:
                                    if "POLY" in str(geoms):
                                        if lpo==0:
                                            poly3=geoms
                                            lpo+=1
                                        else:
                                            poly3=poly3.union(geoms) 
                                l=poly3            
                            else:
                                l=polyA
                #########################                 
                        if "COLLECTION" in str(l):
                            for geoms in l.geoms:
                                if "POLY" in str(geoms):
                                    if lpo==0:
                                        poly3=geoms
                                        lpo+=1
                                    else:
                                        poly3=poly3.union(geoms) 
                            l=poly3            
                        else:
                            l=l  
                        #######################
                        if "LINE" in str(l):
                            pass
                        else:
                            if enum_0==0:
                                POLY=l
                                enum_0+=1
                            else:
                                POLY=POLY.union(l) 
                        if enum_1%cc==0:
                            one_patch=one_patch.iloc[0:1][['lat','lon','geometry']]
                            one_patch['geometry']=unary_union(POLY)
                            one_patch['I']=str(I)
                            one_patch['id']=sub_dir[0:len(sub_dir)-4]
                            if zigon==0:
                                patches=one_patch
                                zigon+=1
                            else:
                                kA=pd.DataFrame(one_patch)
                                kB=pd.DataFrame(patches)
                                kA=pd.concat((kB,kA))
                                patches=gpd.GeoDataFrame(kA)
                            enum_0=0

        #########################                 
                if "COLLECTION" in str(l):
                    for geoms in l.geoms:
                        if "POLY" in str(geoms):
                            if lpo==0:
                                poly3=geoms
                                lpo+=1
                            else:
                                poly3=poly3.union(geoms) 
                    l=poly3            
                else:
                    l=l  
        ###############################
                if "LINE" in str(l):
                    pass
                else:
                    if enum_0==0:
                        POLY=l
                        enum_0+=1
                    else:
                        POLY=POLY.union(l)

                if enum_1%cc==0:
                    one_patch=one_patch.iloc[0:1][['lat','lon','geometry']]
                    one_patch['geometry']=unary_union(POLY)
                    one_patch['id']=sub_dir[0:len(sub_dir)-4]
                    one_patch['I']=str(I)
                    if zigon==0:
                        patches=one_patch
                        zigon+=1
                    else:
                        kA=pd.DataFrame(one_patch)
                        kB=pd.DataFrame(patches)
                        kA=pd.concat((kB,kA))
                        patches=gpd.GeoDataFrame(kA)
                    enum_0=0
            else:
                pass
            try:
                if "COLLECTION" in str(POLY):
                    break
            except:
                pass
        patches.to_parquet(cwd+'/Results/'+Model+'/Shape_Files/'+Damage_Cases+'.parquet')
 

    def Poly_to_Matching(self,Models,T):
        for Model in Models:
            Source=cwd
            cwd=cwd
            CSVS_Source=Source+'/Data/CSVS/'
            A=gpd.read_parquet(Source+'/Results/'+Model+'/Shape_Files/D.parquet')
            B=gpd.read_parquet(Source+'/Results/'+Model+'/Shape_Files/C.parquet')
            F_A=Source+'FootPrints/'
            foot=gpd.read_parquet(cwd+'/Data/Inventory.parquet')
            foot['geometry']=foot['geometry'].buffer(0)
            foot['centroid']=foot['geometry'].centroid
            ID=['id', 'D0', 'P0', 'DR0', 'PR0', 'D1', 'P1', 'DR1', 'PR1', 'D2', 'P2',
                    'DR2', 'PR2', 'D3', 'P3', 'DR3', 'PR3', 'D4', 'P4', 'DR4', 'PR4', 'D5',
                    'P5', 'DR5', 'PR5', 'D6', 'P6', 'DR6', 'PR6', 'D7', 'P7', 'DR7', 'PR7',
                    'D8', 'P8', 'DR8', 'PR8', 'D9', 'P9', 'DR9', 'PR9', 'D10', 'P10',
                    'DR10', 'PR10', 'N']
            
            G=os.listdir(CSVS_Source)
            for iG in np.arange(0,len(G),1):
                if iG==0:
                    Geom=gpd.read_parquet(CSVS_Source+G[iG])
                    Geom['lat']=Geom.centroid.x
                    Geom=Geom[Geom.columns[::-1]]
                    Geom[ID]=0
                    Geom['id']=iG+1
                    #G[iG][G[iG].find('.')-1:G[iG].find('.')]
                    Geom1=Geom.copy()
                else:
                    Geom=gpd.read_parquet(CSVS_Source+G[iG])
                    Geom['lat']=Geom.centroid.x
                    Geom=Geom[Geom.columns[::-1]]
                    Geom[ID]=0
                    Geom['id']=iG+1
                    #G[iG][G[iG].find('.')-1:G[iG].find('.')]
                    try:
                        Geom1=Geom1.append(Geom)
                    except:
                        Geom1 = gpd.GeoDataFrame(pd.concat( (Geom1,Geom), ignore_index=True) )
            Sample=Geom1.copy()
        ######### C
            for i in np.arange(0,len(A),1):
                ind=int(A['id'].iloc[i])
                Mask=A['geometry'].iloc[i]
                I=(gpd.read_parquet(CSVS_Source+'Blocks_geom'+str(ind)+'.parquet')['geometry'].iloc[0])
                L=foot[foot['centroid'].within(I)]
                for i1 in range(len(L)):
                    KK=L['geometry'].iloc[i1]
                    try:
                        Overlap_R=(KK.intersection(Mask)).area/KK.area
                    except:
                        Overlap_R=(make_valid(KK).intersection(make_valid(Mask))).area/KK.area
                    if(Overlap_R>0.01):
                        Sample.iloc[ind,4]+=1
                    if(Overlap_R>0.1):
                        Sample.iloc[ind,8]+=1
                    if(Overlap_R>0.2):
                        Sample.iloc[ind,12]+=1
                    if(Overlap_R>0.3):
                        Sample.iloc[ind,16]+=1
                    if(Overlap_R>0.4):
                        Sample.iloc[ind,20]+=1
                    if(Overlap_R>0.5):
                        Sample.iloc[ind,24]+=1
                    if(Overlap_R>0.6):
                        Sample.iloc[ind,28]+=1
                    if(Overlap_R>0.7):
                        Sample.iloc[ind,32]+=1
                    if(Overlap_R>0.8):
                        Sample.iloc[ind,36]+=1
                    if(Overlap_R>0.9):
                        Sample.iloc[ind,38]+=1
                    if(Overlap_R>0.99):
                        Sample.iloc[ind,42]+=1
        # D
            for i in np.arange(0,len(B),1):
                ind=int(B['id'].iloc[i])
                Mask=B['geometry'].iloc[i]
                I=(gpd.read_parquet(CSVS_Source+'Blocks_geom'+str(ind)+'.parquet')['geometry'].iloc[0])
                L=foot[foot['centroid'].within(I)]
                for i1 in range(len(L)):
                    KK=L['geometry'].iloc[i1]
                    try:
                        Overlap_R=(KK.intersection(Mask)).area/KK.area
                    except:
                        Overlap_R=(make_valid(KK).intersection(make_valid(Mask))).area/KK.area
                    if(Overlap_R>0.01):
                        Sample.iloc[ind,3]+=1
                    if(Overlap_R>0.1):
                        Sample.iloc[ind,7]+=1
                    if(Overlap_R>0.2):
                        Sample.iloc[ind,11]+=1
                    if(Overlap_R>0.3):
                        Sample.iloc[ind,15]+=1
                    if(Overlap_R>0.4):
                        Sample.iloc[ind,19]+=1
                    if(Overlap_R>0.5):
                        Sample.iloc[ind,23]+=1
                    if(Overlap_R>0.6):
                        Sample.iloc[ind,27]+=1
                    if(Overlap_R>0.7):
                        Sample.iloc[ind,31]+=1
                    if(Overlap_R>0.8):
                        Sample.iloc[ind,35]+=1
                    if(Overlap_R>0.9):
                        Sample.iloc[ind,39]+=1
                    if(Overlap_R>0.99):
                        Sample.iloc[ind,43]+=1
            Sample.to_parquet(Source+'/Results/'+Model+'/Shape_Files/Block_RatiosT.parquet')


            A=np.zeros((22,1))
            z=[]
            for i in range(0,11):
                A[i*2+0]=np.sum(Sample['D'+str(i)])
                A[i*2+1]=np.sum(Sample['P'+str(i)])
                z.append(str(int(A[i*2+0][0]))+'('+str(100*int(A[i*2+0][0])/len(foot))[0:4]+'%'+')')
                z.append(str(int(A[i*2+1][0]))+'('+str(100*int(A[i*2+1][0])/len(foot))[0:4]+'%'+')')
            print(tabulate([['>10', z[0],z[1]], ['>20', z[2],z[3]], ['>30', z[4],z[5]], ['>40', z[6],z[7]],['>50', z[8],z[9]], ['>60',z[10],z[11]], ['>70', z[12],z[13]], ['>80', z[14],z[15]],['>90', z[16],z[17]]], headers=['Type','Collapsed','Possibly'], tablefmt='orgtbl'))         


            # Generate sample data (replace these with your actual data)
            data1 = np.asarray([A[0][0],A[2][0],A[4][0],A[6][0],A[8][0],A[10][0],A[12][0],A[14][0],A[16][0]])
            data2 = np.asarray([A[1][0],A[3][0],A[5][0],A[7][0],A[9][0],A[11][0],A[13][0],A[15][0],A[17][0]])

            # Fit lognormal distributions to the data
            params1 = lognorm.fit(data1,floc=0)
            params2 = lognorm.fit(data2,floc=0)


            # Plot the PDF of the fitted distributions
            x = np.linspace(0, max(data1.max(), data2.max())*1.5, 1000)
            pdf1 = lognorm.pdf(x, *params1)
            pdf1=pdf1/np.max(pdf1)
            pdf2 = lognorm.pdf(x, *params2)
            pdf2=pdf2/np.max(pdf2)
            plt.plot(x, pdf1, label='Fitted Lognorm C', color='blue')
            plt.plot(x, pdf2, label='Fitted Lognorm D', color='orange')

            # Add labels and legend
            plt.xlabel('NO. Footprints')
            plt.ylabel('Normalized Probability Density')
            plt.title('Fitted Lognormal Distributions, Model: '+Model)
            plt.legend()

            # Show plot
            plt.show()

            try:
                Label_gpd=gpd.read_parquet(cwd+'/Data/Geom_Labels.parquet')
                print('Labels are given, accuracy estimatation begins:')
                geod = Geod(ellps="WGS84")
                Damage_Cases=['C','D']
                #self.damages
                print('\n Accuracy estimation:')
                Foot=gpd.read_parquet(cwd+'/Data/Inventory.parquet')
                SQ=np.where(Foot.columns=='SQMETERS')[0][0]
                Foot[['PC','PD']]=0
                ff=len(Foot.columns)
                #Foot[Foot['Damage']==np.nan]['Damage']=0
                for kk2 in range(len(Damage_Cases)):
                    PP=gpd.read_parquet(cwd+'/Results/'+Model+'/Shape_files/'+Damage_Cases[kk2]+'.parquet')
                    Geom=unary_union(PP['geometry'])
                    for KL in range(len(Foot)):
                        Foot.iloc[KL,SQ]=abs(geod.geometry_area_perimeter(Foot['geometry'].iloc[KL])[0])
                        Foot.iloc[KL,ff-2+kk2]=(make_valid(Foot['geometry'].iloc[KL]).intersection(Geom)).area/(Foot['geometry'].iloc[KL].area)
                        
                        #Breaking the overlap-inventory parquet to three portions
                # one: no damage, second, damage D, thirs damage C
                Foot.to_parquet(Source+'/Results/'+Model+'/Shape_Files/Inventory_Results.parquet')
                Inv_overlap=Foot.copy()

                Z=Inv_overlap[Inv_overlap['PD']<T].copy()
                Z=Z[Z['PC']<T].copy()
                AP=Inv_overlap.drop(index=Z.index)
                ZD=AP[AP['PC']>=T].copy()
                AP=AP.drop(index=ZD.index)
                ZP=AP.copy()
                Z1=Z.copy()
                Z2=ZP.copy()
                Z3=ZD.copy()

                # Binary damage detection problem
                Z1['Label']=0
                Z2['Label']=1
                Z3['Label']=1
                LLL=pd.concat((Z1,pd.concat((Z2,Z3))))
                LLL=LLL.merge(Label_gpd[['geometry','Label']], on='geometry')
                Labels=LLL['Label_x'].copy()
                Pred=LLL['Label_y'].copy()
                Pred[Pred==2]=1
                DF1=precision_recall_fscore_support(Pred,Labels,average='macro')
                print(Model+"'s binary damage detection P-R-F1 scores=\n",'No damage:',str(len(Z1)),'\tDamage:',str(len(Z2)+len(Z3)))
                print(' Prec:\t',DF1[0],'\n Recall:\t',DF1[1],'\n F1:\t',DF1[2],'\n')

                # Three label damage detection problem
                Z1['Label']=0
                Z2['Label']=1
                Z3['Label']=2
                LLL=pd.concat((Z1,pd.concat((Z2,Z3))))
                LLL=LLL.merge(Label_gpd[['geometry','Label']], on='geometry')
                Labels=LLL['Label_x'].copy()
                Pred=LLL['Label_y'].copy()
                DF2=precision_recall_fscore_support(Pred,Labels,average='macro')
                print(Model+"'s damage severity detection P-R-F1 scores=\n",'D damage:',str(len(Z2)),'\tC Damage:',str(len(Z3)))
                print(' Prec:\t',DF2[0],'\n Recall:\t',DF2[1],'\n F1:\t',DF2[2],'\n')
            except:
                print('No Label is given')


        
    def Ensemble(self,Damage_Classes,rotate=False): 
        if rotate:
            Model='EnsembleR'
        else:
          Model='Ensemble'

        Damage_Classes=['D','C']
        Source=self.cwd
        try:
            os.mkdir(Source+'/Results')
        except:
            pass
        try:
            os.mkdir(Source+'/Results/'+Model)
            os.mkdir(Source+'/Results/'+Model+'/Shape_Files')
        except:
            pass
        for damage_class in Damage_Classes:
            try:
                os.mkdir(Source+'/Results/'+Model+'/Mask_Results_'+damage_class+'/')
            except:
                pass
        Models=['Model_2E_Eff','Model_2E_eff_cd','Model_2E_eff_ARIA','Model_2E_Tra','Model_2E_Tra_CD','Model_2E_Tra_ARIA']
        
        for damage_class in Damage_Classes:
            F=os.listdir(Source+'/Results/'+Models[0]+'/Mask_Results_'+damage_class+'/')
            for i1 in tqdm(range(1,len(F)+1),desc='Ensemble for:\t'+damage_class):
                if rotate:
                    im1=Image.open(Source+'/Results/'+Models[0]+'/Mask_Results_'+damage_class+'E/'+str(i1)+'.png')
                    im2=Image.open(Source+'/Results/'+Models[1]+'/Mask_Results_'+damage_class+'E/'+str(i1)+'.png')
                    im3=Image.open(Source+'/Results/'+Models[2]+'/Mask_Results_'+damage_class+'E/'+str(i1)+'.png')
                    im4=Image.open(Source+'/Results/'+Models[3]+'/Mask_Results_'+damage_class+'E/'+str(i1)+'.png')
                    im5=Image.open(Source+'/Results/'+Models[4]+'/Mask_Results_'+damage_class+'E/'+str(i1)+'.png')
                    im6=Image.open(Source+'/Results/'+Models[5]+'/Mask_Results_'+damage_class+'E/'+str(i1)+'.png')
                else:
                    im1=Image.open(Source+'/Results/'+Models[0]+'/Mask_Results_'+damage_class+'/'+str(i1)+'.png')
                    im2=Image.open(Source+'/Results/'+Models[1]+'/Mask_Results_'+damage_class+'/'+str(i1)+'.png')
                    im3=Image.open(Source+'/Results/'+Models[2]+'/Mask_Results_'+damage_class+'/'+str(i1)+'.png')
                    im4=Image.open(Source+'/Results/'+Models[3]+'/Mask_Results_'+damage_class+'/'+str(i1)+'.png')
                    im5=Image.open(Source+'/Results/'+Models[4]+'/Mask_Results_'+damage_class+'/'+str(i1)+'.png')
                    im6=Image.open(Source+'/Results/'+Models[5]+'/Mask_Results_'+damage_class+'/'+str(i1)+'.png')
                n=0
                n1=[0,0,0,0,0,0]
                pp1=np.asarray(im1)/255
                if np.sum(pp1)>200:
                    n+=1
                    n1[0]=1
                pp2=np.asarray(im2)/255
                if np.sum(pp2)>200:
                    n+=1
                    n1[1]=1
                pp3=np.asarray(im3)/255
                if np.sum(pp3)>200:
                    n+=1
                    n1[2]=1
                pp4=np.asarray(im4)/255
                if np.sum(pp4)>200:
                    n+=1
                    n1[3]=1
                pp5=np.asarray(im5)/255
                if np.sum(pp5)>200:
                    n+=1
                    n1[4]=1
                pp6=np.asarray(im6)/255
                if np.sum(pp6)>200:
                    n+=1
                    n1[5]=1
                if n==6:
                    PP=n1[0]*pp1+n1[1]*pp2+n1[2]*pp3+n1[3]*pp4+n1[4]*pp5+n1[5]*pp6
                    PP[PP>0]=1
                elif n>3:
                    PP=n1[0]*pp1+n1[1]*pp2+n1[2]*pp3+n1[3]*pp4+n1[4]*pp5+n1[5]*pp6
                    A=sum(n1)
                    PP[PP<=np.floor(n/2)]=0
                    PP[PP>np.floor(n/2)]=1
                else:
                    PP=0*pp1

                PP=Image.fromarray(np.uint8(PP*255), 'L')
                PP.save(Source+'/Results/'+Model+'/Mask_Results_'+damage_class+'/'+str(i1)+'.png')   
        self.Mask_To_Poly([Model],Damage_Classes)
                    
    def Rotation_Ens(self,Models,damage_cases):
        Source=self.cwd
        print('Ensemble across rotations:')
        for Model in tqdm(Models,desc='Ensemble across rotations: '):
            for damage_case in damage_cases:
                DIR = Source+'/Results/'+Model+'/Mask_Results_'+damage_case+'/'
                DIR3 = Source+'/Results/'+Model+'/Mask_Results_'+damage_case+'R/'
                DIR4 =  Source+'/Results/'+Model+'/Mask_Results_'+damage_case+'E/'
                try:
                    os.mkdir(DIR3)
                    os.mkdir(DIR4)
                except:
                    pass
                GG=os.listdir(DIR)
                for KL in (GG):
                    for degree in ([0,90,180,270]):
                        DIR1 = Source+'/Results/'+Model+'/Mask_Results_'+damage_case+'_'+str(degree)+'/'
                        img1  = Image.open(DIR1+KL).convert("L")
                        img1=img1.rotate(-degree)
                        U=np.asarray(img1)/255
                        if degree==0:
                            U1=U
                        else:
                            U1+=U 
                    KP=U1*1
                    U1[U1>=1]=1
                    #KP[KP>2]=1
                    Image.fromarray(np.uint8(U1 * 255) , 'L').save(DIR4+KL)
                    Image.fromarray(np.uint8(KP * 255) , 'L').save(DIR3+KL)
        self.Ensemble(damage_cases,rotate=True)
