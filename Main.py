#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
get_ipython().system('pip install git+https://github.com/qubvel/segmentation_models.pytorch')
import segmentation_models_pytorch as smp
try:
    import albumentations as A  
    from albumentations.pytorch import ToTensorV2
except:
    pass
from tqdm import tqdm
import matplotlib.pyplot as plt
from tqdm import tqdm


# In[2]:


class Turkiye_Seg():
    def __init__(self, Target, Model, Labels,Main_dir,Save_dir,Fourth_C=[False,'CD']):
        self.damages  = Labels
        self.Target  = Target
        self.Model  = Model
        self.Mdir  = Main_dir
        self.Sdir  = Save_dir
        self.CD=Fourth_C
        
    class SegmentationDataset(Dataset):
        import random
        def __init__(self, input_dir, is_train,Damage,Nor):
            import albumentations as A 
            IMAGE_WIDTH=640
            IMAGE_HEIGHT=640
            self.input_dir  = input_dir
            self.Damage=Damage
            if is_train == True:
                x = round(len(os.listdir(input_dir)) * 1)
                self.images = os.listdir(input_dir)[:x]
            else:
                x = round(len(os.listdir(input_dir)) * 1)
                self.images = os.listdir(input_dir)[x:]
                
                
                                  
            self.transform = A.Compose(
                [
                    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
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


            img=np.array(img, dtype=np.float32)/255
            cd =np.array((cd).convert('L'), dtype=np.float32)/255

            if self.transform is not None:
                augmentations = self.transform(image=img)
                img   = augmentations["image"]

            return img, cd
        
    class SegmentationDataset2(Dataset):
        import random
        def __init__(self, input_dir, is_train,Damage,Nor,degree):
            import albumentations as A 
            IMAGE_WIDTH=640
            IMAGE_HEIGHT=640
            self.input_dir  = input_dir
            self.rotate=degree
            self.Damage=Damage
            if is_train == True:
                x = round(len(os.listdir(input_dir)) * 1)
                self.images = os.listdir(input_dir)[:x]
            else:
                x = round(len(os.listdir(input_dir)) * 1)
                self.images = os.listdir(input_dir)[x:]
                
                
                                  
            self.transform = A.Compose(
                [
                    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
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
            img=img.rotate(self.rotate)
            cd=cd.rotate(self.rotate)
            ####
     

            img=np.array(img, dtype=np.float32)/255
            cd =np.array((cd).convert('L'), dtype=np.float32)/255

            if self.transform is not None:
                augmentations = self.transform(image=img)
                img   = augmentations["image"]
            return img, cd
        
        
    def Model_Mask(self):
        import os
        Model=self.Model
        Source=self.Mdir
        Save=self.Sdir
        CD=self.CD
        Target=[self.Target]
        ggs=self.damages
        try:
            os.mkdir(Save+Model)
            os.mkdir(Save+Model+'/Shape_Files')
        except:
            pass
        for t in range(len(Target)):
            try:
                os.mkdir(Save+Model+'/'+Target[t])
            except:
                pass

        DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

        LEARNING_RATE = 3e-4
        BATCH_SIZE    = 1
        NUM_EPOCHS    = 10
        IMAGE_HEIGHT  = 640
        IMAGE_WIDTH   = 640


        for i in range(len(ggs)):
            try:
                os.mkdir(Save+Model+'/'+Target[0]+'/Mask_Results_'+ggs[i]+'/')
            except:
                pass
        ggs2=self.damages
        batch_size=BATCH_SIZE
        Source=self.Mdir
        for kk1 in range(len(Target)):
                for kk2 in range(len(ggs)):
                    gg=ggs[kk2]
                    print(Target[kk1]+' Mask Generatrion for:\t'+gg )
                    if CD[3]=='Pre':
                        TRAIN_INP_DIR = Source+Target[0]+'/All_m'
                    else:
                        TRAIN_INP_DIR = Source+Target[0]+'/G_Plus_'+CD[1]
                    train_ds=self.SegmentationDataset(TRAIN_INP_DIR,True,ggs2[kk2],self.CD)
                    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False )
                    #train_loader2 = DataLoader(train_ds2, batch_size=batch_size, shuffle=False )
                    #print(len(train_ds))
                    try:
                        model=torch.load(Source+'Deep_Models/'+Model+ggs2[kk2][0]+'.pth')['Model']
                    except:
                        model=torch.load(Source+'Deep_Models/'+Model+ggs2[kk2][0]+'.pth')
                    model.eval()
                    #model.encoder._dropout.train()
                    #model.encoder._dropout.p=1
                    #model.encoder._dropout.train()
                    DIR = Save+Model+'/'+Target[0]+'/Mask_Results_'+ggs[kk2]+'/'
                    zet=0
                    import os
                    F=os.listdir(TRAIN_INP_DIR)
                    #ll=iter(train_loader2)
                    pep=0
                    for image, cd in tqdm(train_loader):
                        #############
                        #
                        if (CD[0] and CD[2]=='Tra'):
                            img   = image[:,0:2,:,:].to(device=DEVICE)
                        else:
                            img   = image[:,:,:,:].to(device=DEVICE)
                        imageCD1   = cd.to(device=DEVICE).reshape(BATCH_SIZE,1,640,640)
                        if CD[0]:
                            img=torch.cat((img,imageCD1),axis=1)
                        output= ((torch.sigmoid(model(img.to('cuda')))) >0.5).float()
                        ###################
                        #inputs2, masks2=next(ll)
                        #image2   = inputs[:,:,:,0:640].to(device=DEVICE)
                        #imageCD2   = inputs[:,0:1,:,640:1280].to(device=DEVICE)
                        #inputs=torch.cat((image2,imageCD2),axis=1)
                        #output=(img1)
                        #image2 = torch.sigmoid(model(image2))
                        #image2 = (image2 > 0.5).float()
                        #image2=model.forward(image2)
                        #image2[image2<=0.5]=0
                        #image2[image2>0.5]=1
                        #output= ((torch.sigmoid(model(inputs.to('cuda')))) >0.5).float()
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
                        #if pep==4:
                            #break
                    #break
                    
    def Model_Mask2(self):
        import os
        Model=self.Model
        Source=self.Mdir
        Save=self.Sdir
        CD=self.CD
        Target=[self.Target]
        ggs=self.damages
        try:
            os.mkdir(Save+Model)
            os.mkdir(Save+Model+'/Shape_Files')
        except:
            pass
        for t in range(len(Target)):
            try:
                os.mkdir(Save+Model+'/'+Target[t])
            except:
                pass

        DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

        LEARNING_RATE = 3e-4
        BATCH_SIZE    = 1
        NUM_EPOCHS    = 10
        IMAGE_HEIGHT  = 640
        IMAGE_WIDTH   = 640


        for i in range(len(ggs)):
            try:
                os.mkdir(Save+Model+'/'+Target[0]+'/Mask_Results_'+ggs[i]+'/')
            except:
                pass
        ggs2=self.damages
        batch_size=BATCH_SIZE
        Source=self.Mdir
        for degree in tqdm([0,90,180,270]):
            for kk1 in range(len(Target)):
                    for kk2 in range(len(ggs)):
                        gg=ggs[kk2]
                        print(Target[kk1]+' Mask Generatrion for:\t'+gg )
                        if CD[3]=='Pre':
                            TRAIN_INP_DIR = Source+Target[0]+'/All_m'
                        else:
                            TRAIN_INP_DIR = Source+Target[0]+'/G_Plus_'+CD[1]
                        train_ds=self.SegmentationDataset2(TRAIN_INP_DIR,True,ggs2[kk2],self.CD,degree)
                        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False )
                        #train_loader2 = DataLoader(train_ds2, batch_size=batch_size, shuffle=False )
                        #print(len(train_ds))
                        try:
                            model=torch.load(Source+'Deep_Models/'+Model+ggs2[kk2][0]+'.pth')['Model']
                        except:
                            model=torch.load(Source+'Deep_Models/'+Model+ggs2[kk2][0]+'.pth')
                        model.eval()
                        #model.encoder._dropout.train()
                        #model.encoder._dropout.p=1
                        #model.encoder._dropout.train()
                        DIR = Save+Model+'/'+Target[0]+'/Mask_Results_'+ggs[kk2]+'_D_'+str(degree)+'/'
                        try:
                            os.mkdir(DIR)
                        except:
                            pass
                        zet=0
                        import os
                        F=os.listdir(TRAIN_INP_DIR)
                        #ll=iter(train_loader2)
                        pep=0
                        for image, cd in tqdm(train_loader):
                            #############
                            #
                            if (CD[0] and CD[2]=='Tra'):
                                img   = image[:,0:2,:,:].to(device=DEVICE)
                            else:
                                img   = image[:,:,:,:].to(device=DEVICE)
                            imageCD1   = cd.to(device=DEVICE).reshape(BATCH_SIZE,1,640,640)
                            if CD[0]:
                                img=torch.cat((img,imageCD1),axis=1)
                            output= ((torch.sigmoid(model(img.to('cuda')))) >0.5).float()
                            ###################
                            #inputs2, masks2=next(ll)
                            #image2   = inputs[:,:,:,0:640].to(device=DEVICE)
                            #imageCD2   = inputs[:,0:1,:,640:1280].to(device=DEVICE)
                            #inputs=torch.cat((image2,imageCD2),axis=1)
                            #output=(img1)
                            #image2 = torch.sigmoid(model(image2))
                            #image2 = (image2 > 0.5).float()
                            #image2=model.forward(image2)
                            #image2[image2<=0.5]=0
                            #image2[image2>0.5]=1
                            #output= ((torch.sigmoid(model(inputs.to('cuda')))) >0.5).float()
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
                            #if pep==4:
                                #break
                        #break                    

    def Mask_To_Poly(self):
        Damage_Cases=self.damages
        for i in range(len(Damage_Cases)):
            self.Mask_To_Polygon(Damage_Cases[i]);
        
    def Mask_To_Polygon(self,damage):
        from shapely.geometry import Polygon
        from shapely.geometry import Point
        import geopandas as gpd
        from shapely import wkt
        import matplotlib.pyplot as plt
        import pandas as pd
        from shapely.geometry import Polygon
        from shapely.validation import make_valid
        import geopandas as gpd
        from shapely.ops import unary_union
        Model=self.Model
        Source=self.Mdir
        Target=[self.Target ]
        Damage_Cases=[damage]
        from PIL import Image, ImageOps
        Model=Model
        Shape_file_Ad=self.Mdir+Model+'/Shape_Files/'
        zigon=0
        l=Point()
        for Enu_1 in range(len(Target)):
            for Enu_2 in range(len(Damage_Cases)):
                print('Masks to polygon for', Target[Enu_1], 'Damage case:\t',Damage_Cases[Enu_2])
                Zel = gpd.read_parquet(self.Mdir+'Sample.parquet')
                try:
                    os.mkdir(self.Mdir+Model+'/Shape_Files/')
                except:
                    pass
                cc=1
                POLYY=Polygon()
                MASKS=self.Mdir+Model+'/'+Target[Enu_1]+'/Mask_Results_'+Damage_Cases[0]+'/'
                if Damage_Cases[Enu_2]=='D':
                    print('D')
                else:
                    print('P')
                import os
                F=os.listdir(MASKS)
                Img_Source=self.Mdir+Target[Enu_1]+'/CSVS/'
                zepeleshk=0
                zepeleshk1=0
                #a=input('aa')
                i1n=0
                for Enu in tqdm(F):
                    #print(zepeleshk1)
                    A=Image.open(MASKS+F[i1n])
                    #plt.imshow(A)
                    A = ImageOps.grayscale(A)
                    import numpy as np
                    AA=np.array(A)
                    AA[0,0]=0
                    AA.shape
                    AR=np.zeros((640,1))
                    AR1=np.zeros((1,642))
                    AA=np.concatenate((AR,AA,AR),axis=1)
                    AA=np.concatenate((AR1,AA,AR1),axis=0)
                    AA.shape

                    PX=np.zeros((0,0))
                    PY=np.zeros((0,0))
                    zet=0
                    for i in range(641):
                        for j in range(641):
                            if AA[i,j]==AA[i,j+1]:
                                if (AA[i-1,j]==0 and AA[i,j]>0) or (AA[i+1,j]==0 and AA[i,j]>0):
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
                                penis=1
                                if zet==0:
                                    PX=np.zeros((1))
                                    PY=np.zeros((1))
                                    PX[0]=j
                                    PY[0]=640-i
                                    zet+=1
                                else:
                                    PX=np.concatenate((PX,np.asarray(j).reshape((-1))))
                                    PY=np.concatenate((PY,np.asarray(640-i).reshape((-1))))


                    #plt.plot(PX,PY,'*')
                    #MM=pd.read_csv(Img_Source+F[i1n][0:len(F[i1n])-4]+'.csv')
                    MM=gpd.read_parquet(Img_Source+'Blocks_geom'+F[i1n][0:len(F[i1n])-4]+'.parquet')
                    #print(Img_Source+F[i1][0:len(F[i1])-4]+'.csv')
                    #I=wkt.loads(MM['geometry'].iloc[0])
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




                    PXC=PX*1
                    PYC=PY*1
                    PXN=np.zeros((len(PX),1))
                    PYN=np.zeros((len(PX),1))
                    PXN1=np.zeros((len(PX),1))
                    PYN1=np.zeros((len(PX),1))
                    zet=0
                    ids=[]
                    HJ=pd.DataFrame(Z)
                    HJ['Ind']=np.arange(0,len(HJ),1)
                    if len(HJ)==0:
                        Zgh=0
                    else:
                        Zgh=1
                    if Zgh!=0:
                        zepeleshk1+=1
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
                            PXN1[zet]=VB[0]+(X/640)*Step_X
                            PYN1[zet]=VB[1]+(Y/640)*Step_Y
                        zebel=0
                        zepek=1
                        for i in range(1,len(PXN)-1):
                            Dist=np.sqrt((PXN[i]-PXN[i+1])**2+(PYN[i]-PYN[i+1])**2)
                            #print(i,len(PXN)-1)
                            if Dist>10 or i==len(PXN)-2:
                                #print(Dist,zepek,i)
                                Zr=[]
                                for j in range(zepek+1,i):
                                    Zr.append((PXN1[j],PYN1[j]))
                                zepek=i+1
                                if len(Zr)>2:
                                    #print('hio')
                                    polyA = Polygon(Zr)
                                    polyA=make_valid(polyA)
                ################################                    
                                    lpo=0
                                    if "COLLECTION" in str(polyA):                        
                                        for geoms in polyA.geoms:
                                            if "POLY" in str(geoms):
                                                #print('hi\n\n',str(geoms))
                                                if lpo==0:
                                                    poly3=geoms
                                                    lpo+=1
                                                else:
                                                    poly3=poly3.union(geoms) 
                                        poly=poly3
                                        #print('hi\n\n',str(poly))
                                    else:
                                        poly=polyA
                                        #print('hi\n\n',str(poly))
                ################


                                    if zebel==0:
                                        l=poly 
                                        zebel+=1
                                    else:
                                        l=l.union(poly)

                        #print('--L--\n\n\n\n',str(l))           
                        if zepek==1:
                                #print('hiooo')
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
                                    if zepeleshk==0:
                                        POLY=l
                                        zepeleshk+=1
                                    else:
                                        POLY=POLY.union(l) 
                                if zepeleshk1%cc==0:
                                    Zel=Zel.iloc[0:1][['lat','lon','geometry']]
                                    Zel['geometry']=unary_union(POLY)
                                    Zel['I']=str(I)
                                    Zel['id']=F[i1n][0:len(F[i1n])-4]
                                    if zigon==0:
                                        ZEL=Zel
                                        zigon+=1
                                    else:
                                        kA=pd.DataFrame(Zel)
                                        kB=pd.DataFrame(ZEL)
                                        kA=pd.concat((kB,kA))
                                        ZEL=gpd.GeoDataFrame(kA)
                                    zepeleshk=0

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
                            if zepeleshk==0:
                                POLY=l
                                zepeleshk+=1
                            else:
                                POLY=POLY.union(l)

                        if zepeleshk1%cc==0:
                            Zel=Zel.iloc[0:1][['lat','lon','geometry']]
                            Zel['geometry']=unary_union(POLY)
                            Zel['id']=F[i1n][0:len(F[i1n])-4]
                            Zel['I']=str(I)
                            if zigon==0:
                                ZEL=Zel
                                zigon+=1
                            else:
                                kA=pd.DataFrame(Zel)
                                kB=pd.DataFrame(ZEL)
                                kA=pd.concat((kB,kA))
                                ZEL=gpd.GeoDataFrame(kA)
                            zepeleshk=0
                    else:
                        pass
                    try:
                        if "COLLECTION" in str(POLY):
                            break
                    except:
                        pass
                    i1n+=1
                ZEL.to_parquet(Shape_file_Ad+Target[Enu_1]+Damage_Cases[Enu_2]+'.parquet')
 

    def Poly_to_Matching(self):
        from shapely.geometry import Polygon
        import geopandas as gpd
        from shapely import wkt
        import matplotlib.pyplot as plt
        import pandas as pd
        from shapely.geometry import Polygon
        from shapely.validation import make_valid
        import geopandas as gpd
        from shapely.ops import unary_union
        from tqdm import tqdm
        Damage_Cases=self.damages
#############        
        Area=[self.Target]
        Source=self.Mdir+'/'
        Model=self.Model
        CVS=Source+Area[0]+'/CSVS/'
        A=gpd.read_parquet(Source+Model+'/Shape_Files/'+Area[0]+'P.parquet')
        B=gpd.read_parquet(Source+Model+'/Shape_Files/'+Area[0]+'D.parquet')
        F_A=Source+'FootPrints/'
        foot=gpd.read_parquet(F_A+Area[0]+'L4.parquet')
        foot['centroid']=foot['geometry'].centroid
        try:
            Sample=gpd.read_parquet(F_A+Area[0]+'_Sample.parquet')
        except:
            cwd=self.Mdir
            Path=cwd+'CSVS/'
            ID=['id', 'D0', 'P0', 'DR0', 'PR0', 'D1', 'P1', 'DR1', 'PR1', 'D2', 'P2',
                   'DR2', 'PR2', 'D3', 'P3', 'DR3', 'PR3', 'D4', 'P4', 'DR4', 'PR4', 'D5',
                   'P5', 'DR5', 'PR5', 'D6', 'P6', 'DR6', 'PR6', 'D7', 'P7', 'DR7', 'PR7',
                   'D8', 'P8', 'DR8', 'PR8', 'D9', 'P9', 'DR9', 'PR9', 'D10', 'P10',
                   'DR10', 'PR10', 'N']
            import numpy as np
            import pandas as pd
            G=os.listdir(Path)
            for iG in np.arange(0,len(G),1):
                if iG==0:
                    Geom=gpd.read_parquet(Path+G[iG])
                    Geom['lat']=Geom.centroid.x
                    Geom=Geom[Geom.columns[::-1]]
                    Geom[ID]=0
                    Geom['id']=iG+1
                    #G[iG][G[iG].find('.')-1:G[iG].find('.')]
                    Geom1=Geom.copy()
                else:
                    Geom=gpd.read_parquet(Path+G[iG])
                    Geom['lat']=Geom.centroid.x
                    Geom=Geom[Geom.columns[::-1]]
                    Geom[ID]=0
                    Geom['id']=iG+1
                    #G[iG][G[iG].find('.')-1:G[iG].find('.')]
                    try:
                        Geom1=Geom1.append(Geom)
                    except:
                        Geom1 = gpd.GeoDataFrame(pd.concat( (Geom1,Geom), ignore_index=True) )
            Geom1.to_parquet(F_A+Area[0]+'_Sample.parquet')
            Sample=gpd.read_parquet(F_A+Area[0]+'_Sample.parquet')
 #########
        cwd=self.Mdir
        import numpy as np
        Path=cwd+'CSVS/'
        for i in tqdm(np.arange(0,len(A),1)):
            ind=int(A['id'].iloc[i])
            Mask=A['geometry'].iloc[i]
            I=(gpd.read_parquet(CVS+'Blocks_geom'+str(ind)+'.parquet')['geometry'].iloc[0])
            #I=wkt.loads((pd.read_csv(CVS+str(ind)+'.csv'))['geometry'].iloc[0])
            L=foot[foot['centroid'].within(I)]
            for i1 in range(len(L)):
                KK=L['geometry'].iloc[i1]
                try:
                    A_A=(KK.intersection(Mask)).area/KK.area
                except:
                    A_A=(make_valid(KK).intersection(make_valid(Mask))).area/KK.area
                if(A_A>0.01):
                    Sample.iloc[ind,4]+=1
                if(A_A>0.1):
                    Sample.iloc[ind,8]+=1
                if(A_A>0.2):
                    Sample.iloc[ind,12]+=1
                if(A_A>0.3):
                    Sample.iloc[ind,16]+=1
                if(A_A>0.4):
                    Sample.iloc[ind,20]+=1
                if(A_A>0.5):
                    Sample.iloc[ind,24]+=1
                if(A_A>0.6):
                    Sample.iloc[ind,28]+=1
                if(A_A>0.7):
                    Sample.iloc[ind,32]+=1
                if(A_A>0.8):
                    Sample.iloc[ind,36]+=1
                if(A_A>0.9):
                    Sample.iloc[ind,38]+=1
                if(A_A>0.99):
                    Sample.iloc[ind,42]+=1

        for i in tqdm(np.arange(0,len(B),1)):
            ind=int(B['id'].iloc[i])
            Mask=B['geometry'].iloc[i]
            #I=wkt.loads((pd.read_csv(CVS+str(ind)+'.csv'))['geometry'].iloc[0])
            I=(gpd.read_parquet(CVS+'Blocks_geom'+str(ind)+'.parquet')['geometry'].iloc[0])
            L=foot[foot['centroid'].within(I)]
            for i1 in range(len(L)):
                KK=L['geometry'].iloc[i1]
                try:
                    A_A=(KK.intersection(Mask)).area/KK.area
                except:
                    A_A=(make_valid(KK).intersection(make_valid(Mask))).area/KK.area
                if(A_A>0.01):
                    Sample.iloc[ind,3]+=1
                if(A_A>0.1):
                    Sample.iloc[ind,7]+=1
                if(A_A>0.2):
                    Sample.iloc[ind,11]+=1
                if(A_A>0.3):
                    Sample.iloc[ind,15]+=1
                if(A_A>0.4):
                    Sample.iloc[ind,19]+=1
                if(A_A>0.5):
                    Sample.iloc[ind,23]+=1
                if(A_A>0.6):
                    Sample.iloc[ind,27]+=1
                if(A_A>0.7):
                    Sample.iloc[ind,31]+=1
                if(A_A>0.8):
                    Sample.iloc[ind,35]+=1
                if(A_A>0.9):
                    Sample.iloc[ind,39]+=1
                if(A_A>0.99):
                    Sample.iloc[ind,43]+=1
        Sample.to_parquet(self.Mdir+Model+'/Shape_Files/'+Area[0]+'_Block_RatiosT.parquet')

        
        from tabulate import tabulate
        A=np.zeros((22,1))
        z=[]
        for i in range(0,11):
            A[i*2+0]=np.sum(Sample['D'+str(i)])
            A[i*2+1]=np.sum(Sample['P'+str(i)])
            z.append(str(int(A[i*2+0][0]))+'('+str(100*int(A[i*2+0][0])/len(foot))[0:4]+'%'+')')
            z.append(str(int(A[i*2+1][0]))+'('+str(100*int(A[i*2+1][0])/len(foot))[0:4]+'%'+')')
        print(tabulate([['>10', z[0],z[1]], ['>20', z[2],z[3]], ['>30', z[4],z[5]], ['>40', z[6],z[7]],['>50', z[8],z[9]], ['>60',z[10],z[11]], ['>70', z[12],z[13]], ['>80', z[14],z[15]]], headers=['Type','Collapsed','Possibly'], tablefmt='orgtbl'))
        
        
        
    def Ensemble(self): 
        #Model=self.Model
        Source=self.Mdir
        Target=[self.Target]
        Damage_Cases=self.damages
        Nor=CD=self.CD[3]
        from tqdm import tqdm
        Model=[self.Model]
        for kk in range(len(Target)):
            try:
                os.mkdir(Source+Model[0])
                os.mkdir(Source+Model[0]+'/Shape_Files')
            except:
                pass
            for i in range(len(Target)):
                try:
                    os.mkdir(Source+Model[0]+'/'+Target[i])
                    os.mkdir(Source+Model[0]+'/'+Target[i]+'/'+'Mask_Results_D')
                    os.mkdir(Source+Model[0]+'/'+Target[i]+'/'+'Mask_Results_P')
                except:
                    pass
            from tqdm import tqdm    
            Models=['Model_2E_Eff','Model_2E_eff_cd','Model_2E_eff_aria','Model_2E_Tra','Model_2E_Tra_CD','Model_2E_Tra_Aria']

            for i in range(len(Damage_Cases)):
                i1=1
                print('Ensemble for\t',Target[kk],'Damage Case',Damage_Cases[i])
                F=os.listdir(Source+Models[0]+'/'+Target[kk]+'/Mask_Results_D'+Nor)
                for OLP in tqdm(F):
                    im1=Image.open(Source+Models[0]+'/'+Target[kk]+'/Mask_Results_'+Damage_Cases[i]+Nor+'/'+str(i1)+'.png')
                    im2=Image.open(Source+Models[1]+'/'+Target[kk]+'/Mask_Results_'+Damage_Cases[i]+Nor+'/'+str(i1)+'.png')
                    im3=Image.open(Source+Models[2]+'/'+Target[kk]+'/Mask_Results_'+Damage_Cases[i]+Nor+'/'+str(i1)+'.png')
                    im4=Image.open(Source+Models[3]+'/'+Target[kk]+'/Mask_Results_'+Damage_Cases[i]+Nor+'/'+str(i1)+'.png')
                    im5=Image.open(Source+Models[4]+'/'+Target[kk]+'/Mask_Results_'+Damage_Cases[i]+Nor+'/'+str(i1)+'.png')
                    im6=Image.open(Source+Models[5]+'/'+Target[kk]+'/Mask_Results_'+Damage_Cases[i]+Nor+'/'+str(i1)+'.png')
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
                    PP.save(Source+Model[0]+'/'+Target[kk]+'/'+'Mask_Results_'+Damage_Cases[i]+'/'+str(i1)+'.png')
                    i1+=1   
                    
                    
    def Rotation_Ens(self,Models):
        Source=self.Mdir
        Save=self.Mdir
        Target=[self.Target]
        ggs=self.damages
        print('Running for Models')
        for Model in tqdm(Models):
            for kk2 in range(len(ggs)):
                DIR = Save+Model+'/'+Target[0]+'/Mask_Results_'+ggs[kk2]+'/'
                DIR3 = Save+Model+'/'+Target[0]+'/Mask_Results_'+ggs[kk2]+'R/'
                DIR4 = Save+Model+'/'+Target[0]+'/Mask_Results_'+ggs[kk2]+'E/'
                try:
                    os.mkdir(DIR3)
                    os.mkdir(DIR4)
                except:
                    pass
                GG=os.listdir(DIR)
                for KL in (GG):
                    for degree in ([0,90,180,270]):
                        DIR1 = Save+Model+'/'+Target[0]+'/Mask_Results_'+ggs[kk2]+'_D_'+str(degree)+'/'
                        img1  = Image.open(DIR1+KL).convert("L")
                        img1=img1.rotate(-degree)
                        U=np.asarray(img1)/255
                        if degree==0:
                            U1=U
                        else:
                            U1+=U 
                    KP=U1*1
                    U1[U1>=1]=1
                    KP[KP>2]=1
                    #U1[U1==1]=255
                    #KP[KP==1]=255
                    imgA = Image.fromarray(np.uint8(U1 * 255) , 'L').save(DIR4+KL)
                    imgB = Image.fromarray(np.uint8(KP * 255) , 'L').save(DIR3+KL)
                    
    def Footprints(self):
        from shapely.ops import unary_union
        from tqdm import tqdm
        from sklearn.metrics import f1_score
        from sklearn.metrics import precision_recall_fscore_support
        import geopandas as gpd
        import numpy as np
        from pyproj import Geod
        from shapely import wkt
        from shapely.validation import explain_validity
        from shapely.validation import make_valid
        geod = Geod(ellps="WGS84")
        Source='D:/Fire/'
        Save='D:/Fire/'
        Target=['Lahaina']
        Model=self.Model
        #Model='EnsembleR'
        #[self.Target]
        ggs=['D','P']
        #self.damages
        print('Running for Footprints')
        Foot=gpd.read_parquet(Save+'Footprints/LahainaL3.parquet')
        SQ=np.where(Foot.columns=='SQMETERS')[0][0]
        Foot[['PD','PP']]=0
        ff=len(Foot.columns)
        Foot['L']=''
        Foot['Label']=0
        Foot[Foot['Damage']==np.nan]['Damage']=0
        for kk2 in range(len(ggs)):
            PP=gpd.read_parquet('D:/Fire/'+Model+'/Shape_Files/Lahaina'+ggs[kk2]+'.parquet')
            Geom=unary_union(PP['geometry'])
            for KL in range(len(Foot)):
                Foot.iloc[KL,SQ]=abs(geod.geometry_area_perimeter(Foot['geometry'].iloc[KL])[0])
                Foot.iloc[KL,ff-4+kk2]=(make_valid(Foot['geometry'].iloc[KL]).intersection(Geom)).area/(Foot['geometry'].iloc[KL].area)
                #print((make_valid(Foot['geometry'].iloc[KL]).intersection(Geom)).area/(Foot['geometry'].iloc[KL].area))
                if kk2==1:
                    if Foot.iloc[KL,ff-4+0]>Foot.iloc[KL,ff-4+1] and Foot.iloc[KL,ff-4+0]>0:
                        Foot.iloc[KL,ff-4+kk2+1]='D'
                        if Foot.iloc[KL,ff-4+0]>0.1:
                            Foot.iloc[KL,ff-4+kk2+1+1]=1
                    elif Foot.iloc[KL,ff-4+1]>0:
                        Foot.iloc[KL,ff-4+kk2+1]='P'
                        if Foot.iloc[KL,ff-4+1+0]>0.1:
                            Foot.iloc[KL,ff-4+kk2+1+1]=1
        Foot.to_parquet('D:/Fire/'+Model+'/Shape_Files/'+Target[0]+'F2.parquet') 
        print("P-R-F1 scores=\n",'No damage:',str(len(Foot[Foot['Damage']==0])),'\tDamage:',str(len(Foot[Foot['Damage']==1])))
        N=len(Foot)
        AS=Foot[Foot['Label']==1]['Damage'].to_numpy()
        BS=Foot[Foot['Label']==0]['Damage'].to_numpy()
        TP=len(AS[AS==1])
        FN=len(AS[AS==0])
        TN=len(BS[BS==0])
        FP=len(BS[BS==1])
        print(' Prec:\t',precision_recall_fscore_support(Foot['Damage'],Foot['Label'])[0],'\n Recall:\t',precision_recall_fscore_support(Foot['Damage'],Foot['Label'])[1],'\n F1:\t',precision_recall_fscore_support(Foot['Damage'],Foot['Label'])[2],'\n',[TP,FP,TN,FN])
        
    def BTYPE(self):
        FEMA_1_dic = {
          'Commercial': 'Com',
          'Government': "Gov",
           'Industrial':"Ind",
            'Residential': "Res",
            'Education':"Edu",
            '':''
        }

        nsi_1_dic = {
          "COM1": 'Com',
          "COM2": 'Com',
          "COM3": 'Com',
          "COM4": 'Com',
          "COM5": 'Com',
          "COM6": 'Com',
          "COM7": 'Com',
          "COM8": 'Com',
          "COM9": 'Com',
          "COM10": 'Com',
          "	GOV1":'Gov',
          "IND1": 'Ind',
          "IND2": 'Ind',
          "IND3": 'Ind',
          "IND4":'Ind',
          "IND5": 'Ind',
          "IND6": 'Ind',
         "RES1-1SNB":'Res',
         "RES1-1SWB":'Res',
         "RES1-2SNB":'Res',
         "RES1-2SWB":'Res',
         "RES1-3SNB":'Res',
         "RES1-3SWB":'Res',
         "RES1-SLNB":'Res',
         "RES1-SLWB":'Res',
         "RES2":'Res',
         "RES3A":'Res',
         "RES3B":'Res',
         "RES3C":'Res',
         "RES3D":'Res',
         "RES3E":'Res',
         "RES3F":'Res',
         "RES4":'Res',
         "RES5":'Res',
         "RES6":'Res', 
         "EDU1":'Edu',
         "EDU2":'Edu',
            '':''
        }

        Foot['B-Type']=''
        Foot['B-Sto']=1
        Foot['B-Base']=0
        Foot['B-Unit']=1
        NT=np.where(Foot.columns=='occtype')[0]
        FT=np.where(Foot.columns=='OCC_CLS')[0]
        BT=np.where(Foot.columns=='B-Type')[0]
        #occtype=74
        #Sec_OCC=10
        #PRIM_OCC=9
        #bldgtype_right=75
        zet=0
        for i in tqdm(np.arange(0,len(Foot),1)):
            try:
                A=FEMA_1_dic[Foot.iloc[i,FT][0]]
            except:
                A=''
            try:
                B=nsi_1_dic[Foot.iloc[i,NT][0]]
            except:
                B=''
            if len(B)>0:
                Foot.iloc[i,BT]=B
                if "NB" in Foot.iloc[i,NT][0]:
                    pass
                else:
                    Foot.iloc[i,BT+2]=1

                if "SL" in Foot.iloc[i,NT][0]:
                    Foot.iloc[i,BT+1]=1.5
                if "2S" in Foot.iloc[i,NT][0]:
                    Foot.iloc[i,BT+1]=2
                if "3S" in Foot.iloc[i,NT][0]:
                    Foot.iloc[i,BT+1]=3

                if "3A" in Foot.iloc[i,NT][0]:
                    Foot.iloc[i,BT+3]=2
                if "3B" in Foot.iloc[i,NT][0]:
                    Foot.iloc[i,BT+3]=3.5
                if "3C" in Foot.iloc[i,NT][0]:
                    Foot.iloc[i,BT+3]=7.5
                if "3D" in Foot.iloc[i,NT][0]:
                    Foot.iloc[i,BT+3]=20
                if "3E" in Foot.iloc[i,NT][0]:
                    Foot.iloc[i,BT+3]=35
                if "3F" in Foot.iloc[i,NT][0]:
                    Foot.iloc[i,BT+3]=50

                zet+=1
            elif len(A)>0:
                Foot.iloc[i,BT]=A
                zet+=1
        zet