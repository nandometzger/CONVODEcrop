import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pylab import savefig
from PIL import Image
import h5py
import scipy
#from sklearn.metrics import plot_confusion_matrix

fold = int(1)
patch_num = 5
patch_size = 24
#patch_1 = [2678,1763]
#patch_1 = [262,1924]
#patch_1 = [364,2066]
#patch_1 = [843,2222]
patch_1 = [4091,2115]

model_list = [ "/home/pf/pfstud/metzgern_PF/IPA/Code/CONVODEcrop/latent_ode-my_mod_hparam/data/SwissCrops/PredictionsODEGRU/Predictions.hdf5"]


for model in model_list:
    #preddata = np.load('/home/pf/pfstaff/projects/ozgur_deep_filed/multi_stage/viz/' + model + '_' + str(fold) + '.npz')
    with h5py.File(model, "r") as preddata:

    
    
        output_file = './viz/'    
        target_test = preddata['targets'][:,:,:,0]
        pred_test = preddata['predictions2'][:,:,:,0]
        
        # Do nearest neighbour interpolation for the border effects
        startind = min(target_test.sum((1,2)).nonzero()[0])

        target_test[:, -1, 1:-1] = target_test[:, -2, 1:-1]
        target_test[:, 0, 1:-1] = target_test[:, 1, 1:-1]
        pred_test[:, -1, 1:-1]  = pred_test[:, -2, 1:-1]
        pred_test[:, 0, 1:-1] = pred_test[:, 1, 1:-1]

        target_test[:, :, 0] = target_test[:, :, 1]
        target_test[:, :, -1] = target_test[:, :, -2]
        pred_test[:, :, 0] = pred_test[:, :, 1]
        pred_test[:, :, -1] = pred_test[:, :, -2]

        #gt_list = data['arr_2']
        #gt_list_names = data['arr_3']
        #print(gt_list)
        #print(gt_list_names)
        
        #data = h5py.File("/home/pf/pfstaff/projects/ozgur_deep_filed/data_crop_CH/train_set_24x24_debug.hdf5", "r")
        rawpath = '/home/pf/pfstud/metzgern_PF/ODE_Nando/ODE_crop_Project/latent_ode-my_mod_hparam/data/SwissCrops/raw/train_set_24x24_debug.hdf5'
        with h5py.File(rawpath, "r") as data:
            
            valid_list = data['valid_list'][:]
            n_val = np.sum(valid_list)
            
            pred_test = pred_test * (target_test>0)
            
            test_shape = target_test.shape
            print('data shape: ', test_shape)
                
            target = np.zeros((n_val, test_shape[1], test_shape[2]))
            pred = np.zeros((n_val, test_shape[1], test_shape[2]))
            
            target[test_shape[0]*(fold-1):test_shape[0]*fold,...] = target_test
            pred[test_shape[0]*(fold-1):test_shape[0]*fold,...] = pred_test
            
            
            print(valid_list.shape)
            print(n_val)
            print(target_test.shape)
            print(target.shape)
            
            
            dummy = np.zeros([int(np.sum(valid_list)),test_shape[1],test_shape[2]])*255
            dummy[:pred.shape[0],:,:] = pred
            pred_map = np.zeros([valid_list.shape[0],test_shape[1],test_shape[2]])*255
            pred_map[valid_list.astype(bool)] = dummy
            
            
            dummy = np.zeros([int(np.sum(valid_list)),test_shape[1],test_shape[2]])
            dummy[:pred.shape[0],:,:] = target
            target_map = np.zeros([valid_list.shape[0],test_shape[1],test_shape[2]])
            target_map[valid_list.astype(bool)] = dummy
            
            #Reshape the maps - test ZH
            Mx = 5064//24-1
            My = 4815//24-1
            
            #Reshape the maps - test SG - TG
            #Mx = 7051//24-1
            #My = 4408//24-1
            
            num_patches = Mx*My
            print('Num patches: ', num_patches)
            
            target_map_image = np.zeros([int(Mx*test_shape[1]),int(My*test_shape[2])])
            pred_map_image = np.zeros([int(Mx*test_shape[1]),int(My*test_shape[2])])
            step = test_shape[1]
            patch_size= test_shape[1]
            count=0
            for i_y in range(0,My):
                for i_x in range(0, Mx):
                    target_map_image[int(step * i_x):int(step * i_x )+patch_size, int(step * i_y):int(step * i_y )+patch_size] = target_map[count]
                    pred_map_image[int(step * i_x):int(step * i_x )+patch_size, int(step * i_y):int(step * i_y )+patch_size] = pred_map[count]
                    count+=1
            
            performance_map_p = (target_map_image == pred_map_image) * (target_map_image != 0) * 182
            performance_map_n = (target_map_image != pred_map_image) * (target_map_image != 0) 
            
            performance_map_n = scipy.ndimage.binary_opening(performance_map_n, structure=np.ones((2,2))).astype(performance_map_n.dtype)
            performance_map_n = performance_map_n * 212
            
            #if model == 'baseline':
            if True:
                performance_map_occ = (performance_map_p+performance_map_n) != 0
                #np.save('performance_map_occ.npy',performance_map_occ)
                #performance_map_occ = np.load('./performance_map_occ.npy')
                
                
            target_map_image = target_map_image * performance_map_occ
            performance_map_n = performance_map_n * performance_map_occ
            performance_map_p = performance_map_p * performance_map_occ
            
            performance_map_e = ((performance_map_p+performance_map_n) == 0) * 255
            performance_map = np.zeros([target_map_image.shape[0],target_map_image.shape[1],3])
            
            performance_map[:,:,0] = performance_map_e
            performance_map[:,:,1] = performance_map_e
            performance_map[:,:,2] = performance_map_e
            performance_map[:,:,0] += performance_map_n
            performance_map[:,:,1] += performance_map_p

            red = [np.unique(performance_map_n)[1],0,0]
            green = [np.unique(performance_map_p)[1],0,0]
            
            pred_map_image = pred_map_image.astype(np.int8)
            target_map_image = target_map_image.astype(np.int8)
            performance_map = performance_map.astype(np.int8)
            
            ##Crop images
        #    if model == 'msConvSTAR':
        #        performance_map_occ = performance_map_occ[patch_1[0]:patch_1[0]+patch_size,patch_1[1]:patch_1[1]+patch_size]

            # Manual cropping of images going on. do not crop it here, decide later on the cropping
            #performance_map = performance_map[patch_1[0]:patch_1[0]+patch_size,patch_1[1]:patch_1[1]+patch_size,:] 
            #target_map_image = target_map_image[patch_1[0]:patch_1[0]+patch_size,patch_1[1]:patch_1[1]+patch_size] 
                
            img = Image.fromarray(np.uint8(performance_map))
            #img = img.resize((128,128), Image.NEAREST)
            img.save('./viz/patch/patch_' + str(patch_num) + '_' + '_fold_'+str(fold)+'.png')



#-----------------------------------------------------------------------------------------------------------
# change names; change colors, 

label_names = ['No Label','Maize', 'Meadow', 'Pasture', 'Potatoes', 'Spelt', 'Sugarbeets', 'Sunflowers', 'Vegetables', 'Vines',
    'Wheat', 'Winter barley', 'Winter rapeseed', 'Winter wheat']
labels = [0, 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13]
colordict = {'No Label':[255,255,255],
             'Maize': [241,97,95], 
             'Meadow': [222,138,44], 
             'Pasture':[23,244,111], 
             'Potatoes':[213,71,202], 
             'Spelt':[188,205,151],
             'Sugarbeets':[119,190,32], 
             'Sunflowers':[179,132,145], 
             'Vegetables':[52,222,187], 
             'Vines':[57,146,131], 
             'Wheat':[88,162,238], 
             'Winter barley':[217,201,55],
             'Winter rapeseed': [252,194,251], 
             'Winter wheat':[150,113,210] }

oldlabels = [0, 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 49, 50, 51]

oldlabel_names = ['Unknown', 'Apples', 'Beets', 'Berries', 'Biodiversity area', 'Buckwheat',
 'Chestnut', 'Chicory', 'Einkorn wheat', 'Fallow', 'Field bean', 'Forest',
 'Gardens', 'Grain', 'Hedge', 'Hemp', 'Hops', 'Legumes', 'Linen', 'Lupine',
 'Maize', 'Meadow', 'Mixed crop', 'Multiple', 'Mustard', 'Oat', 'Pasture', 'Pears',
 'Peas', 'Potatoes', 'Pumpkin', 'Rye', 'Sorghum', 'Soy', 'Spelt', 'Stone fruit',
 'Sugar beet', 'Summer barley', 'Summer rapeseed', 'Summer wheat', 'Sunflowers',
 'Tobacco', 'Tree crop', 'Vegetables', 'Vines', 'Wheat', 'Winter barley',
 'Winter rapeseed', 'Winter wheat']

oldcolordict = {'Unknown':[255,255,255],
             'Apples':[128,0,0], 
             'Beets': [220,20,60], 
             'Berries':[255,107,70], 
             'Biodiversity area':[0,191,255], 
             'Buckwheat':[135,206,235],
             'Chestnut':[0,0,128], 
             'Chicory':[138,43,226], 
             'Einkorn wheat':[255,105,180], 
             'Fallow':[0,255,255], 
             'Field bean':[210,105,30], 
             'Forest':[65,105,225],
             'Gardens': [255,140,0], 
             'Grain':[139,0,139], 
             'Hedge':[95,158,160], 
             'Hemp':[128,128,128], 
             'Hops':[147,112,219], 
             'Legumes': [85,107,47],
             'Linen':[176,196,222], 
             'Lupine':[127,255,212],
             'Maize': [100,149,237],
             'Meadow':[240,128,128], 
             'Mixed crop': [255,99,71] ,
             'Multiple': [220,220,220], 
             'Mustard':[0,128,128], 
             'Oat':[0,206,209], 
             'Pasture':[106,90,205] , 
             'Pears':[34,139,34],
             'Peas':[186,85,211], 
             'Potatoes':[189,183,107], 
             'Pumpkin':[205,92,92], 
             'Rye': [184,134,11], 
             'Sorghum': [0,100,0], 
             'Soy': [199,21,133], 
             'Spelt':[25,25,112], 
             'Stone fruit': [0,0,0],
             'Sugar beet':[152,251,152], 
             'Summer barley': [245,222,179],
             'Summer rapeseed': [32,178,170],
             'Summer wheat': [255,69,0],
             'Sunflowers': [0,0,255],
             'Tobacco':[238,232,170],
             'Tree crop':[255,255,102], 
             'Vegetables': [255,20,147], 
             'Vines': [255,0,0], 
             'Wheat': [255,215,0], 
             'Winter barley':[128,128,0],
             'Winter rapeseed':[154,205,50],
             'Winter wheat':[124,252,0]}

for key in colordict:
    x = colordict[key] 
    x[0]/=255
    x[1]/=255
    x[2]/=255
    colordict[key] = x

from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
fontP = FontProperties()
fontP.set_size('xx-small')

valid_labels = np.unique(target_map_image)
valid_labels = valid_labels.tolist()
valid_label_names = []
for i in valid_labels:
    valid_label_names.append(label_names[labels.index(i)])

#valid_label_names.remove('Unknown')
colors = [colordict[p] for p in valid_label_names]

legendfig, ax = plt.subplots(1, 1)
legend_elements = [Line2D([0], [0], color=color, lw=8, label=band) for band,color in dict(zip(valid_label_names, colors)).items()]
ax.legend(handles=legend_elements, ncol=3, loc="center", borderpad=1.2, handlelength=0.4, borderaxespad=0.1, prop=fontP, frameon=None)
#ax.legend(handles=legend_elements, ncol=3, loc="center", prop=fontP, frameon=None)

ax.axis("off")
legendfig.savefig("./viz/patch/legend_croped_patch_" +str(patch_num)+ ".png",dpi=600)



"""
target_map_RGB = np.ones([target_map_image.shape[0],target_map_image.shape[1],3])*255

for i_x in range(target_map_RGB.shape[0]):
    for i_y in range(target_map_RGB.shape[1]):
        target_pix_val = target_map_image[i_x,i_y]
        #pred_pix_val = target_map_image[i_x,i_y]
        if target_pix_val==0:
            continue
        
        target_pix_color = colordict[ label_names[labels.index(target_pix_val)] ]
        target_map_RGB[i_x,i_y,:] = np.array(target_pix_color)*255

target_map_image_img = Image.fromarray(np.uint8(target_map_RGB))
#target_map_image_img = target_map_image_img.resize((128,128), Image.NEAREST)
target_map_image_img.save('./viz/patch/target_patch_'+str(patch_num)+'.png')
"""




pred_map_RGB = np.ones([pred_map_image.shape[0],pred_map_image.shape[1],3])*255

for i_x in range(pred_map_RGB.shape[0]):
    for i_y in range(pred_map_RGB.shape[1]):
        pred_pix_val = pred_map_image[i_x,i_y]
        #pred_pix_val = target_map_image[i_x,i_y]
        if pred_pix_val==0:
            continue
        
        target_pix_color = colordict[ label_names[labels.index(pred_pix_val)] ]
        pred_map_RGB[i_x,i_y,:] = np.array(target_pix_color)*255

pred_map_image_img = Image.fromarray(np.uint8(pred_map_RGB))
#target_map_image_img = target_map_image_img.resize((128,128), Image.NEAREST)
pred_map_image_img.save('./viz/patch/pred_patch_'+str(patch_num)+'.png')