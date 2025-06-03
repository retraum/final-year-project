# TA
# image reading
import matplotlib.pyplot as plt
from skimage import io, restoration
from skimage.filters.rank import entropy as entr # amounts the entropy (disorder) in an image
from skimage.morphology import disk
import numpy as np
import glob
import cv2
import scipy.stats as st
from PIL import Image

#img = io.imread('D:/kuliah/TA/ML magnesium coating/images/TA-PEI/TA-PEI_0.png',as_gray=True)
#entr_img = entr(img,disk(3))
#plt.imshow(entr_img,cmap='gray')

# showing the possible thresholds
#from skimage.filters import try_all_threshold
#fig, ax = try_all_threshold(entr_img, figsize=(10,8), verbose=False)
#plt.show()

# trying otsu threshold
from skimage.filters import threshold_mean
#thresh = threshold_otsu(entr_img)

# binarization
#binary = entr_img <= thresh
#plt.imshow(binary,cmap='gray')

# printing percentage of bright pixels
#percentage = 100 * np.sum(binary==1)/(np.sum(binary==1) + np.sum(binary==0))
#print(f'percentage of bright pixels: {percentage:.2f}%')

# -----------
# deconvolution
def gkern(kernlen=21, nsig=2): # returns a 2D Gaussian kernel
    lim = kernlen//2 + (kernlen % 2)/2 
    x = np.linspace(-lim,lim,kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d,kern1d)
    return kern2d/kern2d.sum()

psf = np.ones((3,3))/9 # 3×3 matrix divided by 9 (normalizing it)
psf = gkern(5,3)

# -----------------------------------------------------------
# reading images all at once using glob

path_image = 'D:/kuliah/TA/ML magnesium coating/images/TA-PEI/original image/*'
percentages_crack = []
for file in glob.glob(path_image):
    print(file)
    img = io.imread(file,as_gray=True)
# scaling - deconvolution
    deconvolved_img, _ = restoration.unsupervised_wiener(img,psf)
# binarization
    entr_img = entr(deconvolved_img,disk(3))
    thresh = threshold_mean(entr_img)
    binary = entr_img <= thresh
    plt.axis('off')
    plt.imshow(binary,cmap='gray')
    #cv2.imshow(file,bnw_img)
# calculate area fraction
    percentage = 100 * np.sum(binary==1)/(np.sum(binary==1) + np.sum(binary==0))
    percentages_crack.append(percentage)
#    print(f'percentage of pores: {percentage:.2f}%')
#    save_img = binary.astype(np.uint8) * 255
#    cv2.imwrite(file+'_with-deconvolution.png',save_img)

print(percentages_crack)
    
#cv2.waitKey(0)
#cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# curve fitting
import pandas as pd
from scipy.optimize import curve_fit

def calculate_Ecorr(data):
    index = data[0].index(min(data[0]))
    E_corr = data[1][index]
    return (E_corr,index)

def calculate_slope(data,index):
    for i in range(index+1):
        slope_1 = -1 * (data[1][i+1] - data[1][i])/(data[0][i+1] - data[0][i])
        slope_2 = -1 * (data[1][i+2] - data[1][i+1])/(data[0][i+2] - data[0][i+1])
        if abs(slope_1 - slope_2) > 0.1:
            index_stop = i
            break
    return (slope_1,index_stop)

def remove_below_Ecorr_thresh(data,E_corr,offset):
    E_corr_upper_thresh = [x for x in data[1] if x >= E_corr + offset]
    index_list = []
    for i in range(len(data[1])):
        if data[1][i] in E_corr_upper_thresh:
            index_list.append(data[1].index(data[1][i]))
    I_corr_upper_thresh = []
    for i in range(len(index_list)):
        I_corr_upper_thresh.append(data[0][index_list[i]])
    return (E_corr_upper_thresh,I_corr_upper_thresh)

def remove_below_Ecorr(data,E_corr):
    E_corr_upper_thresh = [x for x in data[1] if x >= E_corr]
    index_list = []
    for i in range(len(data[1])):
        if data[1][i] in E_corr_upper_thresh:
            index_list.append(data[1].index(data[1][i]))
    I_corr_upper_thresh = []
    for i in range(len(index_list)):
        I_corr_upper_thresh.append(data[0][index_list[i]])
    return (E_corr_upper_thresh,I_corr_upper_thresh)

def sort_from_behind(lst):
    dummy = []
    for i in reversed(range(len(lst))):
        dummy.append(lst[i])
    return dummy

def func_linear(x,a,b):
    return a*x + b

def func_log(x,a,b,c):
    return a*np.log10(x+b)+c

def func_exp(x,a,b):
    x = np.asarray(x)
    return a*np.exp(x*b)

def list_to_exp(listname):
    list_dummy = []
    for i in range(len(listname)):
        list_dummy.append(10 ** listname[i])
    return list_dummy

def R_squared(data,prediction):
    residuals = data - prediction
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((data - np.mean(data)) ** 2)
    return 1 - (ss_res/ss_tot)

'''
which_dataset = input('which dataset you want to load? (0/1/2/4/compiled) ')
if not which_dataset.isdigit():
    path_data_TA_PEI = 'D:/kuliah/TA/ML magnesium coating/data/TA-PEI Corrosion Data/Tafel/Data and Plot/compiled/*'
else:
    path_data_TA_PEI = f'D:/kuliah/TA/ML magnesium coating/data/TA-PEI Corrosion Data/Tafel/Data and Plot/TA-{which_dataset}PEI/*'
offset = float(input('offset you want (V): '))
'''

path_data_TA_PEI = 'D:/kuliah/TA/ML magnesium coating/data/TA-PEI Corrosion Data/Tafel/Data and Plot/compiled/*'
offset = 0.05

datasets = []
datasets_above = []
datasets_above_thresh = []
tableset = np.array(['sample','E_corr','a (slope)','log(I_corr)','b (intercept)','R²'])
tableset_exp = np.array(['sample','a','b','R²'])
for file in glob.glob(path_data_TA_PEI):
    data = []
    table = []
    df = pd.read_excel(file,sheet_name='TAFEL')
    columns_list = list(df.columns)
    
#    for i in range(len(columns_list)):
#        print(f'{i+1}. {columns_list[i]}')
    
    df = df.sort_values(by=df.columns[0])
    
    E = df.iloc[:,0].tolist()
    log_I = df.iloc[:,-1].tolist()
    
#    E = E[~np.isnan(E)]
#    I = I[~np.isnan(I)]
    
#    log_I = []
#    for i in range(len(I)):
#        log_I.append(np.log(abs(I[i])))
    
    file_name = file[:-5][89:]
    
    # grouping data so they can be plotted
    data.append(log_I)
    data.append(E)
    data.append(file_name)
    data = tuple(data)
    
    datasets.append(data)
    
    # determine E_corr
    E_corr = calculate_Ecorr(data)[0]
    index = calculate_Ecorr(data)[1]
        
    # plots above E_corr + threshold only
    plots_above = remove_below_Ecorr_thresh(data,E_corr,offset)
    E_corr_above = plots_above[0]
    I_corr_above = plots_above[1]
    
    E_corr_above, I_corr_above = zip(*sorted(zip(E_corr_above,I_corr_above)))
    E_corr_above = list(E_corr_above)
    I_corr_above = list(I_corr_above)
    
    E_corr_above = sort_from_behind(E_corr_above)
    I_corr_above = sort_from_behind(I_corr_above)
    
    E_corr_above = np.array(E_corr_above)
    I_corr_above = np.array(I_corr_above)
    
    data_above = []
    data_above.append(I_corr_above)
    data_above.append(E_corr_above)
    data_above.append(file_name)
    
    datasets_above.append(data_above)
    
    '''
    # calculate slope
    slope = calculate_slope(data_above,index)[0]
    index_stop = calculate_slope(data_above,index)[1]
    '''
    
    # determine equation for the slope, y = ax + b → b = y - ax
    # data[0] = log I (x-axis), data[1] = E (y-axis)
    # popt, pcov, etc
    popt, pcov = curve_fit(func_linear,I_corr_above,E_corr_above,p0=(I_corr_above[-1],E_corr_above[-1]))
    a_linear, b_linear = popt[0], popt[1]
    a_linear_std, b_linear_std = np.sqrt(np.diag(pcov))
    
    # determine R²
    E_predicted = func_linear(I_corr_above,*popt)
    r_squared = R_squared(E_corr_above,E_predicted)
    
    # determine I_corr → x when y = E_corr
    # x = (y - b)/a
    I_corr = (E_corr - b_linear)/a_linear
    #I_corr = np.log(I_corr)
    
    # adding data to the table
    table = [file_name,E_corr,a_linear,I_corr,b_linear,r_squared]
#    table = [file_name,round(E_corr,3),round(a_linear,4),round(I_corr,3),round(b_linear,4),round(r_squared,4)]
    tableset = np.vstack((tableset,table))
    
    '''
    a = slope
    b = data_above[1][index_stop] - data_above[0][index_stop] * a
    '''
    
    # plots above E_corr only
    plots_above = remove_below_Ecorr(data,E_corr)
    E_corr_above = plots_above[0]
    I_corr_above = plots_above[1]
    
    E_corr_above, I_corr_above = zip(*sorted(zip(E_corr_above,I_corr_above)))
    E_corr_above = list(E_corr_above)
    I_corr_above = list(I_corr_above)
    
    E_corr_above = sort_from_behind(E_corr_above)
    I_corr_above = sort_from_behind(I_corr_above)
    
    data_above = []
    data_above.append(I_corr_above)
    data_above.append(E_corr_above)
    data_above.append(file_name)
    
    datasets_above_thresh.append(data_above)
    

tableset = pd.DataFrame(tableset[1:],columns=tableset[0])
tableset.insert(1,'%crack',percentages_crack)
print('\nLinear coefficients')
print(tableset)


  
# ----------------------------
# creating a plot with different legends
# create figure and axis
fig, ax = plt.subplots()

# loop through the datasets to plot them
for x,y,label in datasets:
    line, = ax.plot(x,y,label=label) # plot each dataset and get the line handle

# adding legend after all plots are created
ax.legend(title='TA',loc='lower left')

# adding titles and labels
ax.set_title('TA')
ax.set_ylabel('E (V)')
ax.set_xlabel('log I (A)')

# minimum & maximum values of axes
plt.xlim(-5,-2)   # -2   ≤ x ≤ -5
plt.ylim(-2.5,-1) # -2.5 ≤ y ≤ -1

# show the plot
plt.show()

# ---------------------------------------------------------------------------
# creating a scatter plot for each datasets 
for i in range(len(datasets)):
    plt.scatter(datasets[i][0],datasets[i][1],label=datasets[i][-1]) # arguments: x-axis, y-axis, label
                                                                     # datasets[i][0] = log I, datasets[i][1] = E
plt.legend(title='TA',loc='lower left')
plt.ylabel('E (V)')
plt.xlabel('log I (A)')

# minimum & maximum values of axes
plt.xlim(-5,-2)   # -2   ≤ x ≤ -5
plt.ylim(-2.5,-1) # -2.5 ≤ y ≤ -1

plt.show()

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# creating a plot with different legends for datasets ABOVE THE E_CORR + OFFSET
# create figure and axis
fig, ax = plt.subplots()

# loop through the datasets to plot them
for x,y,label in datasets_above:
    line, = ax.plot(x,y,label=label) # plot each dataset and get the line handle

# adding legend after all plots are created
ax.legend(title='TA',loc='upper left')

# adding titles and labels
ax.set_title('TA')
ax.set_ylabel('E (V)')
ax.set_xlabel('log I (A)')

# show the plot
plt.show()

'''
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# logarithmic plot
# create figure and axis
fig, ax = plt.subplots()

# loop through the datasets to plot them
for x,y,label in datasets_above_thresh:
    line, = ax.plot(y,x,label=label) # plot each dataset and get the line handle

# adding legend after all plots are created
ax.legend(title='TA',loc='lower right')

# adding titles and labels
ax.set_title('TA')
ax.set_xlabel('E (V)')
ax.set_ylabel('log I (A)')

# show the plot
plt.show()
'''

# -------------------------------------------------------------------
# predicted Tafel slope
def func_Tafel1(x,a,b,c): # a = slope, b = log(I corr), c = E_corr
    return a*((x-b) + np.sqrt((x-b)**2 + 0.1))/2 + c
def func_Tafel2(x,a,b,c):
    return -1*a*((x-b) + np.sqrt((x-b)**2 + 0.1))/2 + c

tableset = tableset.values.tolist()

# plotting
import itertools

plt.figure()
colors = itertools.cycle(('#941307','#057519','#08469e','#a86a0c'))

'''
for i in range(len(tableset)):
    E_corr = float(tableset[i][1])
    slope = float(tableset[i][2])
    I_corr = float(tableset[i][3])
    intercept = float(tableset[i][4])
    sample = tableset[i][0]
    
    x = np.linspace(I_corr-10,I_corr+5,10000)
    y1 = func_Tafel1(x,slope,I_corr,E_corr) # positive slope
    y2 = func_Tafel2(x,slope,I_corr,E_corr) # negative slope

    col = next(colors)    
    plt.plot(x,y1,label=sample,color=col)
    plt.plot(x,y2,color=col)
    plt.plot(I_corr,E_corr,color='black')
'''


for i in range(4):
    E_corr = float(tableset[i][2])
    slope = float(tableset[i][3])
    I_corr = float(tableset[i][4])
    intercept = float(tableset[i][5])
    sample = tableset[i][0]
    
    x_data = datasets[i][0]
    y_data = datasets[i][1]
    x = np.linspace(I_corr-1000,I_corr+5,100000)
    y1 = func_Tafel1(x,slope,I_corr,E_corr) # positive slope
    y2 = func_Tafel2(x,slope,I_corr,E_corr) # negative slope
    y_slope = func_linear(x,slope,intercept)
    
    col = next(colors)    
    plt.plot(x,y1,label=sample,color=col)
    plt.plot(x,y2,color=col)
    plt.plot(x_data,y_data,next(colors))
    plt.axhline(y=E_corr,linestyle='--',color='grey')
    plt.axvline(x=I_corr,linestyle='--',color='grey')
    plt.plot(x,y_slope,linestyle='--',color='grey')
    
    # adding labels → x-axis & y-axis
    plt.ylabel('E (V)')
    plt.xlabel('log I (A)')
    
    # minimum & maximum values of axes
    plt.xlim(-5,-2)   # -5   ≤ x ≤ -2
    plt.ylim(-2.5,-1) # -2.5 ≤ y ≤ -1
    
    plt.legend()
    plt.show()
    
conc = [0,1,2,4]

#tableset.pop(2)
crack = [x[1] for x in tableset]
E_corr = [float(x[2]) for x in tableset]
slope = [float(x[3]) for x in tableset]
I_corr = [float(x[4]) for x in tableset]

'''
print(f'concentration: {conc}')
print(f'%crack: {crack}')
print(f'E_corr: {E_corr}')
print(f'slope: {slope}')
print(f'I_corr: {I_corr}')
'''

# --------------------------------------------------
# plotting
# Concentration vs. %Crack
plt.plot(conc,crack)
plt.ylabel('Crack in SEM image (%)')
plt.xlabel('PEI Concentration (%)')
plt.show()

# Concentration vs. E_corr
plt.plot(conc,E_corr)
plt.ylabel('E_corr (V)')
plt.xlabel('PEI Concentration (%)')
plt.show()

# Concentration vs. I_corr
plt.plot(conc,I_corr)
plt.ylabel('log(I_corr) (A)')
plt.xlabel('PEI Concentration (%)')
plt.show()

# Concentration vs. slope
plt.plot(conc,slope)
plt.ylabel('Tafel slope (mV/dec)')
plt.xlabel('PEI Concentration (%)')
plt.show()