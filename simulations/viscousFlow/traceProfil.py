import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import thermo
import constants
from scipy.optimize import curve_fit
font = {'family': 'times',
        'size': 20}

matplotlib.rc('font', **font)

#common data
H = 0.1 #m

def model(y,vmax):
    return 4*vmax/(H)**2* y * ((H)-y)

def eta0(l,T,Ns):
    """
    l: mean free path
    T: temperature
    Ns: Nb of particles
    :return:viscosity provided by boltzmann theory
    """
    sigma = H**2 / (2*2**0.5 * l * Ns)
    nu = Ns * np.pi * (sigma/2)**2 / H**2

    kbs = thermo.getKbSimu(1e5,H**2,T,Ns)
    ms = thermo.getMSimu(constants.MASS,constants.Kb,kbs)
    g2 = (1-7*nu)/(1-nu)**2
    etaB = 1/(2*sigma) * (ms*kbs*T/np.pi)**0.5
    return etaB #* (1/g2 + 2*nu + (1+8/np.pi) * g2 * nu**2)

def getEta(vmax,T):
    rhog = constants.MASS * 1e5 / (constants.Kb*T) * 9.81*300
    return rhog * (2*H) ** 2 / (8 * vmax)

"""
l=1e-3 m : First profile
"""
T1 = np.average([335,333,332,330.5])
l1 = 1e-3 #m
vxs1 = []
#16k
vxs1.append(np.array([5.06759524719018, 13.299070435203038, 20.532351151784376, 27.097731673555582, 32.861767109923, 38.128010254655045, 42.08956641646266, 45.92842706044068, 48.33841757202285, 49.39863753183256, 51.024716335177615, 52.0759521455183, ]))
#24 k
vxs1.append(np.array([5.440649006266174, 13.508474973319405, 20.871384331054898, 27.289494838109597, 33.55773902934969, 39.23271136461646, 43.76770111627445, 47.453098336795215, 50.42176583086061, 52.04367541860373, 53.40855159612982, 54.766129983458896]))
#32 k
vxs1.append(np.array([5.2509345371336344, 13.772249725130973, 21.45364663574097, 27.87103563538816, 34.12254992142962, 38.83642182601999, 42.87671955857324, 47.62980262927401, 50.998734929006474, 53.36465511035571, 55.17121969273996, 56.03764895290337 ]))
# 48 k
vxs1.append(np.array([5.612117126300185, 14.741191520167733, 22.426002025220978, 28.625628874032664, 34.42578526269864, 40.03457796373666, 45.17452457825685, 49.384983869336864, 53.0424735292835, 55.628357088414155, 56.98783453408383, 57.914286138765526]))
#fill ratio s_parts/S :  2.04531e-02


"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
l :  1.42857e-03  m
"""
l2 = 1.4286e-3 #m
vxs2 = []
T2 = np.average([326.3,324.5,326,327])
#16k
vxs2.append(np.array([ 3.685546121455493, 9.62323251374001, 15.342778665112005, 19.762055047084797, 24.054340531591595, 26.98575690160862, 29.5642743600221, 32.49837679721827, 35.06890719571138, 36.950523290358625, 38.451510673829105, 38.91843181018549]))
#24k
vxs2.append(np.array([ 4.687097500301047, 11.119346960835092, 16.73237410634662, 21.750099580478977, 25.81509607126482, 29.310524909472523, 32.651237833187416, 35.634325212910575, 37.544379661013615, 38.67223036123973, 39.51463508450285, 40.465031667202354 ]))
#36k
vxs2.append(np.array([4.31893721225572,10.930409683190225,16.589903189858575,21.485671724989693,25.674308639421294,29.602415592814097,32.85811906469573,35.342325429296636,37.290101727220375,39.1087917083168,40.264683751507704,40.835846864385964]))
#48k
vxs2.append(np.array([4.128814339119911,10.523050811483493,16.024525010659403,21.261818384110175,25.798605262434204,29.694415950459117,32.9829781078689,35.61786234611779,37.83489933462678,39.469288768401206,40.30124242732161,40.803621745505055]))

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
l :  2.00000e-03  m
"""
l3 = 2.0e-3
T3 =np.average([322,321.5,323,323])
vxs3 =[]

# pour 16k particules
vxs3.append(np.array([3.499587042289428,8.18405908889523,11.986556666245628,15.358845748947942,18.61744927246459,21.25592867305659,23.384328233106128,25.281880868063958,26.843220858747966,27.898913215905864,28.610023023402114,29.401668348928933]))
# pour 32k particules
vxs3.append(np.array([3.251913134842149,7.8290580384687685,11.683914645009295,15.443426478921191,18.69262609424939,21.563333359748682,23.973239730384364,25.936383631807058,27.36690407448724,28.369567390697018,29.002044380785765,29.414725366827888]))

# pour 24k particules
vxs3.append(np.array([3.353240715889109,7.560899978055091,11.439664637804892,15.115668108964874,18.136407139201292,20.95947332630446,23.127380071383563,25.160251556650067,26.836001746518043,28.211571443386283,28.97696866214627,29.252088755565282]))
# pour 48k particules
vxs3.append(np.array([3.3129024145216275,7.9658212109871025,12.028265273743946,15.598674279648973,18.74970983509905,21.73858752002431,24.085236885177476,26.161808306421005,27.84042720220259,28.980415354977854,29.79571467574388,30.30482910917563]))



def addProfileToPlot(vxs,T,l,color):
    vx = np.array([0.]*len(vxs[0]))
    uvx = np.array([0.] * len(vxs[0]))
    nb = 0
    nbv = len(vx)
    ys = np.array([H*(i+0.5) / nbv/2 for i in range(nbv)])

    vmaxs = []
    for vs in vxs:
        popt, pcov = curve_fit(model, ys, vs, p0=[53.])
        vmaxs.append(popt[0])
        vx+= vs
        uvx += vs**2
        nb+=1
    vx/=nb
    uvx = np.sqrt(uvx /nb - vx**2) / np.sqrt(nb) # uncertainty


    plt.errorbar(vx, ys,xerr=uvx, color=color, label="$l$="+str(l*1e3)[:4]+" mm",linestyle='',capsize=4, elinewidth=2)
    popt, pcov = curve_fit(model, ys, vx, p0=[53.])
    vmax = popt[0] # vmax pour la moyenne


    plt.plot(model(ys, vmax), ys, "-", color=color, linewidth=2)

    Ns=48e3 # Ns usefull only if advanced model is used to compute the viscosity
    etas = [getEta(v,T) for v in vmaxs]
    print("eta simu/modèle", str(np.average(etas))+" +/-"+ str(np.std(etas)/np.sqrt(len(etas))), eta0(l, T, Ns))



addProfileToPlot(vxs1,T1,l1,"k")
addProfileToPlot(vxs2,T2,l2,"b")
addProfileToPlot(vxs3,T3,l3,"r")

plt.grid()
plt.legend()
plt.ylabel("$y$ [m]")
plt.xlabel("$v(y)$ [m/s]")
plt.show()