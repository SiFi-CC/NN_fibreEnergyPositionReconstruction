from RootToNN import Simulation, Tensor3d
import numpy as np
import matplotlib.pyplot as plt

simulation = Simulation(
    file_name="/net/data_g4rt/projects/SiFiCC/InputforNN/SiPMNNNewGeometry/FinalDetectorVersion_RasterCoupling_OPM_38e8protons.root")

QDCs    = list()
ts      = list()
Es      = list()
ys      = list()

for idx, event in enumerate(simulation.iterate_events()):
    # load event features
    event_features = event.get_features()
    
    # make entries in tensor and saving tensor in list
    for counter, sipm_id in enumerate(event_features[2]):
        qdc = event_features[0][counter]
        t = event_features[1][counter]-np.min(event_features[1])
        if qdc <= 0:
            qdc = -1
            t   = -1
        QDCs.append(qdc)
        ts.append(t)
    
    # make entries in tensor and saving tensor in list
    for counter, fibre_id in enumerate(event_features[5]):
        E = event_features[3][counter]
        y = event_features[4][counter]
        if E <= 0:
            E = -1
            y = -1
        if y>=-50 and y<=50:
            y = (y+50)/100
        else:
            y = -1
            E = -1
        Es.append(E)
        ys.append(y)

Es_arr      = np.array(Es)
ts_arr      = np.array(ts)
QDCs_arr    = np.array(QDCs)
ys_arr      = np.array(ys)

Es.sort()
energy_norm = Es[int(len(Es)*0.9)]

QDCs.sort()
qdc_norm    = QDCs[int(len(QDCs)*0.9)]

out = open("Energy_and_QDC_norm.txt","w")
out.write("90% of Energy values are below the threshold: "+str(energy_norm)+"\n")
out.write("90% of QDC values are below the threshold: "+str(qdc_norm))
out.close()

plt.hist(Es_arr)
plt.savefig("Es.png")
plt.hist(QDCs_arr)
plt.savefig("QDCs.png")
plt.hist(ts_arr)
plt.savefig("ts.png")
plt.hist(ys_arr)
plt.savefig("ys.png")


    
    
