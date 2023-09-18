# %% CHECK NETWORK STRUCTURE AND PROPERTIES 
import os

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np

plt.rc('font', **{'family': 'sans-serif'})

# Load Erdos Renyi network 
flag = "ER"
layer1 = ig.Graph.Read_GraphML('Networks/layer1_' + str(flag)+'')
layer2 = ig.Graph.Read_GraphML('Networks/layer2_' + str(flag))

# Load SF network 
flag = "SF"
layer1_sf = ig.Graph.Read_GraphML('Networks/layer1_' + str(flag)+'')
layer2_sf = ig.Graph.Read_GraphML('Networks/layer2_' + str(flag))

# Load NB network 
flag = "NB"
layer1_nb = ig.Graph.Read_GraphML('Networks/layer1_' + str(flag)+'')
layer2_nb = ig.Graph.Read_GraphML('Networks/layer2_' + str(flag))

# Degree distribution plot: 
fig = plt.figure()
plt.subplot(1, 2, 1)
plt.hist(layer1.degree(), bins=list(range(0, np.max(layer1.degree()))), density=True, alpha=0.6)
plt.xlabel('Degree')
plt.xticks()
plt.yticks()
plt.suptitle('Degree distribution', fontweight='bold')
plt.title('Epidemic layer')
plt.ylabel('Probability')
plt.ylim([0, 0.25])
plt.xlim([0,50])
plt.hist(layer1_sf.degree(), bins=list(range(0, np.max(layer1_sf.degree()))), density=True, alpha=0.6)
plt.hist(layer1_nb.degree(), bins=list(range(0, np.max(layer1_nb.degree()))), density=True, alpha=0.6)

plt.subplot(1, 2, 2)
plt.title('App layer')
plt.hist(layer2.degree(), bins=list(range(0, np.max(layer2.degree()))), density=True, alpha=0.6)
plt.hist(layer2_sf.degree(), bins=list(range(0, np.max(layer2_sf.degree()))), density=True, alpha=0.6)
plt.hist(layer2_nb.degree(), bins=list(range(0, np.max(layer2_nb.degree()))), density=True, alpha=0.6)
plt.xlabel('Degree')
plt.yticks([])
plt.xlim([0,50])

fig.legend(['ER', 'SF','NB'], ncol=3, loc='upper center', frameon=False,bbox_to_anchor=(0.5, 0.03),
          fancybox=True)

# Save plot 
if not os.path.isdir("Plots"):
        os.makedirs("Plots")
plt.savefig('Plots/degree_dist_comparision.pdf', bbox_inches='tight')
plt.show()

# Log-log SF plot 
plt.figure()
plt.subplot(1, 2, 1)
hist = np.histogram(layer1.degree(), density=True, bins=list(range(np.min(layer1.degree()), np.max(layer1.degree()))))
x = [0.5 * (hist[1][1:] + hist[1][:-1]), hist[0]]
plt.loglog(x[0], x[1], 'o')
plt.xlabel('log10(k)')
plt.ylabel('log10(P(k))')
plt.title('Epidemic layer')
hist2 = np.histogram(layer1_sf.degree(), density=True,
                     bins=list(range(np.min(layer1_sf.degree()), np.max(layer1_sf.degree()))))
x2 = [0.5 * (hist2[1][1:] + hist2[1][:-1]), hist2[0]]
plt.loglog(x2[0], x2[1], 'o')
hist3 = np.histogram(layer2_nb.degree(), density=True,
                     bins=list(range(np.min(layer2_nb.degree()), np.max(layer2_nb.degree()))))
x3 = [0.5 * (hist3[1][1:] + hist3[1][:-1]), hist3[0]]
plt.loglog(x3[0], x3[1], 'o')
plt.legend(['ER', 'SF','NB'])

plt.subplot(1, 2, 2)
hist = np.histogram(layer2.degree(), density=True, bins=list(range(np.min(layer2.degree()), np.max(layer2.degree()))))
x = [0.5 * (hist[1][1:] + hist[1][:-1]), hist[0]]
plt.loglog(x[0], x[1], 'o')
plt.title('App layer')
hist2 = np.histogram(layer2_sf.degree(), density=True,
                     bins=list(range(np.min(layer2_sf.degree()), np.max(layer2_sf.degree()))))
x2 = [0.5 * (hist2[1][1:] + hist2[1][:-1]), hist2[0]]
plt.loglog(x2[0], x2[1], 'o')

hist3 = np.histogram(layer2_nb.degree(), density=True,
                     bins=list(range(np.min(layer2_nb.degree()), np.max(layer2_nb.degree()))))
x3 = [0.5 * (hist3[1][1:] + hist3[1][:-1]), hist3[0]]
plt.loglog(x3[0], x3[1], 'o')
plt.xlabel('log10(k)')
plt.tight_layout()
plt.show()
#plt.savefig('Plots/SF_plot.pdf')
# %%
