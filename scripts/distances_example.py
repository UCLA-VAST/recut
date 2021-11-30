import sys
sys.path.append("/home/kdmarrett/TMD")

import tmd
import tmd.view as view
import numpy as np

validated = tmd.io.load_population("/home/kdmarrett/data/7-3A-validated")
print("Validated len: " + str(len(validated.neurons)))
recut = tmd.io.load_population("/home/kdmarrett/data/recut-07-3A-final-swcs")
print("Recut len: " + str(len(recut.neurons)))
app2 = tmd.io.load_population("/home/kdmarrett/data/7-3A-app2")
print("APP2 len: " + str(len(app2.neurons)))

phs_validated = [tmd.methods.get_ph_neuron(n, neurite_type='all') for n in validated.neurons]
phs_recut = [tmd.methods.get_ph_neuron(n, neurite_type='all') for n in recut.neurons]
phs_recut = phs_recut[:len(phs_validated)]
phs_app2 = [tmd.methods.get_ph_neuron(n, neurite_type='all') for n in app2.neurons]

# Normalize the limits
xlims, ylims = tmd.analysis.get_limits(phs_validated + phs_recut + phs_app2)

# Create average images for populations
# imgs1 = [tmd.analysis.get_persistence_image_data(p, xlims=xlims, ylims=ylims) for p in phs_validated]
img_validated = tmd.analysis.get_average_persistence_image(phs_validated, xlims=xlims, ylims=ylims)

# imgs2 = [tmd.analysis.get_persistence_image_data(p, xlims=xlims, ylims=ylims) for p in phs_recut]
img_recut = tmd.analysis.get_average_persistence_image(phs_recut, xlims=xlims, ylims=ylims)

img_app2 = tmd.analysis.get_average_persistence_image(phs_app2, xlims=xlims, ylims=ylims)

# You can plot the images if you want to create pretty figures
average_figure1 = view.common.plot_img_basic(img_validated, title='Validated', xlims=xlims, ylims=ylims,xlabel='',ylabel='', tight_layout=True)
average_figure1[0].set_tight_layout(True)
average_figure1[0].savefig("validated")

average_figure2 = view.common.plot_img_basic(img_recut, title='Recut', xlims=xlims, ylims=ylims,xlabel='', ylabel='', tight_layout=True)
average_figure2[0].set_tight_layout(True);
average_figure2[0].savefig("recut")

app_fig = view.common.plot_img_basic(img_app2, title='APP2', xlims=xlims, ylims=ylims,xlabel='', ylabel='', tight_layout=True)
app_fig[0].set_tight_layout(True);
app_fig[0].savefig("app2")

# Create the diffence between the two images
recut_v_validated = tmd.analysis.get_image_diff_data(img_validated, img_recut) # subtracts img_recut from img_validated
# Plot the difference between them
diff_image = view.common.plot_img_basic(recut_v_validated, title='Recut vs. Validated', vmin=-1.0, vmax=1.0,xlabel='', ylabel='', tight_layout=True) # vmin, vmax important to see changes
diff_image[0].set_tight_layout(True);
diff_image[0].savefig("recut_v_validated")
# Quantify the absolute distance between the two averages
dist = np.sum(np.abs(recut_v_validated))
print("Recut v validated dist: " + str(dist))

# Create the diffence between the two images
app2_v_validated = tmd.analysis.get_image_diff_data(img_validated, img_app2) 
# Plot the difference between them
diff_image = view.common.plot_img_basic(app2_v_validated, title='APP2 vs. Validated', vmin=-1.0, vmax=1.0,xlabel='', ylabel='', tight_layout=True) # vmin, vmax important to see changes
diff_image[0].set_tight_layout(True);
diff_image[0].savefig("app2_v_validated")
# Quantify the absolute distance between the two averages
dist = np.sum(np.abs(app2_v_validated))
print("APP2 v validated Dist: " + str(dist))

# repl hook:
# import pdb; pdb.set_trace()
