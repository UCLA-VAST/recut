import sys
sys.path.append("/home/kdmarrett/TMD")

import tmd
import tmd.view as view

pop1 = tmd.io.load_population("/home/kdmarrett/data/TME-08-validated-swcs")
print("Validated len: " + str(len(pop1.neurons)))
pop2 = tmd.io.load_population("/home/kdmarrett/data/recut-07-3A-final-swcs")
print("Recut len: " + str(len(pop2.neurons)))

phs1 = [tmd.methods.get_ph_neuron(n, neurite_type='all') for n in pop1.neurons]
phs2 = [tmd.methods.get_ph_neuron(n, neurite_type='all') for n in pop2.neurons]

# Normalize the limits
xlims, ylims = tmd.analysis.get_limits(phs1 + phs2)

# Create average images for populations
imgs1 = [tmd.analysis.get_persistence_image_data(p, xlims=xlims, ylims=ylims) for p in phs1]
IMG1 = tmd.analysis.get_average_persistence_image(phs1, xlims=xlims, ylims=ylims)

# imgs2 = [tmd.analysis.get_persistence_image_data(p, xlims=xlims, ylims=ylims) for p in phs2]
IMG2 = tmd.analysis.get_average_persistence_image(phs2, xlims=xlims, ylims=ylims)

# imgs2 = []
# failed_count = 0
# for p in phs2:
    # try:
        # pimage = tmd.analysis.get_persistence_image_data(p, xlims=xlims, ylims=ylims)
        # pimage.append(pimage)
    # except:
        # failed_count += 1
# print("Failed count: " + str(failed_count))

# You can plot the images if you want to create pretty figures
average_figure1 = view.common.plot_img_basic(IMG1, title='', xlims=xlims, ylims=ylims)
average_figure1[0].savefig("validated")
average_figure2 = view.common.plot_img_basic(IMG2, title='', xlims=xlims, ylims=ylims)
average_figure2[0].savefig("recut")

# Create the diffence between the two images
DIMG = tmd.analysis.get_image_diff_data(IMG1, IMG2) # subtracts IMG2 from IMG1

# Plot the difference between them
diff_image = view.common.plot_img_basic(DIMG, vmin=-1.0, vmax=1.0) # vmin, vmax important to see changes
diff_image[0].savefig("difference")
# Quantify the absolute distance between the two averages
# dist = np.sum(np.abs(DIMG))
