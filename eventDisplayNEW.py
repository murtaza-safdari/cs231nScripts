import numpy as np
from matplotlib import pyplot as plt
##from PIL import Image

idx1 = 60200 # pu
idx2 = 60300
idx3 = 60500
idx4 = 60700

idx5 = 60100 #not pu
idx6 = 60400
idx7 = 60600
idx8 = 60800

fileDir = "inputArraysVSmall/"
endlabel = 0.6
train_pixel_image = np.load(fileDir + "train_pixel_image_j0_EM.npy")
isPU              = np.load(fileDir + "train_isPU_j0_EM.npy")
pt                = np.load("inputArrays/recopts_j0_jz1_EM.npy")
#CV_pixel_image    = np.load(fileDir + "CV_pixel_image_j0_EM.npy")
#test_pixel_image  = np.load(fileDir + "test_pixel_image_j0_EM.npy")
#x = numpy.random.rand(10, 10)*255
#   , vmin=img.min(), vmax=img.max()
img1 = train_pixel_image[idx1]
img2 = train_pixel_image[idx2]
img3 = train_pixel_image[idx3]
img4 = train_pixel_image[idx4]
img5 = train_pixel_image[idx5]
img6 = train_pixel_image[idx6]
img7 = train_pixel_image[idx7]
img8 = train_pixel_image[idx8]
imgPU = np.mean(train_pixel_image[isPU == True], axis=0)
imgNotPU = np.mean(train_pixel_image[isPU == False], axis=0)
#img = img/img.max()
#   print(img)
# , interpolation='nearest'
#   plt.matshow(train_pixel_image[50000], cmap='gray')
#   plt.savefig('text.png')
#   plt.show()

#xax = [ x for x in np.arange(-endlabel,endlabel+.1,0.2) ]
#yax = [ x for x in np.arange(-endlabel,endlabel+.1,0.2) ]
xax  = [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4]
yax  = [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4]
#############################
#fig, axs = plt.subplots(nrows=2, ncols=2)
fig = plt.figure()
ax1 = fig.add_subplot(221)
im1 = ax1.matshow(img1, cmap='hot')
fig.colorbar(im1, ax=ax1)
#ax1.set_title("Event Display for a Jet (is Pile Up? {0})".format(isPU[idx1]))
ax1.set_xticklabels(['']+xax)
ax1.set_yticklabels(['']+yax)
ax1.set_xlabel("Eta")
ax1.set_ylabel("Phi")

ax2 = fig.add_subplot(222)
im2 = ax2.matshow(img2, cmap='hot')
fig.colorbar(im2, ax=ax2)
#ax2.set_title("Event Display for a Jet (is Pile Up? {0})".format(isPU[idx2]))
ax2.set_xticklabels(['']+xax)
ax2.set_yticklabels(['']+yax)
ax2.set_xlabel("Eta")
ax2.set_ylabel("Phi")

ax3 = fig.add_subplot(223)
im3 = ax3.matshow(img3, cmap='hot')
fig.colorbar(im3, ax=ax3)
#ax3.set_title("Event Display for a Jet (is Pile Up? {0})".format(isPU[idx3]))
ax3.set_xticklabels(['']+xax)
ax3.set_yticklabels(['']+yax)
ax3.set_xlabel("Eta")
ax3.set_ylabel("Phi")

ax4 = fig.add_subplot(224)
im4 = ax4.matshow(img4, cmap='hot')
fig.colorbar(im4, ax=ax4)
#ax4.set_title("Event Display for a Jet (is Pile Up? {0})".format(isPU[idx4]))
ax4.set_xticklabels(['']+xax)
ax4.set_yticklabels(['']+yax)
ax4.set_xlabel("Eta")
ax4.set_ylabel("Phi")

#fig.tight_layout()
plt.suptitle('Event Displays for Pile Up Jets')
plt.savefig("evtDispPU.png")
plt.close()
#plt.show()
###########################

#############################
#fig2, axs2 = plt.subplots(nrows=2, ncols=2)
fig2 = plt.figure()
ax5 = fig2.add_subplot(221)
im5 = ax5.matshow(img5, cmap='hot')
fig2.colorbar(im5, ax=ax5)
#ax5.set_title("Event Display for a Jet (is Pile Up? {0})".format(isPU[idx5]))
ax5.set_xticklabels(['']+xax)
ax5.set_yticklabels(['']+yax)
ax5.set_xlabel("Eta")
ax5.set_ylabel("Phi")

ax6 = fig2.add_subplot(222)
im6 = ax6.matshow(img6, cmap='hot')
fig2.colorbar(im6, ax=ax6)
#ax6.set_title("Event Display for a Jet (is Pile Up? {0})".format(isPU[idx6]))
ax6.set_xticklabels(['']+xax)
ax6.set_yticklabels(['']+yax)
ax6.set_xlabel("Eta")
ax6.set_ylabel("Phi")

ax7 = fig2.add_subplot(223)
im7 = ax7.matshow(img7, cmap='hot')
fig2.colorbar(im7, ax=ax7)
#ax7.set_title("Event Display for a Jet (is Pile Up? {0})".format(isPU[idx7]))
ax7.set_xticklabels(['']+xax)
ax7.set_yticklabels(['']+yax)
ax7.set_xlabel("Eta")
ax7.set_ylabel("Phi")

ax8 = fig2.add_subplot(224)
im8 = ax8.matshow(img8, cmap='hot')
fig2.colorbar(im8, ax=ax8)
#ax8.set_title("Event Display for a Jet (is Pile Up? {0})".format(isPU[idx8]))
ax8.set_xticklabels(['']+xax)
ax8.set_yticklabels(['']+yax)
ax8.set_xlabel("Eta")
ax8.set_ylabel("Phi")

#fig2.tight_layout()
plt.suptitle('Event Displays for NON Pile Up Jets')
plt.savefig("evtDispNotPU.png")
plt.close()
#plt.show()
###########################

###########################
fig3 = plt.figure()
axp = fig3.add_subplot(111)
imp = axp.matshow(imgPU, cmap='hot')
fig3.colorbar(imp, ax=axp)
axp.set_title("Average Event Display for a Pile Up Jet")
axp.set_xticklabels(['']+xax)
axp.set_yticklabels(['']+yax)
axp.set_xlabel("Eta")
axp.set_ylabel("Phi")
plt.savefig("meanEvtDispPU.png")
plt.close()

###########################
fig4 = plt.figure()
axnp = fig4.add_subplot(111)
imnp = axnp.matshow(imgNotPU, cmap='hot')
fig4.colorbar(imnp, ax=axnp)
axnp.set_title("Average Event Display for a NON Pile Up Jet")
axnp.set_xticklabels(['']+xax)
axnp.set_yticklabels(['']+yax)
axnp.set_xlabel("Eta")
axnp.set_ylabel("Phi")
plt.savefig("meanEvtDispNotPU.png")
plt.close()
###########################

#  range=histRange,weights=event_weight[train_idx]
hist, bin_edges = np.histogram(pt, bins=50)
fig5 = plt.figure()
axh = fig5.add_subplot(111)
plt.hist(pt, bins=100, range=(0,60), log=True)  # plt.hist passes it's arguments to np.histogram
plt.title("Histogram of Jet pT")
plt.savefig("jetPt.png")
plt.close()

# fig = plt.figure()
# ax = fig.add_subplot(221)
# ax.set_title("Event Display for a Jet (is Pile Up? {0})".format(isPU[idx1]))
# cax = ax.matshow(img1, cmap='hot')
# ax.colorbar(cax)
# ax.set_xticklabels(['']+xax)
# ax.set_yticklabels(['']+yax)
# ax.set_xlabel("Eta")
# ax.set_ylabel("Phi")

# ax.axis('image')
# plt.savefig('PU1.png')
# ##plt.show()
# #plt.clf()

# fig2 = plt.figure()
# ax2 = fig2.add_subplot(111)
# ax2.set_title("Event Display for a Jet (is Pile Up? {0})".format(isPU[idx2]))
# cax2 = ax2.matshow(img2, cmap = 'hot')
# fig2.colorbar(cax2)
# ax2.set_xticklabels(['']+xax)
# ax2.set_yticklabels(['']+yax)
# ax2.set_xlabel("Eta")
# ax2.set_ylabel("Phi")
# ax2.axis('image')
# plt.savefig('PU2.png')
# #plt.show()
# plt.clf()

# fig3 = plt.figure()
# ax3 = fig3.add_subplot(111)
# ax3.set_title("Event Display for a Jet (is Pile Up? {0})".format(isPU[idx3]))
# cax3 = ax3.matshow(img3, cmap = 'hot')
# fig3.colorbar(cax3)
# ax3.set_xticklabels(['']+xax)
# ax3.set_yticklabels(['']+yax)
# ax3.set_xlabel("Eta")
# ax3.set_ylabel("Phi")
# ax3.axis('image')
# plt.savefig('PU3.png')
# #plt.show()
# plt.clf()

# fig4 = plt.figure()
# ax4 = fig4.add_subplot(111)
# ax4.set_title("Event Display for a Jet (is Pile Up? {0})".format(isPU[idx4]))
# cax4 = ax4.matshow(img4, cmap = 'hot')
# fig4.colorbar(cax4)
# ax4.set_xticklabels(['']+xax)
# ax4.set_yticklabels(['']+yax)
# ax4.set_xlabel("Eta")
# ax4.set_ylabel("Phi")
# ax4.axis('image')
# plt.savefig('PU4.png')
# #plt.show()
# plt.clf()
# # fig = plt.figure()
# # ax = fig.add_subplot(111)
# # ax.set_title('colorMap')
# # ax.matshow(img)
# # ax.set_aspect('equal')
# # cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
# # cax.get_xaxis().set_visible(False)
# # cax.get_yaxis().set_visible(False)
# # cax.patch.set_alpha(0)
# # cax.set_frame_on(False)
# # plt.colorbar(orientation='vertical')
# # ax.savefig('text.png')
# # ax.show()

# # Creates a random image 100*100 pixels

# # Creates PIL image
# #img2 = img * 255
# #imgC = Image.fromarray(img2, 'L')
# #imgC.savefig('text2.png')
# #imgC.show()