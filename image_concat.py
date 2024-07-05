import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load images
img1 = mpimg.imread('./result_plot_train_loss.png')
img2 = mpimg.imread('./result_plot_acc_train.png')
img3 = mpimg.imread('./result_plot_acc_test.png')
img4 = mpimg.imread('./result_plot_train_loss.png')
img5 = mpimg.imread('./result_plot_train_loss.png')
img6 = mpimg.imread('./result_plot_acc_test.png')

fig, axs = plt.subplots(2, 2, figsize=(25, 25),gridspec_kw={'wspace': 0, 'hspace': 0})

# Display images in the subplots
axs[0, 0].imshow(img1)
axs[0, 0].axis('off')

axs[0, 1].imshow(img2)
axs[0, 1].axis('off')

axs[1, 0].imshow(img3)
axs[1, 0].axis('off')

axs[1, 1].imshow(img4)
axs[1, 1].axis('off')

# Adjust layout
plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
# Save the figure
plt.savefig('./all_images.png')

# Show the figure
plt.show()