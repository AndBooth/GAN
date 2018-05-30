import matplotlib.pyplot as plt

def plotimgs(plots):
    
    plt.figure(figsize=(10,10))  # todo: write plotting as a function
    for i in range(plots.shape[0]):
        plt.subplot(4, 4, i+1)
        image = plots[i, :, :]
        #image = np.reshape(image, [self.img_rows, self.img_cols])
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
        
    plt.savefig('plot_' + str(step) + ".png" )
    plt.close('all')