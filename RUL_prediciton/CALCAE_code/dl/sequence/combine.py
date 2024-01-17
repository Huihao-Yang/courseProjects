import matplotlib.pyplot as plt
import matplotlib.image as mpimg

if __name__ == '__main__':
    names = ['GRU', 'LSTM', 'RNN', 'MLP']
    i = 0
    for name in names:
        plt.subplot(2, 2, i + 1)
        plt.axis('off')
        image = mpimg.imread('img/CS2_35-' + name + '.png')
        plt.imshow(image)
        i += 1
    plt.show()
