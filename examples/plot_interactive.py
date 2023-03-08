import matplotlib.pyplot as plt
import numpy as np


def example_draw_ion(iters=100):

    def upd_imgdata(size) -> np.ndarray:
        return np.random.random((size, size, 3))

    plt.ion()
    fig, axs = plt.subplots(1, 2)
    artist0 = axs[0].imshow(upd_imgdata(8))
    artist1 = axs[1].imshow(upd_imgdata(2))
    # artist = plt.imshow(imgdata)

    for it in range(iters):
        print(f'iteration {it}')
        artist0.set_data(upd_imgdata(size=8))
        artist1.set_data(upd_imgdata(size=2))
        plt.draw()
        plt.pause(0.01)


if __name__ == '__main__':
    example_draw_ion()
