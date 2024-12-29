import numpy as np
import matplotlib.pyplot as plt

def plot_2d_contour(ax, f, xlim, ylim, gridsize=300, zero=False): # f is an 2D function.
    x = np.linspace(xlim[0], xlim[1], gridsize)
    y = np.linspace(ylim[0], ylim[1], gridsize)
    X, Y = np.meshgrid(x, y)
    XY = np.vstack([X.ravel(), Y.ravel()]).T
    Zf = f(XY).reshape((gridsize, gridsize))
    if zero:
        return ax.contour(X, Y, Zf, 0, colors='tomato')
    else:
        return ax.contour(X, Y, Zf)

# Example usage
if __name__ == "__main__":
    def f(x):
        return x[:,0]**2 + x[:,1]**2 - 4

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_2d_contour(ax, f, (-5,5), (-5,5), 100)
    # plot_2d_contour(ax, f, (-5,5), (-5,5), 100, True)
    plt.show()
