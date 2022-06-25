from radial_basis_function_extractor import RadialBasisFunctionExtractor
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from tqdm import tqdm


def plot_features(RBFE, feature_num, fig):
    axis_size = 100
    z = np.zeros([axis_size, axis_size])
    ax = fig.add_subplot(1, 2, feature_num + 1, projection='3d')
    xs = np.linspace(-2.5, 2.5, axis_size)
    ys = np.linspace(-2.5, 2.5, axis_size)
    xv, yv = np.meshgrid(xs, ys)
    for ix, x in tqdm(enumerate(xs)):
        for iy, y in enumerate(ys):
            z[ix, iy] = RBFE.encode_states_with_radial_basis_functions([[x, y]])[0][feature_num]
    ax.plot_surface(xv, yv, z, cmap=cm.coolwarm)
    ax.set_xlabel('position')
    ax.set_ylabel('velocity')
    ax.set_title('feature ' + str(feature_num + 1))


if __name__ == '__main__':
    RBFE = RadialBasisFunctionExtractor([12, 10])

    # plotting features of states
    fig = plt.figure(figsize=(13, 4))
    plot_features(RBFE, 0, fig)
    plot_features(RBFE, 1, fig)

    # saving result
    plt.savefig("./Results/Q1_features.png")
