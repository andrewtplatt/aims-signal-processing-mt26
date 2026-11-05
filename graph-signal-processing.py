import copy

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from pygsp import filters, graphs, plotting
from pygsp.graphs import Graph


def graph() -> Graph:
    G = graphs.Minnesota()
    print(G.coords.shape)  # coordinates are already set for nodes for visualisation
    fig, axes = plt.subplots(1, 2)
    _ = axes[0].spy(G.W, markersize=0.5)  # visualise the adjacency matrix in a spy plot
    G.plot(ax=axes[1])  # visualise the graph in 2D coordinates
    plt.show()

    G = graphs.ErdosRenyi(N=30, p=0.2)
    G.set_coordinates(kind="spring")  # set coordinates for nodes for visualisation
    fig, axes = plt.subplots(1, 2)
    _ = axes[0].spy(G.W, markersize=2)
    G.plot(ax=axes[1])
    plt.show()
    return G


def graph_laplacian(G: Graph):
    G.compute_laplacian("combinatorial")
    L = G.L
    fig, axes = plt.subplots(1, 2)
    axes[0].spy(L, markersize=5)
    axes[1].hist(L.data, bins=50, log=True)
    plt.show()

    ###
    # Q: Can you observe the difference between W and L from their spy plots?
    fig, axes = plt.subplots(1, 2)
    axes[0].spy(G.W, markersize=1)
    axes[1].spy(L, markersize=1)
    plt.show()
    ###
    # A: From their spy plots, W and L are practically indistinguishable, since L = D - W where D is diagonal
    #    and the spy plot does not show the difference between W and -W. When sufficiently zoomed in, the extra
    #    diagonal component D is visible and allows differentiating between L and W.

    ###
    # Q: Can you understand the histogram for the entries of L?
    # A: Since D[i, j] >= 0 and W[i, j] in {0, 1}, the histogram of L shows ~= 10^4 entries of -1, which means that
    #    there are ~= 10^4/2 edges in the graph (10^4 edge-ends). The histogram has bins for values 1-5, these are the values of D
    #    which show the number of vertices that have each corresponding number of connecting edges. e.g. ~100
    #    vertices have 1 connecting edge, ~1000 vertices have 2 connecting edges...
    #    Since the data is stored in a sparse matrix, the histogram is not showing the number of zero entries
    #    in the L matrix
    G.compute_laplacian("normalized")
    L_norm = G.L
    fig, axes = plt.subplots(1, 2)
    axes[0].hist(L.data, bins=50, log=True)
    axes[1].hist(L_norm.data, bins=50, log=True)
    plt.show()
    ###
    # Q: Can you understand the histogram for the entries of L_norm?
    # A: By definition L_norm = D^(-1/2)LD^(-1/2). The diagonal entries of L are being scaled down by the corresponding
    #    factor in D, such that L_norm[i, i] = 1 for i where D[i, i] > 0. Hence the spike at 1 is showing the number
    #    of nodes in the network that are not isolated (D[i, i] > 0).
    #    For the other peaks:
    #    L_norm = D^(-1/2)LD^(-1/2)
    #           = D^(-1/2)DD^(-1/2) - D^(-1/2)WD^(-1/2)
    #           = I - D^(-1/2)WD^(-1/2)
    #    L_norm[i, j] = (-D^(-1/2)WD^(-1/2))[i, j] # for i != j
    #                 = -(D[i]D[j])^(-1/2) # for W[i, j] == 1, 0 otherwise
    #    Hence the peaks less than zero are telling us the frequency of each product of D[i]D[j] for connected nodes.


def graph_signal():
    G = graphs.Ring(N=60)
    f: np.ndarray = np.random.normal(size=G.N)
    G.plot_signal(f, vertex_size=50)
    plt.show()

    ###
    # Q: Compute the Laplacian quadratic form for f . Now sort the values in f in increasing order and compute the Laplacian
    #    quadratic form again. How do the two quantities compare, and why? (Note: You can plot both the original and
    #    sorted signals on the graph and compare.)
    lqf = f @ (G.L @ f)
    f_sorted = f.copy()
    f_sorted.sort()
    lqf_sorted = f_sorted @ (G.L @ f_sorted)
    fig, axes = plt.subplots(1, 2)
    print(lqf)
    print(lqf_sorted)
    G.plot_signal(f, vertex_size=50, ax=axes[0])
    G.plot_signal(f_sorted, vertex_size=50, ax=axes[1])
    plt.show()
    ###
    # A: sorting the values for f, we get a smaller Laplacian. This means that the signal is smoother across the graph.
    #    This is because the vertices in the graph were defined in order such that W[i, i+1] = 1 and W[i, i-1] = 1 for
    #    all i. Hence sorting the values for f we get less variation in f between the neighbouring vertices, making the
    #    signal smoother.
    #
    # Q: This simple ‘sorting trick’ does not always make the signal smoother on an arbitrary graph. Can you see why?
    # A: If the vertices of the graph are defined in a different way (e.g. randomly) then the nodes may not be in the
    #    same order as their adjacency. Hence, f[i] being close to f[i+1] will not correspond to neighbouring vertices
    #    having similar values of f, and the smoothness trick will not have worked.


def fourier_basis():
    G_grid = graphs.Ring(N=60)
    G_grid.compute_fourier_basis()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    G_grid.plot_signal(
        G_grid.U[:, 2], vertex_size=50, ax=axes[0]
    )  # the 3rd eigenvector as a signal on the graph
    G_grid.set_coordinates("line1D")
    G_grid.plot_signal(
        G_grid.U[:, 0:3], ax=axes[1]
    )  # the first three eigenvectors on the real line
    fig.tight_layout()
    plt.show()
    ###
    # Q: Can you observe the equivalence between the classical Fourier basis and the Fourier basis defined on the ring graph?
    # A: the first three fourier basis functions are a constant, a cosine function that oscillates once per revolution of the
    #    ring, and a sine function that oscillates with the same frequency. This is the same as the first 3 classical fourier
    #    basis functions.

    G_grid = graphs.Grid2d(10, 10)
    G_grid.compute_fourier_basis()
    fig, axes = plt.subplots(2, 3, figsize=(10, 4))
    for i in range(2):
        for j in range(3):
            G_grid.plot_signal(
                G_grid.U[:, 3 * i + j], vertex_size=50, ax=axes[i][j]
            )  # the ith eigenvector as a signal on the graph
    fig.tight_layout()
    plt.show()

    G_minnesota = graphs.Minnesota()
    G_minnesota.compute_fourier_basis()
    fig, axes = plt.subplots(2, 3, figsize=(10, 4))
    for i in range(2):
        for j in range(3):
            G_minnesota.plot_signal(
                G_minnesota.U[:, 3 * i + j], vertex_size=5, ax=axes[i][j]
            )  # the ith eigenvector as a signal on the graph
    fig.tight_layout()
    plt.show()

    ###
    # Q: What do you observe about these eigenvectors?
    # A: The first eigenvector is constant, successive eigenvectors are changing along the graph more frequently.
    #    e.g. on the grid eigenvectors 2 and 3 change in orthogonal directions half an oscillation along the graph,
    #    whereas the 4th, 5th and 6th are oscillating more.
    #
    # Q: Compute the Laplacian quadratic form for the first 5 eigenvectors of the 2D grid graph and the Minnesota road
    #    network. How do the quantities vary, and why?
    def compute_n_eigenvector_lqfs(graph: Graph, N: int) -> np.ndarray:
        graph.compute_fourier_basis()
        graph.compute_laplacian()
        U = graph.U[:, :N]
        return np.sum(U * (graph.L @ U), axis=0)

    print(
        compute_n_eigenvector_lqfs(G_grid, 5)
    )  # [9.71445147e-18 9.78869674e-02 9.78869674e-02 1.95773935e-01 3.81966011e-01]
    print(
        compute_n_eigenvector_lqfs(G_minnesota, 5)
    )  # [1.21497223e-18 8.43734154e-04 2.07582019e-03 2.26480760e-03 3.12358291e-03]
    ###
    # A: The Laplacian quadratic form for the first eigenvector is approximately zero, and it increases monotonically
    #    for subsequent eigenvectors. This is expected because the values should equal the eigenvalues corresponding
    #    to the eigenvectors:
    #    e_i^T L e_i = e_i^T X Lambda X^T e_i
    #                = lambda_i
    print(compute_n_eigenvector_lqfs(G_grid, 5) - G_grid.e[:5])  # ~= 0
    print(compute_n_eigenvector_lqfs(G_minnesota, 5) - G_minnesota.e[:5])  # ~= 0


def graph_fourier_transform():
    communities = [50, 50, 50]
    # ’comm_density’ and ’world_density’ determines intra- and inter-community edge probability
    G = graphs.Community(
        N=150, Nc=3, comm_sizes=communities, comm_density=0.2, world_density=0.01
    )
    f = np.random.normal(size=G.N)
    G.compute_fourier_basis()
    f_hat = G.gft(f)
    limits = [np.min(f) - 0.1, np.max(f) + 0.1]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    G.plot_signal(f, vertex_size=20, ax=axes[0], limits=limits)
    axes[1].plot(G.e, np.abs(f_hat), ".-")
    plt.show()
    ###
    # Q: Can you comment on the smoothness of f on the graph, from its Fourier coefficients?
    # A: The fourier coefficients are high for high eigenvalues as well as low eigenvalues,
    #    hence f is not very smooth on the graph. If the lower eigenvalues had larger fourier
    #    coefficients than the higher eigenvalues, this would indicate that f was relatively
    #    smooth.

    fp = np.zeros(G.N)
    fp[: communities[0]] = -1 * np.ones(communities[0])
    fp[-communities[-1] :] = 1 * np.ones(communities[-1])

    ###
    # Q: Plot both f and fpart on the graph. What do you observe?
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    G.plot_signal(f, vertex_size=20, ax=axes[0], limits=limits)
    G.plot_signal(fp, vertex_size=20, ax=axes[1])
    plt.show()
    ###
    # A: fpart has grouped the vertices in the graph into 3 groups corresponding
    #    to the three clusters they are positioned in, as opposed to the random
    #    signal assignments of f.
    #
    # Q: Compute the Fourier coefficients of fpart and compare with that of f . What do you observe, and why?
    fp_hat = G.gft(fp)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(G.e, np.abs(f_hat), ".-")
    axes[1].plot(G.e, np.abs(fp_hat), ".-")
    plt.show()
    ###
    # A: Unlike f, the fourier coefficients of fpart are high for low eigenvalues and
    #    near zero for high eigenvalues. This means that fpart is a smoother function
    #    on the graph's domain, which is expected because it is constructed directly
    #    from the graph's underlying structure.
    #
    # Q: Compare the Laplacian quadratic form of f and fpart. Is it consistent with what you observe above? (Note: For a
    #    fair comparison, you should divide the Laplacian quadratic form by the squared norm of the signal.)
    lqf_f = (f @ G.L @ f) / np.dot(f, f)
    lqf_fp = fp @ G.L @ fp / np.dot(fp, fp)
    print(f"{lqf_f=}")  # ~=10.4
    print(f"{lqf_fp=}")  # ~=1.5
    ###
    # A: the (normalised) LQF of f is much bigger than the LQF of fpart. This is in agreement with what was
    #    observed from the fourier coefficients, since they both indicate that fpart is smoother across the
    #    graph than f.
    return G, f


def filtering(G: Graph, f: np.ndarray):
    f_hat = G.gft(f)
    fb_hat = copy.deepcopy(f_hat)
    fb_hat[20:] = 0  # this corresponds to an ideal brick-wall low-pass filtering
    fb = G.igft(fb_hat)

    ###
    # Q: Plot both the original signal f and the filtered signal fbrick on the graph. What do you observe?
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    G.plot(f, ax=axes[0])
    G.plot(fb, ax=axes[1])
    plt.show()
    ###
    # A: fbrick appears smoother with less variation between nearby vertices

    tau = 10
    delta = 60  # define a Kronecker delta function having value 1 on node 60 and 0 elsewhere
    g = filters.Heat(G, tau)  # the heat kernel defined via the graph spectral domain
    s = g.localize(delta)  # the heat kernel localised at node 60 in the graph
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    G.plot_signal(s, vertex_size=20, ax=axes[0])
    g.plot(ax=axes[1])
    plt.show()

    fh = g.filter(f)
    ###
    # Q: Verify that localising the heat kernel at a particular node is equivalent to applying filtering to the corresponding
    #    Kronecker delta function;
    #
    # Q: Plot the original signal f , the filtered signal fbrick by the brick-wall low-pass filter, and the filtered signal fheat by the
    #    heat kernel on the graph. How do the two filtered signals compare?
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    G.plot_signal(f, vertex_size=20, ax=axes[0])
    G.plot_signal(fb, vertex_size=20, ax=axes[1])
    G.plot_signal(fh, vertex_size=20, ax=axes[2])
    plt.show()
    ###
    # Q: What if we repeat the heat kernel based filtering with a smaller τ or bigger τ ?
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    for i, t in enumerate([1, 10, 100]):
        g = filters.Heat(G, t)
        fh = g.filter(f)
        G.plot_signal(fh, vertex_size=20, ax=axes[i])
    plt.show()


def main():
    # G = graph()
    # graph_laplacian(graphs.Minnesota())
    # graph_signal()
    # fourier_basis()
    G, f_hat = graph_fourier_transform()
    filtering(G, f_hat)


if __name__ == "__main__":
    main()
