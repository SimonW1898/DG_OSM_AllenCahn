import numpy as np

class Quadrature:
    """ 
    Class for handling quadrature points and weights
    """
    def __init__(self, degree):
        self.degree = degree
        self.N_quad = degree**2
        self.Nquad_1D = degree
        self.points_1D, self.weights_1D = self.get_quadrature_points_1D()
        self.points, self.weights = self.get_quadrature_points()

    def get_quadrature_points_1D(self):
        """ 
        get quadrature points and weights for an 1D integral
        Output:
            xg: degree+1 (np.array) quadrature points
            wg: degree+1 (np.array) quadrature weights
        """
        # weights points for [-1,1]
        xg, wg = np.polynomial.legendre.leggauss(self.degree) 
        # transform for interval [0,1] because reference elemetn is defined like this
        xg = 0.5 * (xg + 1)
        wg = 0.5 * wg
        return xg, wg


    def get_quadrature_points(self):
        """ 
        get quadrature points and weights for a 2D integral
        Output:
            xg: Nquad x 2 (np.array) quadrature points
            wg: Nquad (np.array) quadrature weights
        """
        s = np.array(np.meshgrid(self.points_1D, self.points_1D)).reshape(2, -1)
        xg = s.T
        wg = (self.weights_1D * self.weights_1D[:, None]).ravel()
        return xg, wg
    

