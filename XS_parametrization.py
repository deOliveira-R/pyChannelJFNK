import numpy as np
import scipy.sparse as sp
import h5py
import unittest
import io


class XSContainer:
    """
    Container for the cross-sections file.
    Responsible for clean opening and closing of the HDF5 file.
    """

    def __init__(self, xs_file):
        self.xs_data = h5py.File(xs_file, "r")

    def __del__(self):
        if not self.xs_data.close():
            self.xs_data.close()

    def __getitem__(self, key):
        if not isinstance(key, str):
            key = '/'.join(map(str, key))
        return self.xs_data[key]

    def __contains__(self, key):
        if not isinstance(key, str):
            key = '/'.join(map(str, key))
        return key in self.xs_data

    # def __enter__(self):
    #     self.xs_data = open(self.file, MODE)
    #     return self
    #
    # def __exit__(self, type, value, traceback):
    #     # Exception handling here
    #     self.xs_data.close()

    def attrs(self, key):
        return self.xs_data.attrs[key]

    def get(self, key):
        if not isinstance(key, str):
            key = '/'.join(map(str, key))
        return self.xs_data[key]


class XSParametrization:
    """
    Container for the cross-sections used by the neutron transport solver
    """

    def __init__(self, xs_file):
        self.xs_data = h5py.File(xs_file, "r")
        # self.xs_data = XSContainer(xs_file)
        self.eg = self.xs_data.attrs['eg']

        self.nz = None
        self.deltaCoolant = None
        self.deltaFuel = None

    def __del__(self):
        if not self.xs_data.close():
            self.xs_data.close()

    @staticmethod
    def k2h(*key):
        if not isinstance(key, str):
            key = '/'.join(map(str, key))
        return key

    @staticmethod
    def diagonals(L):
        h, w = len(L), len(L[0])
        return [[L[h - p + q - 1][q] for q in range(max(p - h + 1, 0), min(p + 1, w))] for p in range(h + w - 1)]

    @staticmethod
    def diagonals2(L):
        shape = L.shape
        n_diags = sum(shape) - 1
        start = shape[1] - n_diags
        stop = n_diags - shape[0]
        diags_index = np.arange(start, stop + 1)
        diags = [np.diagonal(L, offset=k) for k in diags_index]
        return diags, diags_index

    @staticmethod
    def repeat(L, n):
        newL = list()
        for row in L:
            n_row = list()
            for k in row:
                n_row.extend([k] * n)
            newL.append(n_row)
        return newL

    def update(self, coolant, fuel):
        """
        Updated the deltas for coolant and fuel
        :param coolant:
        :param fuel:
        :return:
        """
        # self.deltaCoolant = np.expand_dims(coolant - self.xs_data[self.k2h('der', 'Na')].attrs['temperature'], axis=0)
        self.deltaCoolant = coolant - self.xs_data[self.k2h('der', 'Na')].attrs['temperature']
        # self.deltaFuel = np.expand_dims(fuel - self.xs_data[self.k2h('der', 'fuel')].attrs['temperature'], axis=0)
        self.deltaFuel = fuel - self.xs_data[self.k2h('der', 'fuel')].attrs['temperature']
        self.nz = self.deltaCoolant.shape[0]
        return True

    def parametrize(self, key):
        """
        Parametrized the XS found by key in the HDF5 XS file
        :param key: key to XS
        :return: parametrized XS
        """
        return (np.outer(np.ones(self.nz), self.xs_data[self.k2h('ref', key)])
                + np.outer(self.deltaCoolant, self.xs_data[self.k2h('der', 'Na', key)])
                + np.outer(self.deltaFuel, self.xs_data[self.k2h('der', 'fuel', key)])).T.ravel()

    def scat(self):
        """
        return parametrized scattering XS

        :return: scattering cross-section
        """
        key = 'SCAT'

        d1, i1 = self.diagonals2(self.xs_data[self.k2h('ref', key)])
        d2, i2 = self.diagonals2(self.xs_data[self.k2h('der', 'Na', key)])
        d3, i3 = self.diagonals2(self.xs_data[self.k2h('der', 'fuel', key)])

        return sp.diags(self.repeat(d1, self.nz), i1 * self.nz) \
                + sp.diags(self.repeat(d2, self.nz), i2 * self.nz).multiply(np.tile(self.deltaCoolant, self.eg)) \
                + sp.diags(self.repeat(d3, self.nz), i3 * self.nz).multiply(np.tile(self.deltaFuel, self.eg))

    def abs(self):
        """
        return parametrized scattering XS

        :return: absorption cross-section
        """
        key = 'ABS'
        return self.parametrize(key)

    def fis(self):
        key = 'FIS'
        return self.parametrize(key)

    def tr(self):
        key = 'TR'
        return self.parametrize(key)

    def nu(self):
        key = 'NU'
        return self.parametrize(key)

    def kappa(self):
        key = 'KAPPA'
        return 1.6e-13 * self.parametrize(key)

    def chi(self):
        key = 'CHI'
        return self.parametrize(key)

    def nuFission(self):
        return self.nu() * self.fis()

    def kappaFission(self):
        return self.kappa() * self.fis()

    def diffusion(self):
        """
        Calculate the diffusion coefficient from the transport-corrected cross-section
        :return: diffusion coefficients
        """
        return (1/(3*self.tr()))


class MemIOTest(unittest.TestCase):

    def setUp(self):
        self.mem_file = io.BytesIO()

        self.data = h5py.File(self.mem_file, 'w')
        self.test_range = 5
        self.data['test'] = range(self.test_range)

    def tearDown(self):
        self.data.close()

    def test_access(self):
        self.assertCountEqual(self.data['test'], range(self.test_range))
        self.assertSequenceEqual(self.data['test'], range(self.test_range))
    #     self.geometry = {'pin_pitch': 9.8E-3,
    #                      'De': 0.003958735792072682,
    #                      'Rco': 0.00425}
    #     self.v = np.array([6, 6.5, 7, 7.5], dtype='float32')
    #     self.T = np.array([600, 650, 700, 750], dtype='float32')
    #
    # def test_h_Na(self):
    #     reference_h_Na = np.array([305776.6, 301872.1, 297739.2, 293410.5])
    #     self.assertTrue(np.allclose(h_Na(self.geometry, self.v, self.T), reference_h_Na, rtol=1e-5))
    #
    # def test_fric(self):
    #     reference_fric = np.array([0.0301055, 0.028856, 0.02778773, 0.02686283])
    #     self.assertTrue(np.allclose(fric_factor(self.geometry, self.v, self.T), reference_fric, rtol=1e-5))


class ContainerTest(unittest.TestCase):

    def setUp(self):
        self.container = XSContainer('XS_data.hdf5')

    def test_getmethod(self):
        self.assertEqual(self.container.get('ref/ABS').shape, (1, 8))

    def test_attrs(self):
        self.assertEqual(self.container.attrs('eg'), 8)

    def test_getitem(self):
        self.assertEqual(self.container['ref/ABS'].shape, (1, 8))
        self.assertEqual(self.container['ref', 'ABS'].shape, (1, 8))

    def test_contains(self):
        self.assertTrue('ref/ABS' in self.container)
        self.assertTrue(('ref', 'ABS') in self.container)


class ParametrizationTest(unittest.TestCase):

    def setUp(self):
        self.para = XSParametrization('XS_data.hdf5')
        self.cool = np.arange(0, 21, 10) + 600
        self.fuel = np.arange(0, 21, 10) + 700
        self.para.update(self.cool, self.fuel)

    def test_access(self):
        self.assertEqual(self.para.eg, 8)

    def test_abs(self):
        reference_abs = np.array([0.00783955, 0.00783936, 0.00783917, 0.00439486, 0.00439474, 0.00439463,
                                  0.00271259, 0.00271261, 0.00271263, 0.00287021, 0.00287021, 0.00287020,
                                  0.00405471, 0.00405481, 0.00405491, 0.00614142, 0.00614181, 0.00614220,
                                  0.01193372, 0.01193607, 0.01193842, 0.03179103, 0.03181380, 0.03183657])
        self.assertTrue(np.allclose(self.para.abs(), reference_abs, rtol=1e-5))

    def test_scat(self):
        reference_scat = np.array([[0.0587293000000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0.0587076000000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0.0586859000000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0.0290014000000000, 0, 0, 0.113639000000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0.0289909000000000, 0, 0, 0.113594000000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0.0289804000000000, 0, 0, 0.113548000000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0.0121449000000000, 0, 0, 0.0301921000000000, 0, 0, 0.178273000000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0.0121437000000000, 0, 0, 0.0301806000000000, 0, 0, 0.178202000000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0.0121424000000000, 0, 0, 0.0301692000000000, 0, 0, 0.178130000000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0.00396476000000000, 0, 0, 0.00516753000000000, 0, 0, 0.0183402000000000, 0, 0, 0.221168000000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0.00396441000000000, 0, 0, 0.00516757000000000, 0, 0, 0.0183298000000000, 0, 0, 0.221101000000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0.00396407000000000, 0, 0, 0.00516762000000000, 0, 0, 0.0183194000000000, 0, 0, 0.221034000000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0.000741382000000000, 0, 0, 0.000958523000000000, 0, 0, 0.000240569000000000, 0, 0, 0.0148433000000000, 0, 0, 0.273500000000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0.000741323000000000, 0, 0, 0.000958521000000000, 0, 0, 0.000240492000000000, 0, 0, 0.0148365000000000, 0, 0, 0.273426000000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0.000741263000000000, 0, 0, 0.000958519000000000, 0, 0, 0.000240415000000000, 0, 0, 0.0148296000000000, 0, 0, 0.273353000000000, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0.000124451000000000, 0, 0, 0.000156701000000000, 0, 0, 5.59934000000000e-05, 0, 0, 1.17425000000000e-05, 0, 0, 0.0145687000000000, 0, 0, 0.347700000000000, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0.000124443000000000, 0, 0, 0.000156696000000000, 0, 0, 5.60010000000000e-05, 0, 0, 1.17424000000000e-05, 0, 0, 0.0145612000000000, 0, 0, 0.347627000000000, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0.000124435000000000, 0, 0, 0.000156692000000000, 0, 0, 5.60085000000000e-05, 0, 0, 1.17423000000000e-05, 0, 0, 0.0145537000000000, 0, 0, 0.347553000000000, 0, 0, 0, 0, 0, 0],
                                   [3.20551000000000e-05, 0, 0, 6.76953000000000e-05, 0, 0, 1.24092000000000e-05, 0, 0, 2.06530000000000e-06, 0, 0, 0.000150255000000000, 0, 0, 0.0153147000000000, 0, 0, 0.491491000000000, 0, 0, 0, 0, 0],
                                   [0, 3.20592000000000e-05, 0, 0, 6.77006000000000e-05, 0, 0, 1.24124000000000e-05, 0, 0, 2.06557000000000e-06, 0, 0, 0.000150280000000000, 0, 0, 0.0153073000000000, 0, 0, 0.491325000000000, 0, 0, 0, 0],
                                   [0, 0, 3.20633000000000e-05, 0, 0, 6.77059000000000e-05, 0, 0, 1.24156000000000e-05, 0, 0, 2.06583000000000e-06, 0, 0, 0.000150305000000000, 0, 0, 0.0152999000000000, 0, 0, 0.491158000000000, 0, 0, 0],
                                   [5.51733000000000e-07, 0, 0, 3.58770000000000e-07, 0, 0, 6.25062000000000e-08, 0, 0, 5.90362000000000e-09, 0, 0, 5.39524000000000e-07, 0, 0, 8.63284000000000e-07, 0, 0, 0.00297239000000000, 0, 0, 0.429601000000000, 0, 0],
                                   [0, 5.52262000000000e-07, 0, 0, 3.58961000000000e-07, 0, 0, 6.24856000000000e-08, 0, 0, 5.88603000000000e-09, 0, 0, 5.39484000000000e-07, 0, 0, 8.63411000000000e-07, 0, 0, 0.00296708000000000, 0, 0, 0.429570000000000, 0],
                                   [0, 0, 5.52790000000000e-07, 0, 0, 3.59152000000000e-07, 0, 0, 6.24650000000000e-08, 0, 0, 5.86845000000000e-09, 0, 0, 5.39445000000000e-07, 0, 0, 8.63539000000000e-07, 0, 0, 0.00296176000000000, 0, 0, 0.429539000000000]])
        self.assertTrue(np.allclose(self.para.scat().toarray(), reference_scat, rtol=1e-5))

    def test_sparse(self):
        self.assertTrue(sp.issparse(self.para.scat()))


if __name__ == '__main__':
    unittest.main()
