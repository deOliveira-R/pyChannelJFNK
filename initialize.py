from math import tau, sqrt
import numpy as np
import unittest
import yaml


class CaseInput:
    YAMLTag = u'!CaseInput'

    def __init__(self, boundary_conditions, geometry, discretization, physics_parameters, xs_file, numerics):
        self.boundary_conditions = boundary_conditions
        self.geometry = geometry
        self.discretization = discretization
        self.physics_parameters = physics_parameters
        self.xs_file = xs_file
        self.numerics = numerics

    def __repr__(self):
        return "{}(boundary_conditions={}, geometry={}, discretization={})".format(
            self.__class__.__name__,
            self.boundary_conditions,
            self.geometry,
            self.discretization,
        )

    def to_dict(self):
        return dict(boundary_conditions=self.boundary_conditions,
                    geometry=self.geometry,
                    discretization=self.discretization,
                    physics_parameters=self.physics_parameters,
                    xs_file=self.xs_file,
                    numerics=self.numerics)

    @staticmethod
    def to_yaml(dumper, data):
        # return dumper.represent_mapping(data.YAMLTag, data.to_dict())
        return dumper.represent_mapping(data.YAMLTag, data.to_dict())

    @staticmethod
    def from_yaml(loader, node):
        # value = CaseDefinition()
        # yield value
        # node_map = loader.construct_mapping(node, deep=True)
        # value.update(**node_map)

        node_map = loader.construct_mapping(node, deep=True)
        return CaseInput(boundary_conditions=node_map['boundary_conditions'],
                         geometry=node_map['geometry'],
                         discretization=node_map['discretization'],
                         physics_parameters=node_map['physics_parameters'],
                         xs_file=node_map['xs_file'],
                         numerics=node_map['numerics'])


yaml.add_representer(CaseInput, CaseInput.to_yaml, Dumper=yaml.SafeDumper)
yaml.add_constructor(CaseInput.YAMLTag, CaseInput.from_yaml, Loader=yaml.SafeLoader)


class CaseDefinition:

    def __init__(self, case_input):
        self.input = case_input

        # Duplicate entries in order to add keys without altering original input
        self.boundary_conditions = dict(case_input.boundary_conditions)
        self.geometry = dict(case_input.geometry)
        self.discretization = dict(case_input.discretization)
        self.physics_parameters = dict(case_input.physics_parameters)

        # Get XS library file name
        self.xs_file = case_input.xs_file

        # Get solver settings
        self.numerics = case_input.numerics

        # Calculate additional parameters and add keys
        self.calculate_geometry()
        self.discretize()

    def calculate_geometry(self):
        self.geometry['Rco'] = self.geometry['Rci'] + self.geometry['clad_thickness']
        # wetted perimeter for a triangular pin cell [m]
        self.geometry['Pw'] = (tau/2)*self.geometry['Rco']
        # Flow area [m^2]
        self.geometry['A'] = sqrt(3)/4*self.geometry['pin_pitch']**2 - (tau*self.geometry['Rco']**2/4)
        # Equivalent hydraulic diameter [m]
        self.geometry['De'] = 4*self.geometry['A']/self.geometry['Pw']

    def discretize(self):
        self.discretization['Dz'] = self.geometry['H']/self.discretization['axial_nodes']
        self.discretization['R'] = np.zeros(shape=2*self.discretization['radial_nodes_pin'] + 4)

        for x, R in enumerate(self.discretization['R']):
            if x == 0:
                self.discretization['R'][x] = sqrt(
                    self.geometry['Rfo']**2/(2*self.discretization['radial_nodes_pin']))
            else:
                self.discretization['R'][x] = sqrt(self.geometry['Rfo']**2/(2*self.discretization['radial_nodes_pin'])
                                                   + self.discretization['R'][x - 1]**2)

        self.discretization['R'][self.discretization['radial_nodes_pin']*2] = sqrt(
            (self.geometry['Rci']**2 - self.geometry['Rfo']**2)/2 + self.discretization['R'][
                self.discretization['radial_nodes_pin']*2 - 1]**2)
        self.discretization['R'][self.discretization['radial_nodes_pin']*2 + 1] = sqrt(
            (self.geometry['Rci']**2 - self.geometry['Rfo']**2)/2 + self.discretization['R'][
                self.discretization['radial_nodes_pin']*2]**2)
        self.discretization['R'][self.discretization['radial_nodes_pin']*2 + 2] = sqrt(
            (self.geometry['Rco']**2 - self.geometry['Rci']**2)/2 + self.discretization['R'][
                self.discretization['radial_nodes_pin']*2 + 1]**2)
        self.discretization['R'][self.discretization['radial_nodes_pin']*2 + 3] = sqrt(
            (self.geometry['Rco']**2 - self.geometry['Rci']**2)/2 + self.discretization['R'][
                self.discretization['radial_nodes_pin']*2 + 2]**2)

        self.discretization['DR'] = np.concatenate([[0], self.discretization['R'][1:] - self.discretization['R'][:-1]])
        self.discretization['SR'] = tau*self.discretization['R']
        self.discretization['VR'] = np.concatenate([[(tau*self.discretization['R'][0]**2)/2],
                                                    (tau*self.discretization['R'][1:]**2)/2\
                                                    - (tau*self.discretization['R'][:-1]**2)/2])


class InputTest(unittest.TestCase):

    def setUp(self):
        self.input = yaml.safe_load(
            """
            !CaseInput
            boundary_conditions:
              p_out: 1.5
              T_in: 673
              v_in: 7.5
              qp_ave: 3E4
            geometry:
              H: 1.6
              Rfo: 3.57E-3
              Rci: 3.685E-3
              clad_thickness: 5.65E-4
              pin_pitch: 9.8E-3
            discretization:
              axial_nodes: 25
              radial_nodes_pin: 5
            physics_parameters:
              g: 9.81
            xs_file: XS_data.hdf5
            numerics:
              tol_newton: 1E-8
              tol_krylov: 1E-7
            """
        )
        self.definition = CaseDefinition(self.input)

    def testTypeInput(self):
        # Test CaseInput instance
        self.assertIsInstance(self.input, CaseInput)

    def testTypeDefinition(self):
        # Test CaseDefinition instance
        self.assertIsInstance(self.definition, CaseDefinition)

    def testDifferent(self):
        # Test that self.definition.input is a different object than self.input
        self.assertIsNot(self.definition.discretization, self.input.discretization)

    def testContains(self):
        # Test that keys in self.input exist in self.definition
        self.assertTrue(all(key in list(self.definition.discretization.keys())
                            for key in list(self.input.discretization.keys())))

    def testRco(self):
        self.assertAlmostEqual(self.definition.geometry['Rco'], 0.00425)

    def testPw(self):
        self.assertAlmostEqual(self.definition.geometry['Pw'], 0.0133518)

    def testA(self):
        self.assertAlmostEqual(self.definition.geometry['A'], 1.32140e-5)

    def testDe(self):
        self.assertAlmostEqual(self.definition.geometry['De'], 0.0039587)

    def testDz(self):
        self.assertAlmostEqual(self.definition.discretization['Dz'], 0.064)

    def testSizeR(self):
        self.assertEqual(len(self.definition.discretization['R']), 14)

    def testSizeDR(self):
        self.assertEqual(len(self.definition.discretization['DR']), 14)

    def testSizeSR(self):
        self.assertEqual(len(self.definition.discretization['SR']), 14)

    def testSizeVR(self):
        self.assertEqual(len(self.definition.discretization['VR']), 14)

    def testValueR(self):
        # Test the value of the R array
        reference_R = np.array([0.00112893, 0.00159655, 0.00195537, 0.00225787, 0.00252437,
                                0.00276531, 0.00298688, 0.00319311, 0.0033868, 0.00357,
                                0.00362796, 0.003685, 0.00397754, 0.00425])
        self.assertTrue(np.allclose(self.definition.discretization['R'], reference_R, rtol=1e-5))

    def testValueDR(self):
        # Test the value of the DR array
        reference_DR = np.array([0.00000000e+00, 4.67619411e-04, 3.58816994e-04, 3.02496719e-04, 2.66504959e-04,
                                 2.40938900e-04, 2.21566186e-04, 2.06228777e-04, 1.93694302e-04, 1.83200626e-04,
                                 5.79556916e-05, 5.70443084e-05, 2.92544783e-04, 2.72455217e-04])
        self.assertTrue(np.allclose(self.definition.discretization['DR'], reference_DR, rtol=1e-5))

    def testValueSR(self):
        # Test the value of the SR array
        reference_SR = np.array([0.0070933, 0.01003144, 0.01228595, 0.01418659, 0.01586109,
                                 0.01737496, 0.0187671, 0.02006287, 0.02127989, 0.02243097,
                                 0.02279512, 0.02315354, 0.02499165, 0.02670354])
        self.assertTrue(np.allclose(self.definition.discretization['SR'], reference_SR, rtol=1e-5))

    def testValueVR(self):
        # Test the value of the VR array
        reference_VR = np.array([4.00392842e-06, 4.00392842e-06, 4.00392842e-06, 4.00392842e-06, 4.00392842e-06,
                                 4.00392842e-06, 4.00392842e-06, 4.00392842e-06, 4.00392842e-06, 4.00392842e-06,
                                 1.31055465e-06, 1.31055465e-06, 7.04231190e-06, 7.04231190e-06])
        self.assertTrue(np.allclose(self.definition.discretization['VR'], reference_VR, rtol=1e-5))


if __name__ == '__main__':
    unittest.main()
