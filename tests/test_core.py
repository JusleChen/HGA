"""Small numerical regressions; no participant data or training jobs."""
import unittest
import numpy as np
from hga import build_leadfield,minimum_norm_kernel
from hga.harmonics import sphere_quadrature,real_sh
from hga.covariant_operator import _operator_class
from hga.source import fast_real_sh,source_coefficients_real

class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.coefficients={k:np.r_[np.log(r/.1)*np.sqrt(4*np.pi),np.zeros(8)] for k,r in zip(
            ['inner_skull','outer_skull','outer_skin'],[.08,.09,.1])}
        cls.sources=np.array([[.012,.008,.02],[-.008,.012,.018]])
        cls.electrodes=sphere_quadrature(3,4)['er']*.1

    def test_real_basis_and_vectorized_derivatives_match(self):
        q=sphere_quadrature(8,16)
        a=real_sh(q['theta'],q['phi'],3,True)
        b=fast_real_sh(q['theta'],q['phi'],3,True)
        for first,second in zip(a,b):np.testing.assert_allclose(first,second,atol=2e-14)
        np.testing.assert_allclose(a[0].T@(q['weights'][:,None]*a[0]),np.eye(16),atol=2e-14)

    def test_public_gain_layout_reference_and_kernel(self):
        gain,details=build_leadfield(self.coefficients,np.zeros(3),.1,self.sources,self.electrodes,
            geometry_order=2,field_order=2,compare_permuted_system=True)
        self.assertEqual(gain.shape,(12,6))
        self.assertGreater(np.linalg.norm(gain),0)
        np.testing.assert_allclose(gain.mean(0),0,atol=1e-12)
        self.assertLess(details['permuted_relative_error'],1e-11)
        kernel=minimum_norm_kernel(gain)
        alpha=np.trace(gain@gain.T)/len(gain)/9
        np.testing.assert_allclose(kernel,gain.T@np.linalg.inv(gain@gain.T+alpha*np.eye(len(gain))),rtol=2e-13,atol=1e-15)

    def test_public_wrapper_matches_retained_source_xyz_order(self):
        Operator,_,_,_=_operator_class()
        op=Operator(self.coefficients,2,2,np.zeros(3),100.,sphere_quadrature(24,48),scanner_to_surface=np.eye(4))
        raw,_=op.leadfield(self.sources,self.electrodes,realization='dgm',batch_size=64)
        expected=raw.transpose(1,0,2).reshape(12,6);expected-=expected.mean(0,keepdims=True)
        actual,_=build_leadfield(self.coefficients,np.zeros(3),.1,self.sources,self.electrodes,geometry_order=2,field_order=2)
        np.testing.assert_array_equal(actual,expected)

    def test_source_coefficients_linear_in_moment(self):
        source=np.array([[.01,.008,.02]])
        a=source_coefficients_real(source,np.array([1.,2.,3.]),3,.1,.33)
        b=source_coefficients_real(source,np.array([2.,4.,6.]),3,.1,.33)
        np.testing.assert_allclose(2*a,b)

if __name__=='__main__':unittest.main()
