import importlib.util
import unittest
import numpy as np

@unittest.skipUnless(importlib.util.find_spec('sklearn'),'scikit-learn required')
class DecodingTests(unittest.TestCase):
    def test_feature_layout_and_orientation_power(self):
        from hga.decoding.source_features import extract_trials
        rng=np.random.default_rng(4);x=rng.normal(size=(2,22,1125))*1e-6
        x-=x.mean(1,keepdims=True)
        kernel=rng.normal(size=(8*3,22));roi=[np.array([i]) for i in range(8)]
        values=extract_trials(x,kernel,roi)
        self.assertEqual([a.shape for a in values],[(2,264),(2,272),(2,272),(2,272)])
        rotation=np.linalg.qr(rng.normal(size=(3,3)))[0]
        rotated=np.einsum('ab,vbc->vac',rotation,kernel.reshape(8,3,22)).reshape(24,22)
        rotated_values=extract_trials(x,rotated,roi)
        np.testing.assert_allclose(values[2],rotated_values[2],atol=1e-12)

    def test_preprocessing_excludes_heldout_rows(self):
        from hga.decoding.classical import fit_blocks
        rng=np.random.default_rng(7);a=rng.normal(size=(80,64));train=np.arange(60)
        _,first=fit_blocks([a],train)
        altered=a.copy();altered[60:]+=1e5
        _,second=fit_blocks([altered],train)
        np.testing.assert_array_equal(first[0]['scale'].mean_,second[0]['scale'].mean_)
        np.testing.assert_array_equal(first[0]['pca'].components_,second[0]['pca'].components_)
        np.testing.assert_array_equal(first[0]['score_scale'].mean_,second[0]['score_scale'].mean_)

    def test_zero_weight_and_tie_selection(self):
        from hga.decoding.fusion import choose_weight,fuse
        p=np.eye(4)[np.tile(np.arange(4),6)]*.8+.05
        y=np.tile(np.arange(4),6);runs=np.repeat(np.arange(3),8)
        weight,_=choose_weight(p,p,y,runs)
        self.assertEqual(weight,0.)
        np.testing.assert_array_equal(fuse(p,p,0),p)

@unittest.skipUnless(importlib.util.find_spec('torch'),'optional deep dependencies required')
class BackboneTests(unittest.TestCase):
    def test_final_parameter_counts_and_shapes(self):
        import torch
        from hga.decoding.neural import make_decoder
        torch.set_num_threads(1)
        for name,count,shape in [('ATCNet',113732,(2,1,22,1125)),('TCFormer',127212,(2,22,1125)),('MIRepNet',5154820,(2,45,1001))]:
            with self.subTest(name=name):
                model=make_decoder(name).eval()
                self.assertEqual(sum(p.numel() for p in model.parameters()),count)
                with torch.no_grad():value=model(torch.zeros(shape))
                logits=value[1] if name=='MIRepNet' else value
                self.assertEqual(tuple(logits.shape),(2,4))
                self.assertTrue(torch.isfinite(logits).all())

if __name__=='__main__':unittest.main()
