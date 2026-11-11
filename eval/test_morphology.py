"""Small analytic checks for scientific metric conventions, CPU only."""
import unittest
import numpy as np
import morphology as m


class MetricTests(unittest.TestCase):
    def test_strict_threshold_and_host_many_to_many(self):
        iou=np.array([[.5,.7],[.1,.8]])
        np.testing.assert_allclose(m.counts(iou),[1.5,2,1,0])
        np.testing.assert_allclose(m.counts(iou,False),[.8,1,1,1])

    def test_spine_arc_weights_and_missed_penalty(self):
        samples=m.spine_samples({"spine":[.5,.5,5.5,.5]})
        self.assertEqual(len(samples[0]),11)
        self.assertAlmostEqual(samples[1].sum(),5)
        missed=m.coverage(samples,None)
        self.assertEqual(missed['coverage_3'],0)
        self.assertEqual(missed['gap_3'],1)

    def test_exact_raster_centres_and_longest_gap(self):
        samples=m.spine_samples({"spine":[.5,.5,5.5,.5]})
        mask=np.zeros((2,6),bool);mask[0,0]=True
        c=m.coverage(samples,mask)
        self.assertAlmostEqual(c['coverage_1'],.25)
        self.assertAlmostEqual(c['gap_1'],.75)
        self.assertAlmostEqual(c['coverage_5'],1)

    def test_horizontal_and_diagonal_graph_lengths(self):
        mask=np.zeros((9,9),bool);mask[4,2:7]=True
        self.assertAlmostEqual(m.shape_features(mask)['skeleton_length'],4)
        self.assertAlmostEqual(m.shape_features(np.eye(5,dtype=bool))['skeleton_length'],4*np.sqrt(2))

    def test_label_overlap_with_overlapping_gt(self):
        labels=np.array([[1,1,0],[0,2,2]],np.uint16)
        gt=[np.array([[1,1,0],[0,0,0]],bool),np.array([[0,1,0],[0,1,1]],bool)]
        inter,iou=m.label_overlap(gt,labels)
        np.testing.assert_array_equal(inter,[[2,0],[1,2]])
        np.testing.assert_allclose(iou,[[1,0],[.25,2/3]])

    def test_date_clusters_keep_repeated_records_together(self):
        import pandas as pd
        df=pd.DataFrame({'date':['a','a','b']})
        stats,draws=m._ratio_rows(df,[1,1,0],[1,1,1],['a','b'],np.array([[2,0],[0,2],[1,1]]))
        self.assertAlmostEqual(stats['estimate'],2/3)
        np.testing.assert_allclose(draws,[1,0,2/3])


if __name__=='__main__':unittest.main()
