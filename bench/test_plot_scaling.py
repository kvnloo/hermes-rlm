import importlib.util
import tempfile
from pathlib import Path
import unittest
import plot_scaling as p


def row(task='a', score=1, **kw):
    return dict(task_id=task, score=score, valid=True, size_tokens=100,
                family='f', root_model='m', arm='a', protocol={'v':1}, **kw)


class PlotTests(unittest.TestCase):
    def test_task_weight_not_repeat_weight(self):
        points=p.points_from_rows([row() for _ in range(9)]+[row('b',0)])
        self.assertEqual(points[0]['score'], .5)
        self.assertEqual(points[0]['tasks'],2)
        self.assertEqual(points,p.points_from_rows([row() for _ in range(9)]+[row('b',0)]))
        self.assertEqual(points[0]['ci95'],[0,1])

    def test_invalid_missing_not_zero(self):
        rows=[row(), row('b',None), row('c',float('nan')), dict(row('d',0),valid=False)]
        point=p.points_from_rows(rows)[0]
        self.assertEqual(point['score'],1)
        self.assertIsNone(point['ci95'])

    def test_no_cross_group_pooling(self):
        rows=[row()]
        for field,value in [('family','g'),('root_model','n'),('arm','b'),('protocol',{'v':2}),('size_tokens',200),('sub_model','s'),('cohort','pilot'),('length_tokenizer_sha256','new tokenizer')]:
            rows.append(dict(row(), **{field:value}))
        points=p.points_from_rows(rows)
        self.assertEqual(len(points),9)
        self.assertTrue(all(x['ci95'] is None for x in points))

    def test_aggregate_no_manufactured_intervals(self):
        points=p.points_from_aggregate({'schema_version':'rlm-paper-aggregate.v1', 'primary':[
            dict(row(),mean_score=.5,score_ci95=[0,1]),dict(row(),mean_score=None)]})
        self.assertEqual(len(points),1)
        self.assertIsNone(points[0]['ci95'])
        with self.assertRaises(ValueError): p.points_from_aggregate({})

    def test_duplicates_rejected(self):
        with self.assertRaises(ValueError): p.points_from_rows([row(row_key='x'), row(row_key='x')])

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'plot.png'
            p.render(p.points_from_rows([row(),row('b',0)]),path)
            self.assertGreater(path.stat().st_size,1000)


if __name__=='__main__': unittest.main()
