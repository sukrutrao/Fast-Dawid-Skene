"""
Copyright (c) 2018 Vaibhav B Sinha, Sukrut Rao, Vineeth N Balasubramanian

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import sys
import loader
import numpy as np
import pytest
import config


@pytest.fixture()
def setup():
    data_path = os.path.join(config.root_dir, 'data')
    toy_dataset_path = os.path.join(data_path, 'toy_dataset')
    toy_crowd_path = os.path.join(toy_dataset_path, 'crowd.csv')
    toy_gt_path = os.path.join(toy_dataset_path, 'gold.csv')
    all_exist = os.path.exists(toy_dataset_path) and os.path.exists(
        toy_crowd_path) and os.path.exists(toy_gt_path)
    if not all_exist:
        raise pytest.skip()


class TestLoader(object):

    def test_loader_aggregate(self, setup):

        l = loader.DataLoader('toy', 2, 'aggregate')
        data, _ = l.get_data()
        assert data == {0: {0: [0], 1: [0]}, 1: {
            0: [1], 2: [3]}, 2: {0: [2], 1: [0]}}

    def test_loader_test(self, setup):
        l = loader.DataLoader('toy', 2, 'test')
        data, gt = l.get_data()
        assert data == {0: {0: [0], 1: [0]}, 1: {
            0: [1], 2: [3]}, 2: {0: [2], 1: [0]}}
        assert np.array_equal(gt, [0, 1, 2])

    def test_loader_invalid_mode(self, setup):
        with pytest.raises(AssertionError):
            l = loader.DataLoader('toy', 2, 'mode')

    def test_loader_invalid_annotator_count(self, setup):
        with pytest.raises(AssertionError):
            l = loader.DataLoader('toy', 30, 'aggregate')

    def test_loader_invalid_k(self, setup):
        with pytest.raises(AssertionError):
            l = loader.DataLoader('toy', -1, 'aggregate')

    def test_loader_set_k(self, setup):
        l = loader.DataLoader('toy', 2, 'aggregate')
        assert l.k == 2
        assert len(l.filtered_crowd_df) == l.num_questions * l.k
        l.set_k(1)
        assert l.k == 1
        assert len(l.filtered_crowd_df) == l.num_questions * l.k
        l.set_k(0)
        assert l.k == 0
        assert len(l.filtered_crowd_df) == len(l.crowd_df)

    def test_loader_custom_dataset_path(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        copy_data_dir = os.path.join(current_dir, 'test_data')
        if not os.path.exists(copy_data_dir):
            pytest.skip("Custom dataset missing")
        l = loader.DataLoader('toy', 2, 'aggregate', data_dir=copy_data_dir)
        data, _ = l.get_data()
        assert data == {0: {0: [0], 1: [0]}, 1: {
            0: [1], 2: [3]}, 2: {0: [2], 1: [0]}}

    def test_loader_custom_file_paths(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        copy_data_dir = os.path.join(current_dir, 'test_data')
        custom_crowd_file = os.path.join(copy_data_dir, 'crowd.csv')
        custom_gt_file = os.path.join(copy_data_dir, 'gold.csv')
        if not os.path.exists(custom_crowd_file) or not os.path.exists(custom_gt_file):
            pytest.skip("Custom crowd and ground truth files missing")
        l = loader.DataLoader('toy', 2, 'test', data_dir=None,
                              crowd_annotations_path=custom_crowd_file, ground_truths_path=custom_gt_file)
        data, gt = l.get_data()
        assert data == {0: {0: [0], 1: [0]}, 1: {
            0: [1], 2: [3]}, 2: {0: [2], 1: [0]}}
        assert np.array_equal(gt, [0, 1, 2])

    def test_loader_get_dicts(self, setup):
        l = loader.DataLoader('toy', 2, 'aggregate')
        assert isinstance(l.get_ind_to_question_dict(), dict)
        assert isinstance(l.get_ind_to_annotation_dict(), dict)
