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

from __future__ import print_function

import numpy as np
import pandas as pd
import os
import sys
import config


class DataLoader:
    """Class to load data for use in the algorithms"""

    def __init__(self, dataset, k, mode='aggregate', data_dir=None,
                 crowd_annotations_path=None, ground_truths_path=None):
        self.dataset = dataset
        self.k = k
        self.mode = mode

        assert mode in ['aggregate', 'test'], "Invalid mode specified!"

        if data_dir is not None:
            self.data_path = data_dir
        else:
            self.data_path = os.path.join(
                config.root_dir, 'data', dataset + '_dataset')
        if crowd_annotations_path is not None:
            self.crowd_path = crowd_annotations_path
        else:
            self.crowd_path = os.path.join(self.data_path, 'crowd.csv')
        assert self.k >= 0, "Number of annotators must be a positive integer, or 0 for allowing a variable number of annotators"
        assert os.path.exists(
            self.crowd_path), self.crowd_path + " does not exist!"

        self.crowd_df = pd.read_csv(self.crowd_path, names=[
            'Annotator', 'Question', 'Annotation'])

        self.min_annotators = self.crowd_df.groupby(
            'Question')['Annotator'].count().min()

        assert self.k <= self.min_annotators, "Some data points do not have " + \
            str(self.k) + " annotators!"

        self.annotator_to_ind_dict, self.ind_to_annotator_dict = self.create_val_to_ind_dicts(
            self.crowd_df['Annotator'])
        self.question_to_ind_dict, self.ind_to_question_dict = self.create_val_to_ind_dicts(
            self.crowd_df['Question'])
        self.annotation_to_ind_dict, self.ind_to_annotation_dict = self.create_val_to_ind_dicts(
            self.crowd_df['Annotation'])
        self.crowd_df['Annotator'] = self.crowd_df[
            'Annotator'].map(self.annotator_to_ind_dict)
        self.crowd_df['Question'] = self.crowd_df[
            'Question'].map(self.question_to_ind_dict)
        self.crowd_df['Annotation'] = self.crowd_df[
            'Annotation'].map(self.annotation_to_ind_dict)
        self.num_annotators = self.crowd_df['Annotator'].nunique()
        self.num_questions = self.crowd_df['Question'].nunique()
        self.num_options = self.crowd_df['Annotation'].nunique()

        self.filter_data()

        if self.mode == 'test':
            if ground_truths_path is not None:
                self.gt_path = ground_truths_path
            else:
                self.gt_path = os.path.join(self.data_path, 'gold.csv')
            assert os.path.exists(
                self.gt_path), self.gt_path + " does not exist!"

            self.gt_df = pd.read_csv(self.gt_path, names=[
                'Question', 'Annotation'])

            self.num_questions_gt = self.gt_df['Question'].nunique()
            assert self.num_questions == self.num_questions_gt, "Mismatch in number of questions in annotations and ground truths!"

            self.gt_df['Question'] = self.gt_df[
                'Question'].map(self.question_to_ind_dict)
            assert ~self.gt_df['Question'].isnull().values.any(
            ), "Mismatch in question IDs in annotations and ground truths!"

            # TODO - assumes all possible annotations appear in crowd data
            self.gt_df['Annotation'] = self.gt_df[
                'Annotation'].map(self.annotation_to_ind_dict)
            assert ~self.gt_df['Annotation'].isnull().values.any(
            ), "Mismatch in annotation IDs in annotations and ground truths! Possible causes: a ground truth label does not appear anywhere in the crowd annotations."

    def create_val_to_ind_dicts(self, df_col):
        """
        Creates a value to index dictionary

        Creates a dictionary to map every value in a column in a dataframe to
        a unique integer index. The indices are contiguous value starting from zero

        Args:
            df_col: The column of the dataframe to be mapped 

        Returns:
            A dictionary mapping each unique value in the dataframe column to
            a unique index
        """
        unique_vals = df_col.unique()
        ind_to_val_dict = dict(enumerate(unique_vals))
        val_to_ind_dict = {val: key for key, val in ind_to_val_dict.items()}
        return val_to_ind_dict, ind_to_val_dict

    def get_ind_to_question_dict(self):
        """
        Gets the index to question dictionary

        Returns:
            The index to question dictionary
        """
        return self.ind_to_question_dict

    def get_ind_to_annotation_dict(self):
        """
        Gets the index to annotation

        Returns:
            The index to annotation dictionary
        """
        return self.ind_to_annotation_dict

    def set_k(self, k):
        """
        Sets the number of annotators to use

        Args:
            k: The number of annotators. 0 for using all available annotations

        Raises:
            AssertionError: If some questions have fewer than k annotations, or
            if k < 0.
        """
        assert k >= 0 and k <= self.min_annotators, "Invalid value specified for k!"
        self.k = k
        self.filter_data()

    def filter_data(self):
        """
        Creates a filtered dataframe with first k annotations for each question.
        Does nothing if k = 0
        """
        if self.k > 0:
            self.filtered_crowd_df = self.crowd_df.groupby(
                'Question').head(self.k)
        else:
            self.filtered_crowd_df = self.crowd_df

    def get_data(self):
        """
        Gets the data and ground truths

        This function returns the crowdsourced annotation data, and if
        available, ground truths. In 'test' mode, ground truths are returned. In
        'aggregate' mode, None is returned for ground truths.
        The data is structured as a dictionary, where each question is a key,
        and the value is a dict of the form {person: [list of annotatiosn]}.
        The ground truths is structured as a numpy array, where the ith index
        denotes the label for the ith question.

        Returns:
            Crowdsourced data and ground truths (None for ground truths in
            'aggregate' mode)
        """
        data = {}
        data_split = self.filtered_crowd_df.to_dict('split')['data']
        for data_point in data_split:
            annotator = data_point[0]
            question = data_point[1]
            annotation = data_point[2]
            if question not in data:
                data[question] = {}
            if annotator not in data[question]:
                data[question][annotator] = []
            data[question][annotator].append(annotation)
        self.data = data
        if self.mode == 'test':
            self.gt = self.gt_df['Annotation'].values
        else:
            self.gt = None
        return self.data, self.gt

if __name__ == "__main__":
    print("Data Loader")
