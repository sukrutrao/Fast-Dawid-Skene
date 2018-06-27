import numpy as np
import pandas as pd
import os


class DataLoader:

    def __init__(self, dataset, k, mode='aggregate', data_dir=None, crowd_annotations_path=None, ground_truths_path=None):
        self.dataset = dataset
        self.k = k
        self.mode = mode

        assert mode in ['aggregate', 'test'], "Invalid mode specified!"

        if data_dir is not None:
            self.data_path = self.data_dir
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.data_path = os.path.join(
                current_dir, '..', 'data', dataset + '_dataset')
        if crowd_annotations_path is not None:
            self.crowd_path = crowd_annotations_path
        else:
            self.crowd_path = os.path.join(self.data_path, 'crowd.csv')
        print(self.crowd_path)

        assert self.k > 0, "Number of annotators must be a positive integer!"
        assert os.path.exists(
            self.crowd_path), self.crowd_path + " does not exist!"

        self.crowd_df = pd.read_csv(self.crowd_path, names=[
                                    'Annotator', 'Question', 'Annotation'])

        self.min_annotators = self.crowd_df.groupby(
            'Question')['Annotator'].count().min()
        assert self.k <= self.min_annotators, "Some data points do not have " + \
            str(self.k) + " annotators!"

        self.check_zero_indexing(
            self.crowd_df, 'Annotator', 'Crowd Annotators')
        self.check_zero_indexing(self.crowd_df, 'Question', 'Crowd Questions')
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

            self.check_zero_indexing(
                self.gt_df, 'Question', 'Ground Truth Questions')

            self.num_questions_gt = self.crowd_df['Question'].nunique()
            assert self.num_questions == self.num_questions_gt, "Mismatch in number of questions in annotations and ground truths!"

    def check_zero_indexing(self, df, column, string):
        num_unique = df[column].nunique()
        min_val = df[column].min()
        max_val = df[column].max()
        assert num_unique > 0, "Empty data for " + string + "!"
        assert min_val == 0, string + " not zero indexed!"
        assert max_val == num_unique - 1, string + " has missing values!"

    def set_k(self, k):
        assert k > 0 and k <= self.min_annotators, "Invalid value specified for k!"
        self.k = k
        self.filter_data()

    def filter_data(self):
        self.filtered_crowd_df = self.crowd_df.groupby('Question').head(self.k)

    def get_data(self):
        data = {}
        print(self.crowd_df)
        data_split = self.filtered_crowd_df.to_dict('split')['data']
        for data_point in data_split:
            print(data_point)
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
            return self.data, self.gt
        return self.data

if __name__ == "__main__":
    print("Data Loader")
    s = DataLoader('toy', 2, 'test')
    data, gt = s.get_data()
    print(data)
    print(gt)
