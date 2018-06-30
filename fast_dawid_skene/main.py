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

import argparse
import os
import loader
import algorithms
import utils
import pandas as pd


def run(args):
    l = loader.DataLoader(args.dataset, args.k, args.mode, args.dataset_path,
                          args.crowd_annotations_path, args.ground_truths_path)
    data, gt = l.get_data()
    result, accuracy = algorithms.main(args, data, gt)

    ind_to_question_dict = l.get_ind_to_question_dict()
    ind_to_annotation_dict = l.get_ind_to_annotation_dict()

    result_annotations = pd.DataFrame(data=result, columns=['Annotation'])
    result_annotations.reset_index(level=0, inplace=True)
    result_annotations = result_annotations.rename(
        columns={'index': 'Question'})

    result_annotations['Question'] = result_annotations[
        'Question'].map(ind_to_question_dict)
    result_annotations['Annotation'] = result_annotations[
        'Annotation'].map(ind_to_annotation_dict)

    if args.print_result:
        print("Predictions:")
        print(result_annotations)
        if args.mode == 'test':
            print("Accuracy:")
            print(accuracy)
    if args.output is not None:
        utils.to_csv(result, args.output,
                     ind_to_question_dict, ind_to_annotation_dict)
