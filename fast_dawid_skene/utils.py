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

import csv
import os
import numpy as np


def to_csv(result, output_path, question_dict=None, annotation_dict=None, delimiter=','):
    output_dir = os.path.dirname(output_path)
    assert os.path.exists(output_dir), output_dir + " does not exist!"
    output_file = open(output_path, 'wb')
    output_writer = csv.writer(output_file, delimiter=delimiter)
    for index, annotation in np.ndenumerate(result):
        question = index[0]
        if question_dict is not None:
            question = question_dict[question]
        if annotation_dict is not None:
            annotation = annotation_dict[annotation]
        output_writer.writerow([question, annotation])
    output_file.close()
