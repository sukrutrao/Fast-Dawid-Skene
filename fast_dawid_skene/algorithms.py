"""
Copyright (C) 2014 Dallas Card
Copyright (C) 2018 Vaibhav B Sinha, Sukrut Rao, Vineeth N Balasubramanian

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
 to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

import numpy as np


def main(args, data, gold=None):
    """
    Run the EM estimator on the data passed as the parameter

    Args:
        args: Arguments, must contain algorithm whose value should be one 
            among ['FDS','DS','H','MV']
            And should contain verbose whose value should be either True or False
        data: a dictionary object of crwod-sourced responses:
            {questions: {participants: [labels]}}
        gold: The correct label for each question: [nQuestions]

    Returns:
        result: The estimated label for each question: [nQuestions]
        acc: Accuracy of the estimated labels if gold was specified
    """

    assert args.algorithm in ['FDS', 'DS', 'H', 'MV'], 'Invalid algorithm'

    result = run(data, args=args)

    if gold is not None:
        acc = (gold == result).mean()
    else:
        acc = None

    return result, acc


def run(responses, args, tol=0.0001, CM_tol=0.005, max_iter=100):
    """
    Run the aggregator on response data

    Args:
        responses: a dictionary object of responses:
            {questions: {participants: [labels]}}
        args: Must contain algorithm whose value should be 
            one among ['FDS','DS','H','MV']
            'FDS': use for FDS algorithm
            'DS': use for original DS algorithm
            'H': use for Hybrid algorithm
            'MV': use for Majority Voting
            And should contain verbose whose value should be either True or False
        tol: threshold for class marginals for convergence of the algorithm
        CM_tol: threshold for class marginals for switching to 'hard' mode
            in Hybrid algorithm. Has no effect for FDS or DS
        max_iter: maximum number of iterations of EM

    Returns:
        The estimated label for each question: [nQuestions]
    """

    mode = args.algorithm

    # convert responses to counts
    (questions, participants, classes, counts) = responses_to_counts(responses)
    if args.verbose:
        print("Number of Questions:", len(questions))
        print("Number of Participants:", len(participants))
        print("Classes:", classes)

    question_classes = initialize(counts, mode)

    if mode == 'MV':
        return np.argmax(question_classes, axis=1)

    # initialize
    nIter = 0
    converged = False
    old_class_marginals = None
    old_error_rates = None
    # total_time = 0

    if args.verbose:
        print "Iter\tlog-likelihood\tdelta-CM\tdelta-ER"

    while not converged:
        nIter += 1

        # Start measuring time
        # start = time.time()

        # M-step
        (class_marginals, error_rates) = m_step(counts, question_classes)

        # E-step
        question_classes = e_step(counts, class_marginals, error_rates, mode)

        # End measuring time
        # end = time.time()
        # total_time += end-start

        # check likelihood
        log_L = calc_likelihood(counts, class_marginals, error_rates)

        # check for convergence
        if old_class_marginals is not None:
            class_marginals_diff = np.sum(
                np.abs(class_marginals - old_class_marginals))
            error_rates_diff = np.sum(np.abs(error_rates - old_error_rates))
            if args.verbose:
                print nIter, '\t', log_L, '\t%.6f\t%.6f' % (class_marginals_diff, error_rates_diff)
            if (class_marginals_diff < tol) or nIter >= max_iter:
                converged = True
            elif (mode == 'H' and class_marginals_diff <= CM_tol):
                if args.verbose:
                    print "Mode changed to Hphase2"
                mode = 'Hphase2'
        else:
            if args.verbose:
                print nIter, '\t', log_L

        old_class_marginals = class_marginals
        old_error_rates = error_rates

    np.set_printoptions(precision=2, suppress=True)
    if args.verbose:
        print "Class marginals"
        print class_marginals

    result = np.argmax(question_classes, axis=1)

    return result


def responses_to_counts(responses):
    """
    Convert a matrix of annotations to count data

    Args:
        responses: dictionary of responses {questions:{participants:[responses]}}

    Returns:
        questions: list of questions
        participants: list of participants
        classes: list of possible classes (choices)
        counts: 3d array of counts: [questions x participants x classes]
    """
    questions = responses.keys()
    questions.sort()
    nQuestions = len(questions)

    # determine the participants and classes
    participants = set()
    classes = set()
    for i in questions:
        i_participants = responses[i].keys()
        for k in i_participants:
            if k not in participants:
                participants.add(k)
            ik_responses = responses[i][k]
            classes.update(ik_responses)

    classes = list(classes)
    classes.sort()
    nClasses = len(classes)

    participants = list(participants)
    participants.sort()
    nParticipants = len(participants)

    # create a 3d array to hold counts
    counts = np.zeros([nQuestions, nParticipants, nClasses])

    # convert responses to counts
    for question in questions:
        i = questions.index(question)
        for participant in responses[question].keys():
            k = participants.index(participant)
            for response in responses[question][participant]:
                j = classes.index(response)
                counts[i, k, j] += 1

    return (questions, participants, classes, counts)


def initialize(counts, mode):
    """
    Get majority voting estimates for the true classes using counts

    Args:
        counts: counts of the number of times each response was received 
            by each question from each participant: [questions x participants x classes]
        mode: One among ['FDS', 'DS', 'H', 'MV']
            'FDS', 'MV' and 'H' will give a majority voting initialization
            'DS' will give the initialization mentioned in Dawid and Skene (1979)
            'FDS': use for FDS algorithm
            'DS': use for original DS algorithm
            'H': use for Hybrid algorithm

    Returns:
        question_classes: matrix of estimates of true classes:
            [questions x responses] 
    """
    [nQuestions, nParticipants, nClasses] = np.shape(counts)
    response_sums = np.sum(counts, 1)
    question_classes = np.zeros([nQuestions, nClasses])
    if mode == 'FDS' or mode == 'MV':
        for p in range(nQuestions):
            indices = np.argwhere(response_sums[p, :] == np.max(
                response_sums[p, :])).flatten()
            question_classes[p, np.random.choice(indices)] = 1
    else:
        for p in range(nQuestions):
            question_classes[p, :] = response_sums[p, :] / \
                np.sum(response_sums[p, :], dtype=float)

    return question_classes


def m_step(counts, question_classes):
    """
    M Step for the EM algorithm

    Get estimates for the prior class probabilities (p_j) and the error
    rates (pi_jkl) using MLE with current estimates of true question classes
    See equations 2.3 and 2.4 in Dawid-Skene (1979) or equations 3 and 4 in 
    our paper (Fast Dawid-Skene: A Fast Vote Aggregation Scheme for Sentiment 
    Classification)

    Args: 
        counts: Array of how many times each response was received
            by each question from each participant: [questions x participants x classes]
        question_classes: Matrix of current assignments of questions to classes

    Returns:
        p_j: class marginals - the probability that the correct answer of a question
            is a given choice (class) [classes]
        pi_kjl: error rates - the probability of participant k labeling
            response l for a question whose correct answer is j [participants, classes, classes]
    """

    [nQuestions, nParticipants, nClasses] = np.shape(counts)

    # compute class marginals
    class_marginals = np.sum(question_classes, 0) / float(nQuestions)

    # compute error rates
    error_rates = np.zeros([nParticipants, nClasses, nClasses])
    for k in range(nParticipants):
        for j in range(nClasses):
            for l in range(nClasses):
                error_rates[k, j, l] = np.dot(
                    question_classes[:, j], counts[:, k, l])
            sum_over_responses = np.sum(error_rates[k, j, :])
            if sum_over_responses > 0:
                error_rates[k, j, :] = error_rates[
                    k, j, :] / float(sum_over_responses)

    return (class_marginals, error_rates)


def e_step(counts, class_marginals, error_rates, mode):
    """
    E (+ C) Step for the EM algorithm

    Determine the probability of each question belonging to each class,
    given current ML estimates of the parameters from the M-step. Also 
    perform the C step (along with E step (see section 3.4)) in case of FDS.
    See equation 2.5 in Dawid-Skene (1979) or equations 1 and 2 in 
    our paper (Fast Dawid Skene: A Fast Vote Aggregation Scheme for Sentiment 
    Classification)

    Args:
        counts: Array of how many times each response was received
            by each question from each participant: [questions x participants x classes]
        class_marginals: probability of a random question belonging to each class: [classes]
        error_rates: probability of participant k assigning a question whose correct 
            label is j the label l: [participants x classes x classes]
        mode: One among ['H', 'Hphase2', 'FDS', 'DS']
            'Hphase2' and 'FDS' will perform E + C step
            'DS' and 'H' will perform only the E step
            'FDS': use for FDS algorithm
            'DS': use for original DS algorithm
            'H' and 'Hphase2': use for Hybrid algorithm

    Returns:
        question_classes: Assignments of labels to questions
            [questions x classes]
    """

    [nQuestions, nParticipants, nClasses] = np.shape(counts)

    question_classes = np.zeros([nQuestions, nClasses])
    final_classes = np.zeros([nQuestions, nClasses])

    for i in range(nQuestions):
        for j in range(nClasses):
            estimate = class_marginals[j]
            estimate *= np.prod(np.power(error_rates[:,
                                                     j, :], counts[i, :, :]))

            question_classes[i, j] = estimate
        if mode == 'H' or mode == 'DS':
            question_sum = np.sum(question_classes[i, :])
            if question_sum > 0:
                question_classes[i, :] = question_classes[
                    i, :] / float(question_sum)
        else:
            indices = np.argwhere(question_classes[i, :] == np.max(
                question_classes[i, :])).flatten()
            final_classes[i, np.random.choice(indices)] = 1

    if mode == 'H' or mode == 'DS':
        return question_classes
    else:
        return final_classes


def calc_likelihood(counts, class_marginals, error_rates):
    """
    Calculate the likelihood with the current  parameters

    Calculate the likelihood given the current parameter estimates
    This should go up monotonically as EM proceeds
    See equation 2.7 in Dawid-Skene (1979)

    Args:
        counts: Array of how many times each response was received
            by each question from each participant: [questions x participants x classes]
        class_marginals: probability of a random question belonging to each class: [classes]
        error_rates: probability of participant k assigning a question whose correct 
            label is j the label l: [observers x classes x classes]

    Returns:
        Likelihood given current parameter estimates
    """

    [nPatients, nObservers, nClasses] = np.shape(counts)
    log_L = 0.0

    for i in range(nPatients):
        patient_likelihood = 0.0
        for j in range(nClasses):

            class_prior = class_marginals[j]
            patient_class_likelihood = np.prod(
                np.power(error_rates[:, j, :], counts[i, :, :]))
            patient_class_posterior = class_prior * patient_class_likelihood
            patient_likelihood += patient_class_posterior

        temp = log_L + np.log(patient_likelihood)

        if np.isnan(temp) or np.isinf(temp):
            if args.verbose:
                print i, log_L, np.log(patient_likelihood), temp
            sys.exit()

        log_L = temp

    return log_L


if __name__ == '__main__':
    print("Aggregation Algorithms")
