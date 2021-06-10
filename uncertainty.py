import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_entropy(probs):
    return np.sum(-probs*np.log(probs), axis=1)
def get_expected(L):
    return np.stack(L, axis=0).mean(axis=0)

def get_uncertainties(all_probs, all_lbls):
    for lbl in all_lbls:
        assert (lbl == all_lbls[0]).all().item(), 'labels should match in different runs'
    expected_probs = get_expected(all_probs)
    entropy_of_expected = get_entropy(expected_probs)
    expected_entropy = get_expected([get_entropy(_) for _ in all_probs])

    totalU = entropy_of_expected
    dataU = expected_entropy
    knowU = totalU - dataU

    return totalU, dataU, knowU

def plot_totalU(totalU):
    fig = plt.figure(figsize=(6,4))
    sns.distplot(totalU, bins=100)
    plt.title('Distribution of total uncertainty', fontsize=18)
    plt.grid()
    
def plot_error_rate(errors, totalU):
    errors_sorted = errors[totalU.argsort()]
    fig = plt.figure(figsize=(6,4))
    plt.plot(errors_sorted)
    plt.xlabel('Example_id', fontsize=18)
    plt.ylabel('Errors sum', fontsize=18)
    plt.title("Error rate from low to high total uncertainty", fontsize=18)
    plt.grid()
    
def plot_error_change(errors, totalU):
    errors_sorted = errors[totalU.argsort()]
    fig = plt.figure(figsize=(6,4))
    plt.plot(np.cumsum(errors_sorted))
    plt.xlabel('Example_id', fontsize=18)
    plt.ylabel('Errors sum', fontsize=18)
    plt.title("Change of total error if we go from low to high total uncertainty", fontsize=18)
    plt.grid()