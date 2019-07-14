import json
import matplotlib.pyplot as plt


def parse_params(params_dir):
    '''
        Return json format config file to a dicitonary
    '''
    with open(params_dir) as fp:
        params = json.load(fp)
    return params

def plot_scores(scores):
    fig, ax = plt.subplots(1, figsize=(8, 8))
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    