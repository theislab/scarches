from surgeon.models import CVAE


def plot_runtimes(time: tuple):
    for before_net, after_net in network_tuples:
        before_time = before_net.run_time
        after_time = after_net.run_time


