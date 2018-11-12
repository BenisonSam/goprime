import multiprocessing
import sys
import time
from enum import Enum

from goprime import *
from goprime.board import Position
from goprime.game import Game
from goprime.mcts import MCTree, TreeNode
from goprime.model import GoModel


class ConsoleParams(Enum):
    N = 1
    Mode = 2
    Snapshot = 3
    BatchRange = 4
    DataSet = 5


console_options = ["-n", "-m", "-s", "-b", "-d"]
options_lookup = {
    "-n": ConsoleParams.N,
    "-m": ConsoleParams.Mode,
    "-s": ConsoleParams.Snapshot,
    "-b": ConsoleParams.BatchRange,
    "-d": ConsoleParams.DataSet
}
options_type = {
    "-n": int,
    "-m": str,
    "-s": str,
    "-b": "range",
    "-d": "path"
}


def validate_option(option, value):
    option = str(option).strip()
    value = str(value).strip()

    if options_type[option] is None:
        raise NameError("Unknown option: {0}".format(option))

    if options_type[option] == "range":
        value = tuple([int(val.strip()) for val in value.split(',')])
    elif options_type[option] == "path":
        import os
        if not os.path.exists(value):
            raise OSError("Path for option {0} doesn't exist".format(option))
    else:
        value = options_type[option](value)

    return value


def read_console_params():
    params = sys.argv
    arg_values = {
        ConsoleParams.N: 19,
        ConsoleParams.Mode: "selfplay",
        ConsoleParams.Snapshot: None,
        ConsoleParams.BatchRange: None,
        ConsoleParams.DataSet: None
    }

    i = 1
    while True:
        param = params[i].strip()

        if param in console_options and i + 1 < len(params):
            value = params[i + 1].strip()

            if value in console_options:
                raise ValueError("No value found for option {0}.".format(param))
            else:
                arg_values[options_lookup[param]] = validate_option(param, value)

        i += 2
        if i == len(params):
            break

    return arg_values


if __name__ == "__main__":

    params = read_console_params()
    print(params)
    # exit(1)

    N = int(params[ConsoleParams.N])
    W = N + 2

    mode = params[ConsoleParams.Mode]
    snapshot = params[ConsoleParams.Snapshot]

    game = Game(N)
    net = GoModel(board_size=N, load_snapshot=snapshot)

    if net.is_model_ready():

        # game.play_and_train(net, i=1, batches_per_game=2, output_stream=sys.stdout)

        if mode == "black":
            # Default action
            while True:
                game.game_io(net)
        elif mode == "white":
            while True:
                game.game_io(net=net, computer_black=True)
        elif mode == "gtp":
            game.gtp_io(net)
        elif mode == "tsbenchmark":
            t_start = time.time()

            Position.print(MCTree.tree_search(TreeNode(net=net, pos=Position.empty_position(N)),
                                              N_SIMS, W * W * [0]).pos)

            print('Tree search with %d playouts took %.3fs with %d threads; speed is %.3f playouts/thread/s' %
                  (N_SIMS, time.time() - t_start, multiprocessing.cpu_count(),
                   N_SIMS / ((time.time() - t_start) * multiprocessing.cpu_count())))
        elif mode == "tsdebug":
            Position.print(MCTree.tree_search(TreeNode(net=net, pos=Position.empty_position(N)),
                                              N_SIMS, W * W * [0], output_stream=sys.stdout).pos)
        elif mode == "selfplay":
            game.selfplay(net, games=32)
        elif mode == "replay_train":
            dataset = params[ConsoleParams.DataSet]
            batch_range = params[ConsoleParams.BatchRange]
            game.replay_train(net, dataset=dataset, batch_range=batch_range, disp=True)
        elif mode == "replay_traindist":
            dataset = params[ConsoleParams.DataSet]
            batch_range = params[ConsoleParams.BatchRange]
            game.replay_traindist(net, dataset=dataset, batch_range=batch_range, disp=True)
        else:
            print('Unknown action', file=sys.stderr)

    net.server.terminate()
    net.server.join()
