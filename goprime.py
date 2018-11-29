import multiprocessing
import os
import sys
import time
from enum import Enum
from pprint import pprint

from goprime import *
from goprime.board import Position
from goprime.game import Game
from goprime.mcts import TreeNode, tree_search
from goprime.model import GoModel

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


class ConsoleParams(Enum):
    N = 1
    Mode = 2
    Snapshot = 3
    BatchRange = 4
    DataSet = 5
    Games = 6


console_options = ["-n", "-m", "-s", "-b", "-d", "-g"]
options_lookup = {
    "-n": ConsoleParams.N,
    "-m": ConsoleParams.Mode,
    "-s": ConsoleParams.Snapshot,
    "-b": ConsoleParams.BatchRange,
    "-d": ConsoleParams.DataSet,
    "-g": ConsoleParams.Games
}
options_type = {
    "-n": int,
    "-m": str,
    "-s": str,
    "-b": "range",
    "-d": "path",
    "-g": int
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
    pprint(params)
    # exit(1)

    N = int(params[ConsoleParams.N])
    W = N + 2

    snapshot = params[ConsoleParams.Snapshot]
    # weighted average initialization
    if snapshot is not None and snapshot.startswith('~'):
        dir_path = snapshot.strip('~ /')
        snapshot = []

        for item in os.listdir(dir_path):
            w_snap = os.path.join(dir_path, item)
            if os.path.isfile(w_snap) and w_snap.endswith(".h5"):
                snapshot.append(w_snap)

    batch_size = 32
    net = GoModel(board_size=N, batch_size=batch_size, load_snapshot=snapshot)

    if net.is_model_ready():

        net.set_waiting_status()

        mode = params[ConsoleParams.Mode]
        game = Game(N, elo_k=round(1 / Game.game_limit_factor(N), 4))

        if mode == "play":
            mode = input("Choose the color of your stone [Black(b) / White(w)]: ").strip()[0].lower()
            if mode == "b":
                # Default action
                while True:
                    game.game_io(net)
            else:
                while True:
                    game.game_io(net=net, computer_black=True)
        elif mode == "gtp":
            game.gtp_io(net)
        elif mode == "tsbenchmark":
            t_start = time.time()

            Position.print(tree_search(TreeNode(net=net, pos=Position.empty_position(N)),
                                       N_SIMS, W * W * [0]).pos)

            print('Tree search with %d playouts took %.3fs with %d threads; speed is %.3f playouts/thread/s' %
                  (N_SIMS, time.time() - t_start, multiprocessing.cpu_count(),
                   N_SIMS / ((time.time() - t_start) * multiprocessing.cpu_count())))
        elif mode == "tsdebug":
            Position.print(tree_search(TreeNode(net=net, pos=Position.empty_position(N)),
                                       N_SIMS, output_stream=sys.stdout).pos)
        elif mode == "retrain":
            net.cmd_queue.put(('retrain', {'snapshot_id': snapshot, 'batch_size': batch_size}, 0))
            if net.is_model_ready():
                pass
        elif mode == "selfplay":
            if ConsoleParams.Games in params:
                games = params[ConsoleParams.Games]
                game.selfplay(net, games=games)
            else:
                game.selfplay(net)
        elif mode == "target_selfplay":
            game.selfplay_till_elo(net, log="file")
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

    # TODO File with last file id
    # TODO Submit next job based on the last id
    # TODO Write code for self play cum supervised
    # TODO Elo rating
    # TODO GTP IO
