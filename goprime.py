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
    Option = 2
    Snapshot = 3


if __name__ == "__main__" and 1 < len(sys.argv) < 5:

    N = int(sys.argv[ConsoleParams.N.value])
    W = N + 2

    mode = sys.argv[ConsoleParams.Option.value]
    snapshot = sys.argv[ConsoleParams.Snapshot.value] if len(sys.argv) > 3 else None

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
            # find GoGoD-2008-Winter-Database/ -name '*.sgf' | shuf | python ./goprime.py 19 replay_train
            game.replay_train(net, disp=False)
        elif mode == "replay_traindist":
            game.replay_train(net, continuous_predict=True)
        else:
            print('Unknown action', file=sys.stderr)

    net.server.terminate()
    net.server.join()
