import multiprocessing
import random
import re
import sys
import time
from itertools import chain
from multiprocessing import Process

import numpy as np

from goprime import *
from goprime.board import Position
from goprime.mcts import MCTree, TreeNode
from joblib import Parallel, delayed


class Game:

    def __init__(self, N):
        self.N = N
        self.W = N + 2

    def play_and_train(self, net, i, batches_per_game=2, disp=False):

        N = self.N
        W = self.W

        positions = []
        owner_map = W * W * [0]

        allow_resign = i > 10 and np.random.rand() < P_ALLOW_RESIGN

        tree = TreeNode(net=net, pos=Position.empty_position(N))
        tree.expand()

        while True:
            next_tree = MCTree.tree_search(tree, N_SIMS, owner_map, disp=disp)

            positions.append((tree.pos, tree.distribution()))

            tree = next_tree

            if disp:
                Position.print(tree.pos, sys.stdout, owner_map)

            if tree.pos.last is None and tree.pos.last2 is None:
                score = 1 if tree.pos.score() > 0 else -1
                if tree.pos.n % 2:
                    score = -score
                if disp:
                    print('Two passes, score: B%+.1f' % (score,))

                    count = tree.pos.score()
                    if tree.pos.n % 2:
                        count = -count
                    print('Counted score: B%+.1f' % (count,))
                break

            if allow_resign and float(tree.w) / tree.v < RESIGN_THRES and tree.v > N_SIMS / 10 and tree.pos.n > 10:
                score = 1  # win for player to-play from this position
                if tree.pos.n % 2:
                    score = -score
                if disp:
                    print('Resign (%d), score: B%+.1f' % (tree.pos.n % 2, score))

                    count = tree.pos.score()
                    if tree.pos.n % 2:
                        count = -count
                    print('Counted score: B%+.1f' % (count,))
                break

            if tree.pos.n > N * N * 2:
                if disp:
                    print('Stopping too long a game.')
                score = 0
                break

        # score here is for black to play (player-to-play from empty_position)
        if disp:
            print(score)
            MCTree.dump_subtree(tree)

        for i in range(batches_per_game):
            net.fit_game(positions, score)

        # fit flipped positions
        for i in range(batches_per_game):
            net.fit_game(positions, score, board_transform='flip_vert')

        for i in range(batches_per_game):
            net.fit_game(positions, score, board_transform='flip_horiz')

        for i in range(batches_per_game):
            net.fit_game(positions, score, board_transform='flip_both')

        # TODO 90\deg rot

    def selfplay_singlethread(self, net, worker_id, disp=False, snapshot_interval=1):
        net.ri = worker_id

        i = 0
        while True:
            print('\n[%d %d] Self-play of game #%d ...\n' % (worker_id, time.time(), i,))
            self.play_and_train(net, i, disp=disp)

            i += 1
            if snapshot_interval and i % snapshot_interval == 0:
                snapshot_id = '%s_%09d' % (net.model_name(), i)
                print(snapshot_id)
                net.save(snapshot_id)

            if i % 40 == 0: break

    def selfplay(self, net, disp=True):
        n_workers = multiprocessing.cpu_count() * 8  # 6

        # group up parallel predict requests
        net.stash_size(max(multiprocessing.cpu_count(), 1))

        # First process is verbose and snapshots the model
        processes = [Process(target=self.selfplay_singlethread, kwargs=dict(net=net, worker_id=0, disp=disp))]
        # The rest work silently
        for i in range(1, n_workers):
            processes.append(
                Process(target=self.selfplay_singlethread, kwargs=dict(net=net, worker_id=i, snapshot_interval=None)))

        for p in processes:
            p.start()
        for p in processes:
            p.join()

    def replay_train(self, net, snapshot_interval=500, continuous_predict=False, disp=True):

        N = self.N

        n_workers = multiprocessing.cpu_count()
        # group up parallel predict requests
        # net.stash_size(max(2, 1))  # XXX not all workers will always be busy

        for i, f in enumerate(sys.stdin):
            f = f.rstrip()
            print('[%d] %s' % (i, f))

            try:
                positions, score = self.gather_positions(f, subsample=16)
            except ValueError:
                print('SKIP')
                import traceback
                traceback.print_exc()
                continue

            if continuous_predict:
                dist = Parallel(n_jobs=n_workers, verbose=100)(
                    delayed(MCTree.position_dist)(N, i, pos, disp) for i, pos in enumerate(positions))
            else:
                dist = [MCTree.position_distnext(N, pos) for pos in positions]

            X_positions = list(zip(positions, dist))
            if disp:
                Position.print(X_positions[0][0], sys.stdout, None)
            net.fit_game(X_positions, score)

            if snapshot_interval and i > 0 and i % snapshot_interval == 0:
                snapshot_id = '%s_R%09d' % (net.model_name(), i)
                print(snapshot_id)
                net.save(snapshot_id)

        snapshot_id = '%s_Rfinal' % (net.model_name(),)
        print(snapshot_id)
        net.save(snapshot_id)

    def game_io(self, net, computer_black=False):
        """ A simple minimalistic text mode UI. """

        N = self.N
        W = self.W

        tree = TreeNode(net=net, pos=Position.empty_position(N))
        tree.expand()
        owner_map = W * W * [0]

        while True:
            if not (tree.pos.n == 0 and computer_black):
                Position.print(tree.pos, sys.stdout, owner_map)

                sc = input("Your move: ")
                c = Position.parse_coord(sc)
                if c is not None:
                    # Not a pass
                    if tree.pos.board[c] != '.':
                        print('Bad move (not empty point)')
                        continue

                    # Find the next node in the game tree and proceed there
                    nodes = list(filter(lambda n: n.pos.last == c, tree.children))
                    if not nodes:
                        print('Bad move (rule violation)')
                        continue
                    tree = nodes[0]

                else:
                    # Pass move
                    if tree.children[0].pos.last is None:
                        tree = tree.children[0]
                    else:
                        tree = TreeNode(net=net, pos=tree.pos.pass_move())

                Position.print(tree.pos)

            owner_map = W * W * [0]
            tree = MCTree.tree_search(tree, N_SIMS, owner_map, disp=True)
            if tree.pos.last is None and tree.pos.last2 is None:
                score = tree.pos.score()
                if tree.pos.n % 2:
                    score = -score
                print('Game over, score: B%+.1f' % (score,))
                break

            if float(tree.w) / tree.v < RESIGN_THRES and tree.pos.n > 10:
                print('I resign.')
                break

        print('Thank you for the game!')

    def gtp_io(self, net):
        """ GTP interface for our program.  We can play only on the board size
        which is configured (N), and we ignore color information and assume
        alternating play! """
        known_commands = ['boardsize', 'clear_board', 'komi', 'play', 'genmove',
                          'final_score', 'quit', 'name', 'version', 'known_command',
                          'list_commands', 'protocol_version', 'tsdebug']

        N = self.N
        W = self.W

        tree = TreeNode(net=net, pos=Position.empty_position(N))
        tree.expand()

        while True:
            try:
                line = input().strip()
            except EOFError:
                break
            if line == '':
                continue
            command = [s.lower() for s in line.split()]
            if re.match('\d+', command[0]):
                cmdid = command[0]
                command = command[1:]
            else:
                cmdid = ''
            owner_map = W * W * [0]
            ret = ''
            if command[0] == "boardsize":
                if int(command[1]) != N:
                    print("Warning: Trying to set incompatible boardsize %s (!= %d)" % (command[1], N), file=sys.stderr)
                    ret = None
            elif command[0] == "clear_board":
                tree = TreeNode(net=net, pos=Position.empty_position(N))
                tree.expand()
            elif command[0] == "komi":
                # XXX: can we do this nicer
                tree.pos = Position(board=tree.pos.board, cap=(tree.pos.cap[0], tree.pos.cap[1]),
                                    n=tree.pos.n, ko=tree.pos.ko, last=tree.pos.last, last2=tree.pos.last2,
                                    komi=float(command[1]), data=dict())
            elif command[0] == "play":
                c = Position.parse_coord(command[2])
                if c is not None:
                    # Find the next node in the game tree and proceed there
                    if tree.children is not None and filter(lambda n: n.pos.last == c, tree.children):
                        tree = list(filter(lambda n: n.pos.last == c, tree.children))[0]
                    else:
                        # Several play commands in row, eye-filling move, etc.
                        tree = TreeNode(net=net, pos=tree.pos.move(c))

                else:
                    # Pass move
                    if tree.children[0].pos.last is None:
                        tree = tree.children[0]
                    else:
                        tree = TreeNode(net=net, pos=tree.pos.pass_move())
            elif command[0] == "genmove":
                tree = MCTree.tree_search(tree, N_SIMS, owner_map, disp=True)
                if tree.pos.last is None:
                    ret = 'pass'
                elif float(tree.w) / tree.v < RESIGN_THRES and tree.pos.n > 10:
                    ret = 'resign'
                else:
                    ret = Position.str_coord(tree.pos.last)
            elif command[0] == "final_score":
                score = tree.pos.score()
                if tree.pos.n % 2:
                    score = -score
                if score == 0:
                    ret = '0'
                elif score > 0:
                    ret = 'B+%.1f' % (score,)
                elif score < 0:
                    ret = 'W+%.1f' % (-score,)
            elif command[0] == "name":
                ret = 'goprime'
            elif command[0] == "version":
                ret = 'simple go program demo'
            elif command[0] == "tsdebug":
                Position.print(MCTree.tree_search(tree, N_SIMS, W * W * [0], disp=True))
            elif command[0] == "list_commands":
                ret = '\n'.join(known_commands)
            elif command[0] == "known_command":
                ret = 'true' if command[1] in known_commands else 'false'
            elif command[0] == "protocol_version":
                ret = '2'
            elif command[0] == "quit":
                print('=%s \n\n' % (cmdid,), end='')
                break
            else:
                print('Warning: Ignoring unknown command - %s' % (line,), file=sys.stderr)
                ret = None

            Position.print(tree.pos, sys.stderr, owner_map)
            if ret is not None:
                print('=%s %s\n\n' % (cmdid, ret,), end='')
            else:
                print('?%s ???\n\n' % (cmdid,), end='')
            sys.stdout.flush()

    def gather_positions(self, filename, subsample=16):
        from gomill import sgf

        N = self.N
        W = self.W

        with open(filename) as f:
            g = sgf.Sgf_game.from_string(f.read())
            if g.get_size() != N:
                raise ValueError('size mismatch')
            if g.get_handicap() is not None:
                raise ValueError('handicap game')

            score = 1 if g.get_winner() == 'B' else -1

            pos_to_play = [[], []]  # black-to-play, white-to-play
            pos = Position.empty_position(N)
            for node in g.get_main_sequence()[1:]:
                color, move = node.get_move()
                if move is not None:
                    c = (move[0] + 1) * W + move[1] + 1
                    pos.data['next'] = c
                    pos = pos.move(c)
                else:
                    pos.data['next'] = None
                    pos = pos.pass_move()
                if pos is None:
                    raise ValueError('invalid move %s' % (move,))
                pos_to_play[pos.n % 2].append(pos)
            pos.data['next'] = None

        # subsample positions
        pos_to_play = [random.sample(pos_to_play[0], subsample // 2), random.sample(pos_to_play[1], subsample // 2)]

        # alternate positions and randomly rotate
        positions = list(chain(*zip(*pos_to_play)))

        flipped = [pos.flip_random() for pos in positions]

        return (flipped, score)
