import math
import multiprocessing
import os
import random
import re
import sys
import time
import traceback
from itertools import chain
from multiprocessing import Process, Manager
from shutil import copyfile

import numpy as np

from goprime import Constants
from goprime.board import Position
from goprime.elo import Elo
from goprime.log import TBLogger, Log
from goprime.mcts import TreeNode, tree_search, dump_subtree, position_dist, position_distnext


class Game:

    def __init__(self, N, elo_k=0.0025):
        self.N = N
        self.W = N + 2
        self.K = elo_k

    @staticmethod
    def game_limit_factor(N):
        return N * N * 2 if N < 14 else N * N + N

    def play_and_train(self, net, i, batches_per_game=2, output_stream=None):

        N = self.N
        W = self.W

        count = 0
        positions = []
        owner_map = W * W * [0]

        allow_resign = i > 10 and np.random.rand() < Constants.P_ALLOW_RESIGN

        tree = TreeNode(net=net, pos=Position.empty_position(N))
        tree.expand()

        while True:
            next_tree = tree_search(tree, Constants.N_SIMS, output_stream=output_stream, debug_disp=False)

            positions.append((tree.pos, tree.distribution()))

            tree = next_tree

            if output_stream is not None:
                Position.print(tree.pos, output_stream, owner_map)

            if tree.pos.last is None and tree.pos.last2 is None:
                score = 1 if tree.pos.score() > 0 else -1
                if tree.pos.n % 2:
                    score = -score
                if output_stream is not None:
                    print('Two passes, score: B%+.1f' % (score,), file=output_stream)

                    count = tree.pos.score()
                    if tree.pos.n % 2:
                        count = -count
                    print('Counted score: B%+.1f' % (count,), file=output_stream)
                break

            if allow_resign and float(
                    tree.w) / tree.v < Constants.RESIGN_THRES and tree.v > Constants.N_SIMS / 10 and tree.pos.n > 10:
                score = 1  # win for player to-play from this position
                if tree.pos.n % 2:
                    score = -score
                if output_stream is not None:
                    print('Resign (%d), score: B%+.1f' % (tree.pos.n % 2, score), file=output_stream)

                    count = tree.pos.score()
                    if tree.pos.n % 2:
                        count = -count
                    print('Counted score: B%+.1f' % (count,), file=output_stream)
                break

            if tree.pos.n > Game.game_limit_factor(N):
                if output_stream is not None:
                    print('Stopping too long a game.', file=output_stream)
                score = 0
                break

        # score here is for black to play (player-to-play from empty_position)
        if output_stream is not None:
            print(score, file=output_stream)
            dump_subtree(tree, f=output_stream)

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

        return score, count

    def selfplay_singlethread(self, net, worker_id, elo_rating, games, elo_logger, log="file"):
        net.ri = worker_id
        model_name = net.model_name()

        if log == "file":
            output_stream = open("logs/{0}_{1}.log".format(model_name, worker_id), "a")
        elif log == "stdout":
            output_stream = sys.stdout
        else:
            output_stream = None

        print("Elo rating: {0}".format(elo_rating), file=output_stream)

        black = elo_rating["black"][0]
        white = elo_rating["white"][0]

        for game in range(games):
            print('[%d %d] Self-play of game #%d ...' % (worker_id, time.time(), game,), file=output_stream)
            print('[%d %d] Self-play of game #%d ...' % (worker_id, time.time(), game,))

            score = self.play_and_train(net, game, batches_per_game=games - game, output_stream=output_stream)[0]

            belo = Elo()
            result = 'WIN' if score > 0 else ('LOST' if score < 0 else 'DRAW')
            belo.add_init_elo_ratings(black, white, result=result)
            black = belo.get_final_elo_ratings(k=self.K)[0]
            elo_rating["black"] += [black]

            welo = Elo()
            result = 'LOST' if score > 0 else ('WIN' if score < 0 else 'DRAW')
            welo.add_init_elo_ratings(white, black, result=result)
            white = welo.get_final_elo_ratings(k=self.K)[0]
            elo_rating["white"] += [white]

            elo_logger.log(log=Log.Scalar, session_id=model_name, tag="Black", value=black)
            elo_logger.log(log=Log.Scalar, session_id=model_name, tag="White", value=white)

        print("Elo rating: {0}".format(elo_rating), file=output_stream)
        print("Completed game for worker #{0}".format(worker_id))

    def selfplay(self, net, games=int(5e4), log="file"):
        n_workers = multiprocessing.cpu_count() * 4
        # group up parallel predict requests
        net.set_stash_size(max(n_workers, 1))

        elo_logger = TBLogger('logs/tensorboard/selfplay')

        manager = Manager()
        elo_rating = manager.dict()

        elo_rating["black"] = [-2000.0]
        elo_rating["white"] = [-2000.0]

        rounds = 16
        games_per_thread = int(math.ceil(games / (n_workers * rounds)))

        for round in range(rounds):
            processes = []

            # processes = [Process(target=self.selfplay_singlethread,
            #                      kwargs=dict(net=net, worker_id=0, elo_rating=elo_rating, games=games_per_thread,
            #                                  elo_logger=elo_logger, log='stdout'))]

            # The rest work silently
            for i in range(0 + len(processes), n_workers - len(processes)):
                processes.append(
                    Process(target=self.selfplay_singlethread,
                            kwargs=dict(net=net, worker_id=i, elo_rating=elo_rating, games=games_per_thread,
                                        elo_logger=elo_logger, log=log)))

            for p in processes:
                p.start()

            while True:
                running = any(p.is_alive() for p in processes)
                if not running:
                    break
                time.sleep(1.0)

            for p in processes:
                p.join()

            elo_rating["black"] = [max(elo_rating["black"][1:])]
            elo_rating["white"] = [max(elo_rating["white"][1:])]

            snapshot_id = '%s_%09d' % (net.model_name(), round)
            net.save(snapshot_id)
            print("Saved {0}".format(snapshot_id))

            print("[{0} out {1}] games completed"
                  .format(games_per_thread * (round + 1), games_per_thread * n_workers * rounds))

    def selfplay_till_elo(self, net, elo=2000, games_per_thread=2, log="file"):
        model_name = net.model_name()
        n_workers = multiprocessing.cpu_count() * games_per_thread
        self.K = self.K * n_workers

        # group up parallel predict requests
        stash_size = max(n_workers, 1)
        net.set_stash_size(stash_size)

        weights_dir = "weights"
        elo_logger = TBLogger('logs/tensorboard/selfplay')

        manager = Manager()
        elo_rating = manager.dict()

        model_elo_rating = -2000.0
        model_elo_ratings = [model_elo_rating]

        target_selfplay_dat = "target_selfplay.dat"
        if os.path.isfile(target_selfplay_dat):
            with open(target_selfplay_dat) as f:
                lines = f.readlines()
            if len(lines) >= 2:
                model_elo_rating = float(lines[0])
                model_elo_ratings = [float(rate.strip()) for rate in lines[1].strip("[]").split(",")]
                if model_elo_rating < max(model_elo_ratings):
                    model_elo_rating = max(model_elo_ratings)

        elo_rating["black"] = [model_elo_rating]
        elo_rating["white"] = [model_elo_rating]

        iter_round = 0
        while model_elo_rating < elo:
            processes = []

            # processes = [Process(target=self.selfplay_singlethread,
            #                      kwargs=dict(net=net, worker_id=0, elo_rating=elo_rating, games=games_per_thread,
            #                                  elo_logger=elo_logger, log='stdout'))]

            # The rest work silently
            for i in range(0 + len(processes), n_workers):
                processes.append(
                    Process(target=self.selfplay_singlethread,
                            kwargs=dict(net=net, worker_id=i, elo_rating=elo_rating, games=games_per_thread,
                                        elo_logger=elo_logger, log=log)))

            for p in processes:
                p.start()

            while True:
                running = len([1 for p in processes if p.is_alive()])
                if running == 0:
                    break
                elif running != stash_size:
                    stash_size = running
                    net.set_stash_size(stash_size)
                time.sleep(1.0)

            for p in processes:
                p.join()

            stash_size = max(n_workers, 1)
            net.set_stash_size(stash_size)

            elo_rating["black"] = [max(elo_rating["black"][1:])]
            elo_rating["white"] = [max(elo_rating["white"][1:])]
            model_elo_rating = max(elo_rating["black"][0], elo_rating["white"][0])

            # save the snapshot
            snapshot_id = '%s/%s_%09d' % (weights_dir, model_name, iter_round)
            net.save(snapshot_id)
            print("Saved {0}".format(snapshot_id))

            # get the max elo from all jobs
            model_elo_ratings.append(model_elo_rating)
            if os.path.isfile(target_selfplay_dat):
                with open(target_selfplay_dat) as f:
                    lines = f.readlines()
                if len(lines) >= 2:
                    if float(lines[0]) < model_elo_rating:
                        model_elo_rating = float(lines[0])
                        ratings = [float(rate.strip()) for rate in lines[1].strip("[]").split(",")]
                        ratings.append(model_elo_ratings[-1])
                        model_elo_ratings = ratings
                        if model_elo_rating < max(model_elo_ratings):
                            model_elo_rating = max(model_elo_ratings)

            # save elo rating to file
            time.sleep(1.0)
            with open(target_selfplay_dat, "w") as f:
                f.write("{0}\n{1}".format(model_elo_rating, model_elo_ratings))

            # reduce the position archive (to save memory)
            # try:
            #     if iter_round % (19 - self.N - games_per_thread) == 0:
            #         net.reduce_position_archive()
            #     else:
            #         net.reduce_position_archive(ratio=0.7)
            # except:
            #     traceback.print_exc()

            # checkpoint iteration - sync elos and remove incremental weights
            if iter_round != 0 and iter_round % self.N == 0:

                elo_rating["black"] = [model_elo_rating]
                elo_rating["white"] = [model_elo_rating]

                # making sure the file is created
                while not os.path.isfile('%s.weights.h5' % snapshot_id):
                    print("Weights not saved yet!")
                    time.sleep(1.0)
                    continue

                try:
                    copyfile('%s.weights.h5' % snapshot_id,
                             '%s/incremental/%s_%09d.weights.h5' % (weights_dir, model_name, iter_round))
                    # copyfile('%s.archive.joblib' % snapshot_id,
                    #          '%s/archive/%s_%09d.archive.joblib' % (weights_dir, model_name, iter_round))

                    for the_file in os.listdir(weights_dir):
                        file_path = os.path.join(weights_dir, the_file)
                        try:
                            if os.path.isfile(file_path) and model_name in file_path:
                                os.unlink(file_path)
                        except:
                            traceback.print_exc()
                except:
                    traceback.print_exc()

            iter_round += 1
            print("Stone Elo Rating = {0}".format(elo_rating))
            print("Model Elo Rating = {0}".format(model_elo_rating))
            print("{0} games completed!".format(games_per_thread * iter_round * n_workers))

    def replay_train_thread(self, net, worker_id, file_id, filename, batches_per_game=2, disp=False):
        print("Processing [{0}]: {1}".format(file_id, filename))

        net.ri = worker_id
        positions, score = self.gather_positions(filename, subsample=16)

        dist = [position_dist(net, worker_id, pos, disp) for i, pos in enumerate(positions)]
        X_positions = list(zip(positions, dist))

        if disp:
            Position.print(X_positions[0][0], sys.stdout, None)

        for i in range(batches_per_game):
            net.fit_game(X_positions, score)
            # fit flipped positions
            net.fit_game(X_positions, score, board_transform='flip_vert')
            net.fit_game(X_positions, score, board_transform='flip_horiz')
            net.fit_game(X_positions, score, board_transform='flip_both')
            # TODO 90\deg rot

    def replay_traindist(self, net, dataset, snapshot_interval=500, batches_per_game=2, batch_range=None, disp=False):

        # N = self.N

        n_workers = multiprocessing.cpu_count() * 4
        net.set_stash_size(max(n_workers, 1))

        if batch_range is not None:
            sgf_files = os.listdir(dataset)[batch_range[0]: batch_range[1]]
        else:
            sgf_files = os.listdir(dataset)

        dataset = [os.path.join(dataset, file) for file in sgf_files]
        dataset_size = len(dataset)

        i = 0
        while i < dataset_size:
            processes = []

            for j in range(n_workers if dataset_size >= i + n_workers else n_workers - dataset_size + i):
                processes.append(
                    Process(target=self.replay_train_thread,
                            kwargs=dict(net=net, worker_id=j, file_id=i + j, filename=dataset[i + j],
                                        batches_per_game=batches_per_game, disp=disp)))

            for p in processes:
                p.start()

            while True:
                running = any(p.is_alive() for p in processes)
                if not running:
                    break
                time.sleep(1.0)

            for p in processes:
                p.join()

            if snapshot_interval and i <= snapshot_interval <= i + n_workers:
                snapshot_id = '%s_R%09d' % (net.model_name(), i)
                print(snapshot_id)
                net.save(snapshot_id)

            i += n_workers

        snapshot_id = '%s_final' % (net.model_name(),)
        print(snapshot_id)
        net.save(snapshot_id)

        net.is_model_ready()
        print("Done!!")

    def replay_train(self, net, dataset, snapshot_interval=5000, batches_per_game=2, batch_range=None, disp=True):

        N = self.N

        if batch_range is not None:
            if batch_range[1] == 0:
                sgf_files = os.listdir(dataset)[batch_range[0]:]
            else:
                sgf_files = os.listdir(dataset)[batch_range[0]: batch_range[1]]
        else:
            sgf_files = os.listdir(dataset)

        dataset = [os.path.join(dataset, file) for file in sgf_files]
        dataset_size = len(dataset)

        X_positions = []
        for i in range(dataset_size):
            f = dataset[i]

            print('[%d] %s' % (i, f))
            sys.stdout.flush()

            try:
                positions, score = self.gather_positions(f, subsample=16)
            except:  # ValueError:
                print('SKIP')
                sys.stdout.flush()
                import traceback
                traceback.print_exc()
                continue

            dist = [position_distnext(N, pos) for pos in positions]

            X_positions += list(zip(positions, dist))

            if snapshot_interval and i % snapshot_interval == 0 and i > 0:
                for j in range(batches_per_game):
                    net.fit_game(X_positions, score)
                    # fit flipped positions
                    net.fit_game(X_positions, score, board_transform='flip_vert')
                    net.fit_game(X_positions, score, board_transform='flip_horiz')
                    net.fit_game(X_positions, score, board_transform='flip_both')
                    # TODO 90\deg rot

                X_positions = []
                snapshot_id = '%s_R%09d' % (net.model_name(), i)
                net.save(snapshot_id)

                print(snapshot_id, file=sys.stdout)
                print("Waiting...", file=sys.stdout)
                sys.stdout.flush()

                net.reduce_position_archive(ratio=0.5)

                # to wait before submitting the next batch
                time.sleep(snapshot_interval * 4.0)

        snapshot_id = '%s_final' % (net.model_name(),)
        print(snapshot_id)
        net.save(snapshot_id)

        net.is_model_ready()
        print("Done!!")

    def gather_positions(self, filename, encoding='utf-8', subsample=16):
        from gomill import sgf

        N = self.N
        W = self.W

        try:
            with open(filename, mode='rt', encoding=encoding) as f:
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
        except:
            traceback.print_exc()
            if encoding == 'utf-8':
                return self.gather_positions(filename, encoding='latin1', subsample=16)

        # subsample positions
        pos_to_play = [random.sample(pos_to_play[0], subsample // 2), random.sample(pos_to_play[1], subsample // 2)]

        # alternate positions and randomly rotate
        positions = list(chain(*zip(*pos_to_play)))

        flipped = [pos.flip_random() for pos in positions]

        return flipped, score

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
                try:
                    c = Position.parse_coord(sc)
                except:
                    traceback.print_exc()
                    continue

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

            tree = tree_search(tree, Constants.N_SIMS, output_stream=sys.stdout)
            if tree.pos.last is None and tree.pos.last2 is None:
                score = tree.pos.score()
                if tree.pos.n % 2:
                    score = -score
                print('Game over, score: B%+.1f' % (score,))
                break

            if float(tree.w) / tree.v < Constants.RESIGN_THRES and tree.pos.n > 10:
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
                tree = tree_search(tree, Constants.N_SIMS, output_stream=None)
                if tree.pos.last is None:
                    ret = 'pass'
                elif float(tree.w) / tree.v < Constants.RESIGN_THRES and tree.pos.n > 10:
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
                ret = 'goprime parallel hybrid'
            elif command[0] == "tsdebug":
                Position.print(tree_search(tree, Constants.N_SIMS, output_stream=sys.stdout))
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

            # if command[0] in ['genmove', 'play']:
            #     Position.print(tree.pos, sys.stderr, owner_map)

            # if ret is not None:
            #     print('=%s %s\n\n' % (cmdid, ret,), end='')
            # else:
            #     print('?%s ???\n\n' % (cmdid,), end='')

            print("= {0}\n".format(ret))
            sys.stdout.flush()
