import sys
from multiprocessing import Process, Queue
import numpy as np


class ModelServer(Process):
    def __init__(self, board_size, cmd_queue, res_queues, load_snapshot=None):
        super(ModelServer, self).__init__()

        self.board_size = board_size
        self.cmd_queue = cmd_queue
        self.res_queues = res_queues
        self.load_snapshot = load_snapshot

    def run(self):
        try:
            from goprime.net import AGZeroModel

            N = self.board_size

            net = AGZeroModel(N)
            net.create()

            if self.load_snapshot is not None:
                net.load(self.load_snapshot)

            class PredictStash(object):
                """ prediction batcher """

                def __init__(self, trigger, res_queues):
                    self.stash = []
                    self.trigger = trigger  # XXX must not be higher than #workers
                    self.res_queues = res_queues

                def add(self, kind, X_pos, ri):
                    self.stash.append((kind, X_pos, ri))
                    if len(self.stash) >= self.trigger:
                        self.process()

                def process(self):
                    if not self.stash:
                        return
                    dist, res = net.predict(np.array([s[1] for s in self.stash]))
                    for d, r, s in zip(dist, res, self.stash):
                        kind, _, ri = s
                        self.res_queues[ri].put(d if kind == 0 else r)
                    self.stash = []

            stash = PredictStash(1, self.res_queues)
            fit_counter = 0

            while True:
                cmd, args, ri = self.cmd_queue.get()
                if cmd == 'stash_size':
                    stash.process()
                    stash.trigger = args['stash_size']
                elif cmd == 'fit_game':
                    stash.process()
                    print('\rFit %d...' % (fit_counter,), end='')
                    sys.stdout.flush()
                    fit_counter += 1
                    net.fit_game(**args)
                elif cmd == 'predict_distribution':
                    stash.add(0, args['X_position'], ri)
                elif cmd == 'predict_winrate':
                    stash.add(1, args['X_position'], ri)
                elif cmd == 'model_name':
                    self.res_queues[ri].put(net.model_name)
                elif cmd == 'save':
                    stash.process()
                    net.save(args['snapshot_id'])
        except:
            import traceback
            traceback.print_exc()


class GoModel(object):
    def __init__(self, board_size, load_snapshot=None):

        self.board_size = board_size
        self.cmd_queue = Queue()
        self.res_queues = [Queue() for i in range(128)]

        self.server = ModelServer(self.board_size, self.cmd_queue, self.res_queues, load_snapshot=load_snapshot)
        self.server.start()

        self.ri = 0  # id of process in case of multiple processes, to prevent mixups

    def stash_size(self, stash_size):
        self.cmd_queue.put(('stash_size', {'stash_size': stash_size}, self.ri))

    def encode_position(self, position, board_transform=None):

        N = self.board_size
        W = N + 2

        my_stones, their_stones, edge, last, last2, to_play = np.zeros((N, N)), np.zeros((N, N)), np.zeros(
            (N, N)), np.zeros((N, N)), np.zeros((N, N)), np.zeros((N, N))

        if board_transform:
            # noinspection PyUnresolvedReferences
            from goprime.board import Position
            position = eval('Position.' + board_transform)(position)
        board = position.board

        for c, p in enumerate(board):
            x, y = c % W - 1, c // W - 1
            # In either case, y and x should be sane (not off-board)
            if p == 'X':
                my_stones[y, x] = 1
            elif p == 'x':
                their_stones[y, x] = 1
            if not (0 <= x < N and 0 <= y < N):  # (x >= 0 and x < N and y >= 0 and y < N)
                continue
            if x == 0 or x == N - 1 or y == 0 or y == N - 1:
                edge[y, x] = 1
            if position.last == c:
                last[y, x] = 1
            if position.last2 == c:
                last2[y, x] = 1
            if position.n % 2 == 1:
                to_play[y, x] = 1

        return np.stack((my_stones, their_stones, edge, last, last2, to_play), axis=-1)

    def fit_game(self, positions, result, board_transform=None):
        X_positions = [(self.encode_position(pos, board_transform=board_transform), dist) for pos, dist in positions]
        self.cmd_queue.put(('fit_game', {'X_positions': X_positions, 'result': result}, self.ri))

    def predict_distribution(self, position):
        self.cmd_queue.put(('predict_distribution', {'X_position': self.encode_position(position)}, self.ri))
        return self.res_queues[self.ri].get()

    def predict_winrate(self, position):
        self.cmd_queue.put(('predict_winrate', {'X_position': self.encode_position(position)}, self.ri))
        return self.res_queues[self.ri].get()

    def model_name(self):
        self.cmd_queue.put(('model_name', {}, self.ri))
        return self.res_queues[self.ri].get()

    def save(self, snapshot_id):
        self.cmd_queue.put(('save', {'snapshot_id': snapshot_id}, self.ri))
