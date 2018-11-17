import datetime
import logging
import os
import time
from multiprocessing import Process, Queue
from multiprocessing.sharedctypes import Array, Value

import numpy as np


def log(msg):
    if not os.path.exists("logs/"):
        os.makedirs("logs/")

    # set basic polling config
    logging.basicConfig(
        format="%(message)s",
        level=logging.DEBUG,
        # handlers=[logging.StreamHandler(sys.stdout)],
        filename="logs/ModelServer.logs", filemode="w"
    )

    now = datetime.datetime.now()
    logging.info("{0}: {1}".format(now.strftime("%Y-%m-%d %H:%M:%S"), msg))


class ModelServer(Process):
    def __init__(self, board_size, cmd_queue, res_queues, status, stash_size=1, retrain_after=5e4, load_snapshot=None):
        super(ModelServer, self).__init__()

        self.board_size = board_size
        self.cmd_queue = cmd_queue
        self.res_queues = res_queues
        self.load_snapshot = load_snapshot

        self.status = status
        self.stash_size = stash_size
        self.status.value = b'INITIALIZED'
        self.retrain_after = int(retrain_after)

        log("Model Server initialized!")

    def run(self):
        try:
            from goprime.net import AGZeroModel

            N = self.board_size
            start_time = time.time()

            net = AGZeroModel(N)
            net.create()

            if self.load_snapshot is not None:
                net.load(self.load_snapshot)
                log("Snapshot loaded")

            log("Neural Network Initialized!")

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

            stash = PredictStash(self.stash_size.value, self.res_queues)

            log("Predict Stash Initialized!")

            self.status.value = b'READY'

            fit_counter = 0
            finished = False
            snapshot_id = "First Save ({0})".format(start_time)

            while True and not finished:
                if time.time() - start_time >= 86000:
                    net.save("{0}_last".format(snapshot_id))
                    log("Emergency snapshot saved before program termination!")

                    # TODO write the code to submit next job here...
                    finished = True
                    continue

                cmd, args, ri = self.cmd_queue.get()

                if cmd == 'stash_size':
                    stash.process()
                    stash.trigger = args['stash_size']
                    self.stash_size.value = args['stash_size']
                elif cmd == 'stash_process':
                    stash.process()
                elif cmd == 'fit_game':
                    self.status.value = b'FIT_STARTED'
                    stash.process()
                    log("Fit {0} started...".format(fit_counter))
                    # print('\rFit %d started...' % (fit_counter,), end='')
                    # sys.stdout.flush()

                    net.fit_game(**args)

                    # print('\rFit %d completed...' % (fit_counter,), end='')
                    # sys.stdout.flush()
                    log("Fit {0} completed...".format(fit_counter))

                    fit_counter += 1
                    self.status.value = b'FIT_COMPLETED'

                    if fit_counter % self.retrain_after == 1:
                        log("Retrain model started...")
                        net.retrain_position_archive()
                        log("Retrain model completed...")
                elif cmd == 'predict_distribution':
                    stash.add(0, args['X_position'], ri)
                    # log("Predict distribution request completed!")
                elif cmd == 'predict_winrate':
                    stash.add(1, args['X_position'], ri)
                    # log("Predict win-rate request completed!")
                elif cmd == 'model_name':
                    self.res_queues[ri].put(net.model_name)
                    log("Model name request completed!")
                elif cmd == 'save':
                    self.status.value = b'SAVING'
                    stash.process()
                    snapshot_id = args['snapshot_id']
                    net.save(snapshot_id)
                    log("Model save request completed!")
                    self.status.value = b'READY'

            log("Model Server process completed!")
        except:
            import traceback
            traceback.print_exc()


class GoModel(object):
    def __init__(self, board_size, load_snapshot=None):

        self.board_size = board_size
        self.cmd_queue = Queue()
        self.res_queues = [Queue() for i in range(128)]

        self.status = Array('c', 32)
        self.stash_size = Value('i', 1)
        self.server = ModelServer(self.board_size, self.cmd_queue, self.res_queues, status=self.status,
                                  stash_size=self.stash_size, load_snapshot=load_snapshot)
        self.server.start()

        self.ri = 0  # id of process in case of multiple processes, to prevent mix ups

    def is_model_ready(self):
        try:
            while True:
                if self.server.status.value == b'READY':
                    break
                time.sleep(1.0)
                continue
            return True
        except:
            import traceback
            traceback.print_exc()
            return False

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

    def process_stash(self):
        self.cmd_queue.put(('stash_process', {}, self.ri))

    def set_stash_size(self, stash_size):
        self.stash_size.value = stash_size
        self.cmd_queue.put(('stash_size', {'stash_size': stash_size}, self.ri))

    def fit_game(self, positions, result, board_transform=None):
        X_positions = [(self.encode_position(pos, board_transform=board_transform), dist) for pos, dist in positions]
        self.cmd_queue.put(('fit_game', {'X_positions': X_positions, 'result': result}, self.ri))

    def predict_distribution(self, position):
        self.cmd_queue.put(('predict_distribution', {'X_position': self.encode_position(position)}, self.ri))

        try:
            result = self.res_queues[self.ri].get(timeout=self.stash_size.value * 2)
        except:
            print("predict_distribution queue get timed out")
            self.process_stash()
            result = self.res_queues[self.ri].get(timeout=self.stash_size.value * 3)

        return result

    def predict_winrate(self, position):
        self.cmd_queue.put(('predict_winrate', {'X_position': self.encode_position(position)}, self.ri))

        try:
            result = self.res_queues[self.ri].get(timeout=self.stash_size.value * 2)
        except:
            print("predict_winrate queue get timed out")
            self.process_stash()
            result = self.res_queues[self.ri].get(timeout=self.stash_size.value * 3)

        return result

    def model_name(self):
        self.cmd_queue.put(('model_name', {}, self.ri))
        return self.res_queues[self.ri].get()

    def save(self, snapshot_id):
        self.cmd_queue.put(('save', {'snapshot_id': snapshot_id}, self.ri))
