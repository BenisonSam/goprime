import datetime
import logging
import os
import shutil
import sys
import time
import traceback
from multiprocessing import Process, Queue
from multiprocessing.sharedctypes import Array, Value

import numpy as np
from psutil import virtual_memory


def log(msg, model_name):
    if not os.path.exists("logs/"):
        os.makedirs("logs/")

    # set basic polling config
    logging.basicConfig(
        format="%(message)s",
        level=logging.DEBUG,
        # handlers=[logging.StreamHandler(sys.stdout)],
        filename="logs/ModelServer-{0}.logs".format(model_name), filemode="w"
    )

    now = datetime.datetime.now()
    logging.info("{0}: {1}".format(now.strftime("%Y-%m-%d %H:%M:%S"), msg))


class ModelServer(Process):
    def __init__(self, board_size, cmd_queue, res_queues, status,
                 batch_size=32, stash_size=1, retrain_after=5e4, load_snapshot=None):
        super(ModelServer, self).__init__()

        self.model_name = ""
        self.board_size = board_size
        self.cmd_queue = cmd_queue
        self.res_queues = res_queues
        self.load_snapshot = load_snapshot

        self.status = status
        self.stash_size = stash_size
        self.batch_size = batch_size
        self.status.value = b'INITIALIZED'
        self.retrain_after = int(retrain_after)

    def run(self):
        try:
            from goprime.net import AGZeroModel

            N = self.board_size
            start_time = time.time()

            net = AGZeroModel(N, batch_size=self.batch_size,
                              archive_fit_samples=int(np.ma.ceil(np.power(N, 3) / 8) * 8))
            self.model_name = net.model_name
            log("Model Server initialized!", self.model_name)

            net.create()

            if self.load_snapshot is not None:
                if isinstance(self.load_snapshot, list):
                    net.load_averaged(self.load_snapshot, log)
                else:
                    net.load(self.load_snapshot)
                log("Snapshot loaded!", self.model_name)

            log("Neural Network Initialized!", self.model_name)

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

            log("Predict Stash Initialized!", self.model_name)

            self.status.value = b'READY'

            fit_counter = 0
            finished = False
            snapshot_id = "weights/{0}_FS".format(self.model_name)

            while True and not finished:

                if virtual_memory().percent > 85.0:
                    net.save(snapshot_id=snapshot_id.replace('weights', 'weights/archive'), save_archive=True)
                    net.reduce_position_archive(ratio=0.2)

                if time.time() - start_time >= 85500:
                    # save emergency snapshot
                    snapshot_id = "{0}_last".format(snapshot_id)
                    net.save(snapshot_id, save_archive=True)
                    log("Emergency snapshot saved before program termination!", self.model_name)

                    # submit the next job
                    try:
                        weights_dir = "weights"
                        archive_dir = os.path.join(weights_dir, "archive")

                        if not os.path.exists(archive_dir):
                            os.mkdir(archive_dir)

                        # move all the pos archives to archive folder
                        for file in os.listdir(weights_dir):
                            if file.endswith(".joblib"):
                                shutil.move(os.path.join(weights_dir, file), os.path.join(archive_dir, file))

                        content = None
                        with open("sbatch_goprime.sh") as f:
                            content = f.read()

                        if content is not None:
                            content = content.replace('{CWD}', os.getcwd())
                            content = content.replace('{N}', str(N))
                            content = content.replace('{SNAPSHOT}', snapshot_id)

                            script = "{0}.sh".format(net.model_name)
                            with open(script, "w") as f:
                                f.write(content)

                            os.system("sbatch {0}".format(script))
                    except:
                        traceback.print_exc()

                    finished = True
                    continue

                cmd, args, ri = self.cmd_queue.get()

                if cmd == 'stash_size':
                    stash.process()
                    stash.trigger = args['stash_size']
                    self.stash_size.value = args['stash_size']
                    log("Change stash size request completed!", self.model_name)
                elif cmd == 'stash_process':
                    stash.process()
                    log("Process stash request completed!", self.model_name)
                elif cmd == 'fit_game':
                    self.status.value = b'FIT_STARTED'
                    stash.process()
                    log("Fit {0} started...".format(fit_counter), self.model_name)

                    net.fit_game(**args)

                    log("Fit {0} completed...".format(fit_counter), self.model_name)

                    fit_counter += 1
                    self.status.value = b'FIT_COMPLETED'

                    # if fit_counter != 1 and fit_counter % self.retrain_after == 1:
                    #     self.cmd_queue.put(('retrain', {'batch_size': self.batch_size}, 0))
                elif cmd == 'retrain':
                    self.status.value = b'RETRAINING'
                    log("Retrain model started...", self.model_name)

                    loaded = True
                    if 'archive' in args:
                        loaded = net.load_pos_archive(args['archive'])

                    if loaded:
                        if 'batch_size' in args:
                            batch_size = int(args['batch_size'])
                        else:
                            batch_size = self.batch_size

                        net.retrain_position_archive(batch_size=batch_size)

                        if 'snapshot_id' in args:
                            snapshot_id = "{0}_retrained".format(args['snapshot_id'])
                            net.save(snapshot_id)

                    log("Retrain model completed...", self.model_name)
                elif cmd == 'reduce_position_archive':
                    ratio = 0.5
                    if args['ratio']:
                        ratio = float(args['ratio'])
                    net.reduce_position_archive(ratio=ratio)
                    log("Position Archive reduce request completed!", self.model_name)
                elif cmd == 'unload_position_archive':
                    net.unload_pos_archive()
                elif cmd == 'predict_distribution':
                    stash.add(0, args['X_position'], ri)
                    # log("Predict distribution request completed!")
                elif cmd == 'predict_winrate':
                    stash.add(1, args['X_position'], ri)
                    # log("Predict win-rate request completed!")
                elif cmd == 'model_name':
                    self.res_queues[ri].put(net.model_name)
                    log("Model name request completed!", self.model_name)
                elif cmd == 'save':
                    self.status.value = b'SAVING'
                    stash.process()
                    snapshot_id = args['snapshot_id']
                    save_archive = False if 'archive' not in args else args['archive']
                    net.save(snapshot_id, save_archive)
                    log("Model save request completed!", self.model_name)
                    self.status.value = b'READY'
                elif cmd == 'stop_server':
                    if self.cmd_queue.empty():
                        self.status.value = b'READY'
                        finished = True

                sys.stdout.flush()

            log("Model Server process completed!", self.model_name)
        except:
            traceback.print_exc()


class GoModel(object):
    def __init__(self, board_size, batch_size=32, load_snapshot=None):

        self.board_size = board_size
        self.cmd_queue = Queue()
        self.res_queues = [Queue() for i in range(256)]

        self.status = Array('c', 32)
        self.stash_size = Value('i', 1)
        self.server = ModelServer(self.board_size, self.cmd_queue, self.res_queues, status=self.status,
                                  batch_size=batch_size, stash_size=self.stash_size, load_snapshot=load_snapshot)
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

    def set_waiting_status(self):
        self.status.value = b'WAITING'

    def reduce_position_archive(self, ratio=0.5):
        self.cmd_queue.put(('reduce_position_archive', {'ratio': ratio}, 0))

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

    def save(self, snapshot_id, save_archive=False):
        self.cmd_queue.put(('save', {'snapshot_id': snapshot_id, 'archive': save_archive}, self.ri))

    def retrain(self, batch_size, archive=None, save_after=10):

        snapshot = "weights/{0}".format(self.model_name())

        if archive is None:
            self.cmd_queue.put(('retrain', {'snapshot_id': snapshot, 'batch_size': batch_size}, 0))
        elif isinstance(archive, list):
            i = 1
            for each_archive in archive:
                self.cmd_queue.put(('retrain', {'batch_size': batch_size, 'archive': each_archive}, 0))
                self.cmd_queue.put(('unload_position_archive', {}, 0))

                if i % save_after == 0:
                    self.save("{0}_{1}".format(snapshot, i)),

                i += 1

        snapshot = snapshot if snapshot is not None else self.model_name()
        self.save(snapshot_id=snapshot)

        self.cmd_queue.put(('stop_server', {}, 0))
        if self.is_model_ready():
            pass
