import math

import numpy as np

from goprime import *
from goprime.board import *


class TreeNode:
    """ Monte-Carlo tree node;
    v is #visits, w is #wins for to-play (expected reward is w/v)
    pv, pw are prior values (node value = w/v + pw/pv)
    av, aw are amaf values ("all moves as first", used for the RAVE tree policy)
    children is None for leaf nodes """

    def __init__(self, net, pos):
        self.net = net
        self.pos = pos
        self.v = 0
        self.w = 0
        self.pv = 0
        self.pw = 0
        self.av = 0
        self.aw = 0
        self.children = None

    def expand(self):
        """ add and initialize children to a leaf node """

        # noinspection PyTypeChecker
        N = int(Board.N)
        W = int(Board.W)

        distribution = self.net.predict_distribution(self.pos)
        self.children = []

        for c in self.pos.moves(0):
            pos2 = self.pos.move(c)
            if pos2 is None:
                continue
            node = TreeNode(self.net, pos2)
            self.children.append(node)
            x, y = c % W - 1, c // W - 1
            value = distribution[y * N + x]

            node.pv = PRIOR_NET
            node.pw = PRIOR_NET * value

        # Add also a pass move - but only if this doesn't trigger a losing
        # scoring (or we have no other option)
        if not self.children:
            can_pass = True
        else:
            can_pass = self.pos.score() >= 0

        if can_pass:
            node = TreeNode(self.net, self.pos.pass_move())
            self.children.append(node)
            node.pv = PRIOR_NET
            node.pw = PRIOR_NET * distribution[-1]

    def puct_urgency(self, n0):
        # XXX: This is substituted by global_puct_urgency()
        expectation = float(self.w + PRIOR_EVEN / 2) / (self.v + PRIOR_EVEN)

        try:
            prior = float(self.pw) / self.pv
        except:
            prior = 0.1  # XXX

        return expectation + PUCT_C * prior * math.sqrt(n0) / (1 + self.v)

    def rave_urgency(self):
        v = self.v + self.pv
        expectation = float(self.w + self.pw) / v

        if self.av == 0:
            return expectation

        rave_expectation = float(self.aw) / self.av
        beta = self.av / (self.av + v + float(v) * self.av / RAVE_EQUIV)

        return beta * rave_expectation + (1 - beta) * expectation

    def winrate(self):
        return float(self.w) / self.v if self.v > 0 else float('nan')

    def prior(self):
        return float(self.pw) / self.pv if self.pv > 0 else float('nan')

    def best_move(self, proportional=False):
        """ best move is the most simulated one """
        if self.children is None:
            return None

        if proportional:
            probs = [(float(node.v) / self.v) ** TEMPERATURE for node in self.children]
            probs_tot = sum(probs)
            probs = [p / probs_tot for p in probs]
            # print([(Position.str_coord(n.pos.last), p, p * probs_tot) for n, p in zip(self.children, probs)])
            i = np.random.choice(len(self.children), p=probs)

            return self.children[i]
        else:
            return max(self.children, key=lambda node: node.v)

    def distribution(self):
        # noinspection PyTypeChecker
        N = int(Board.N)
        W = int(Board.W)

        distribution = np.zeros(N * N + 1)

        for child in self.children:
            p = float(child.v) / self.v
            c = child.pos.last
            if c is not None:
                x, y = c % W - 1, c // W - 1
                distribution[y * N + x] = p
            else:
                distribution[-1] = p
        return distribution


class MCTree:

    @staticmethod
    def puct_urgency_input(nodes):
        w = np.array([float(n.w) for n in nodes])
        v = np.array([float(n.v) for n in nodes])
        pw = np.array([float(n.pw) if n.pv > 0 else 1. for n in nodes])
        pv = np.array([float(n.pv) if n.pv > 0 else 10. for n in nodes])

        return w, v, pw, pv

    @staticmethod
    def global_puct_urgency(n0, w, v, pw, pv):
        # Like Node.puct_urgency(), but for all children, more quickly.
        # Expects numpy arrays (except n0 which is scalar).
        expectation = (w + PRIOR_EVEN / 2) / (v + PRIOR_EVEN)
        prior = pw / pv

        return expectation + PUCT_C * prior * math.sqrt(n0) / (1 + v)

    @staticmethod
    def tree_descend(tree, amaf_map, disp=False):
        """ Descend through the tree to a leaf """
        tree.v += 1
        nodes = [tree]
        passes = 0
        root = True

        while nodes[-1].children is not None and passes < 2:
            if disp: Position.print(nodes[-1].pos)

            # Pick the most urgent child
            children = list(nodes[-1].children)
            if disp:
                for c in children:
                    MCTree.dump_subtree(c, recurse=False)

            random.shuffle(children)  # randomize the max in case of equal urgency
            urgencies = MCTree.global_puct_urgency(nodes[-1].v, *MCTree.puct_urgency_input(children))

            if root:
                dirichlet = np.random.dirichlet((0.03, 1), len(children))
                urgencies = urgencies * 0.75 + dirichlet[:, 0] * 0.25
                root = False

            node = max(zip(children, urgencies), key=lambda t: t[1])[0]
            nodes.append(node)

            if disp: print('chosen %s' % (Position.str_coord(node.pos.last),), file=sys.stderr)

            if node.pos.last is None:
                passes += 1
            else:
                passes = 0
                if amaf_map[node.pos.last] == 0:  # Mark the coordinate with 1 for black
                    amaf_map[node.pos.last] = 1 if nodes[-2].pos.n % 2 == 0 else -1

            # updating visits on the way *down* represents "virtual loss", relevant for parallelization
            node.v += 1
            if node.children is None and node.v > EXPAND_VISITS:
                node.expand()

        return nodes

    @staticmethod
    def tree_update(nodes, amaf_map, score, disp=False):
        """ Store simulation result in the tree (@nodes is the tree path) """
        for node in reversed(nodes):
            if disp:  print('updating', Position.str_coord(node.pos.last), score < 0, file=sys.stderr)
            node.w += score < 0  # score is for to-play, node statistics for just-played
            # Update the node children AMAF stats with moves we made
            # with their color
            amaf_map_value = 1 if node.pos.n % 2 == 0 else -1

            if node.children is not None:
                for child in node.children:
                    if child.pos.last is None:
                        continue

                    if amaf_map[child.pos.last] == amaf_map_value:
                        if disp: print('  AMAF updating', Position.str_coord(child.pos.last), score > 0,
                                       file=sys.stderr)
                        child.aw += score > 0  # reversed perspective
                        child.av += 1

            score = -score

    @staticmethod
    def tree_search(tree, n, owner_map, output_stream=None, debug_disp=False):
        """ Perform MCTS search from a given position for a given #iterations """
        W = int(Board.W)

        # Initialize root node
        if tree.children is None:
            tree.expand()

        i = 0
        while i < n:
            amaf_map = W * W * [0]
            nodes = MCTree.tree_descend(tree, amaf_map, disp=debug_disp)

            i += 1
            if output_stream is not None and i % REPORT_PERIOD == 0:
                MCTree.print_tree_summary(tree, i, f=output_stream)

            last_node = nodes[-1]
            if last_node.pos.last is None and last_node.pos.last2 is None:
                score = 1 if last_node.pos.score() > 0 else -1
            else:
                score = tree.net.predict_winrate(last_node.pos)

            MCTree.tree_update(nodes, amaf_map, score, disp=debug_disp)

        if output_stream is not None:
            MCTree.dump_subtree(tree, f=output_stream)
        if output_stream is not None and i % REPORT_PERIOD != 0:
            MCTree.print_tree_summary(tree, i, f=output_stream)

        return tree.best_move(tree.pos.n <= PROPORTIONAL_STAGE)

    @staticmethod
    def dump_subtree(node, thres=N_SIMS / 50, indent=0, f=sys.stderr, recurse=True):
        """ print this node and all its children with v >= thres. """
        print("%s+- %s %.3f (%d/%d, prior %d/%d, rave %d/%d=%.3f, pred %.3f)" %
              (indent * ' ', Position.str_coord(node.pos.last), node.winrate(),
               node.w, node.v, node.pw, node.pv, node.aw, node.av,
               float(node.aw) / node.av if node.av > 0 else float('nan'),
               float(-node.net.predict_winrate(node.pos) + 1) / 2), file=f)

        if not recurse or not node.children:
            return

        for child in sorted(node.children, key=lambda n: n.v, reverse=True):
            if child.v >= thres:
                MCTree.dump_subtree(child, thres=thres, indent=indent + 3, f=f)

    @staticmethod
    def print_tree_summary(tree, sims, f=sys.stderr):
        best_nodes = sorted(tree.children, key=lambda n: n.v, reverse=True)[:5]
        best_seq = []
        node = tree

        while node is not None:
            best_seq.append(node.pos.last)
            node = node.best_move()

        best_predwinrate = float(-tree.net.predict_winrate(best_nodes[0].pos) + 1) / 2

        print('[%4d] winrate %.3f/%.3f | seq %s | can %s' %
              (sims, best_nodes[0].winrate(), best_predwinrate, ' '
               .join([Position.str_coord(c) for c in best_seq[1:6]]),
               ' '.join(['%s(%.3f|%d/%.3f)' % (Position.str_coord(n.pos.last), n.winrate(),
                                               n.v, n.prior()) for n in best_nodes])), file=f)

    @staticmethod
    def position_dist(N, net, worker_id, pos, disp=False):

        W = N + 2

        net.ri = worker_id

        tree = TreeNode(net=net, pos=pos)
        tree.expand()

        owner_map = W * W * [0]
        MCTree.tree_search(tree, N_SIMS, owner_map, output_stream=sys.stdout if disp else None)

        return tree.distribution()

    @staticmethod
    def position_distnext(N, pos):
        W = N + 2
        distribution = np.zeros(N * N + 1)

        c = pos.data['next']
        if c is not None:
            x, y = c % W - 1, c // W - 1
            distribution[y * N + x] = 1
        else:
            distribution[-1] = 1

        return distribution
