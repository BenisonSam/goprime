import random
import re
import sys
from collections import namedtuple
from itertools import count

from goprime.common import *


@static_class
class Board(object):
    __N = None
    __W = None

    komi = 7.5

    def __init__(self):
        raise TypeError("Board: Is a Static class. Object not allowed to be created!")

    @static_property
    def N(self):
        if self.__N is not None:
            return Board.__N
        else:
            raise ValueError("Board: The Board Size is not set")

    @N.getter
    def N(self):
        if self.__N is not None:
            return Board.__N
        else:
            raise ValueError("Board: The Board Size is not set")

    @N.setter
    def N(self, board_size):
        Board.__N = board_size
        Board.__W = board_size + 2

    @static_property
    def W(self):
        return Board.__W

    @static_property
    def empty(self):
        N = Board.__N
        return "\n".join([(N + 1) * ' '] + N * [' ' + N * '.'] + [(N + 2) * ' '])

    @static_property
    def contact_res(self):
        W = Board.__W

        # Regex that matches various kind of points adjacent to '#' (flood filled) points
        contact_res = dict()

        for p in ['.', 'x', 'X']:
            rp = '\\.' if p == '.' else p
            contact_res_src = ['#' + rp,  # p at right
                               rp + '#',  # p at left
                               '#' + '.' * (W - 1) + rp,  # p below
                               rp + '.' * (W - 1) + '#']  # p above
            contact_res[p] = re.compile('|'.join(contact_res_src), flags=re.DOTALL)

        return contact_res

    @staticmethod
    def neighbors(c):
        """ generator of coordinates for all neighbors of c """
        W = Board.__W
        return [c - 1, c + 1, c - W, c + W]

    @staticmethod
    def diag_neighbors(c):
        """ generator of coordinates for all diagonal neighbors of c """
        W = Board.__W
        return [c - W - 1, c - W + 1, c + W - 1, c + W + 1]

    @staticmethod
    def board_put(board, c, p):
        return board[:c] + p + board[c + 1:]

    @staticmethod
    def flood_fill(board, c):
        """ replace continuous-color area starting at c with special color # """

        # This is called so much that a bytearray is worthwhile...
        byte_board = bytearray(board, 'utf-8')

        bb_c = byte_board[c]

        byte_board[c] = ord('#')
        fringe = [c]

        while fringe:
            c = fringe.pop()
            for d in Board.neighbors(c):
                if byte_board[d] == bb_c:
                    byte_board[d] = ord('#')
                    fringe.append(d)

        return byte_board.decode('utf-8')

    @staticmethod
    def contact(board, p):
        """ test if point of color p is adjacent to color # anywhere
        on the board; use in conjunction with floodfill for reachability """
        m = Board.contact_res[p].search(board)
        if not m:
            return None
        return m.start() if m.group(0)[0] == p else m.end() - 1

    @staticmethod
    def is_eyeish(board, c):
        """ test if c is inside a single-color diamond and return the diamond
        color or None; this could be an eye, but also a false one """
        eye_color = other_color = None

        for d in Board.neighbors(c):
            if board[d].isspace():
                continue
            if board[d] == '.':
                return None
            if eye_color is None:
                eye_color = board[d]
                other_color = eye_color.swapcase()
            elif board[d] == other_color:
                return None

        return eye_color

    @staticmethod
    def is_eye(board, c):
        """ test if c is an eye and return its color or None """
        eye_color = Board.is_eyeish(board, c)
        if eye_color is None:
            return None

        # Eye-like shape, but it could be a falsified eye
        false_color = eye_color.swapcase()
        false_count = 0
        at_edge = False

        for d in Board.diag_neighbors(c):
            if board[d].isspace():
                at_edge = True
            elif board[d] == false_color:
                false_count += 1

        if at_edge:
            false_count += 1
        if false_count >= 2:
            return None

        return eye_color


class Position(namedtuple('Position', 'board cap n ko last last2 komi data')):
    """ Implementation of simple Chinese Go rules;
    n is how many moves were played so far """

    col_str = "ABCDEFGHJKLMNOPQRST"

    @staticmethod
    def empty_position(board_size):
        """ Return an initial board position """

        Board.N = board_size

        return Position(board=Board.empty, cap=(0, 0), n=0,
                        ko=set(), last=None, last2=None, komi=Board.komi, data=dict())

    def move(self, c):
        """ play as player X at the given coord c, return the new position """

        # Are we trying to play in enemy's eye?
        in_enemy_eye = Board.is_eyeish(self.board, c) == 'x'

        board = Board.board_put(self.board, c, 'X')
        # Test for captures, and track ko
        capX = self.cap[0]
        singlecaps = []
        for d in Board.neighbors(c):
            if board[d] != 'x':
                continue
            # XXX: The following is an extremely naive and SLOW approach
            # at things - to do it properly, we should maintain some per-group
            # data structures tracking liberties.
            fboard = Board.flood_fill(board, d)  # get a board with the adjacent group replaced by '#'
            if Board.contact(fboard, '.') is not None:
                continue  # some liberties left
            # no liberties left for this group, remove the stones!
            capcount = fboard.count('#')
            if capcount == 1:
                singlecaps.append(d)
            capX += capcount
            board = fboard.replace('#', '.')  # capture the group
        # Test for suicide
        if Board.contact(Board.flood_fill(board, c), '.') is None:
            return None

        # Test for (positional super)ko
        if board in self.ko or board.swapcase() in self.ko:
            return None

        # Update the position and return
        return Position(board=board.swapcase(), cap=(self.cap[1], capX),
                        n=self.n + 1, ko=self.ko | {board}, last=c, last2=self.last, komi=self.komi, data=dict())

    def pass_move(self):
        """ pass - i.e. return simply a flipped position """
        return Position(board=self.board.swapcase(), cap=(self.cap[1], self.cap[0]),
                        n=self.n + 1, ko=self.ko, last=None, last2=self.last, komi=self.komi, data=dict())

    def moves(self, i0):
        """ Generate a list of moves (includes false positives - suicide moves;
        does not include true-eye-filling moves), starting from a given board
        index (that can be used for randomization) """
        i = i0 - 1
        passes = 0
        while True:
            i = self.board.find('.', i + 1)
            if passes > 0 and (i == -1 or i >= i0):
                break  # we have looked through the whole board
            elif i == -1:
                i = 0
                passes += 1
                continue  # go back and start from the beginning
            # Test for to-play player's one-point eye
            if Board.is_eye(self.board, i) == 'X':
                continue
            yield i

    def last_moves_neighbors(self):
        """ generate a randomly shuffled list of points including and
        surrounding the last two moves (but with the last move having
        priority) """
        clist = []
        for c in self.last, self.last2:
            if c is None:  continue
            dlist = [c] + list(Board.neighbors(c) + Board.diag_neighbors(c))
            Board.random.shuffle(dlist)
            clist += [d for d in dlist if d not in clist]
        return clist

    def score(self, owner_map=None):
        """ compute score for to-play player; this assumes a final position
        with all dead stones captured; if owner_map is passed, it is assumed
        to be an array of statistics with average owner at the end of the game
        (+1 black, -1 white) """
        W = int(Board.W)
        board = self.board
        i = 0

        while True:
            i = self.board.find('.', i + 1)
            if i == -1:
                break
            fboard = Board.flood_fill(board, i)
            # fboard is board with some continuous area of empty space replaced by #
            touches_X = Board.contact(fboard, 'X') is not None
            touches_x = Board.contact(fboard, 'x') is not None
            if touches_X and not touches_x:
                board = fboard.replace('#', 'X')
            elif touches_x and not touches_X:
                board = fboard.replace('#', 'x')
            else:
                board = fboard.replace('#', ':')  # seki, rare
            # now that area is replaced either by X, x or :

        komi = self.komi if self.n % 2 == 1 else -self.komi

        if owner_map is not None:
            for c in range(W * W):
                n = 1 if board[c] == 'X' else -1 if board[c] == 'x' else 0
                owner_map[c] += n * (1 if self.n % 2 == 0 else -1)

        return board.count('X') - board.count('x') + komi

    def flip_vert(self):
        W = int(Board.W)
        board = '\n'.join(reversed(self.board[:-1].split('\n'))) + ' '

        def coord_flip_vert(c):
            if c is None:  return None
            return (W - 1 - c // W) * W + c % W

        # XXX: Doesn't update ko properly
        return Position(board=board, cap=self.cap, n=self.n, ko=set(), last=coord_flip_vert(self.last),
                        last2=coord_flip_vert(self.last2), komi=self.komi, data=self.data)

    def flip_horiz(self):
        W = int(Board.W)
        board = '\n'.join([' ' + l[1:][::-1] for l in self.board.split('\n')])

        def coord_flip_horiz(c):
            if c is None:  return None
            return c // W * W + (W - 1 - c % W)

        # XXX: Doesn't update ko properly
        return Position(board=board, cap=self.cap, n=self.n, ko=set(), last=coord_flip_horiz(self.last),
                        last2=coord_flip_horiz(self.last2), komi=self.komi, data=self.data)

    def flip_both(self):
        W = int(Board.W)
        board = '\n'.join(reversed([' ' + l[1:][::-1] for l in self.board[:-1].split('\n')])) + ' '

        def coord_flip_both(c):
            if c is None:  return None
            return (W - 1 - c // W) * W + (W - 1 - c % W)

        # XXX: Doesn't update ko properly
        return Position(board=board, cap=self.cap, n=self.n, ko=set(), last=coord_flip_both(self.last),
                        last2=coord_flip_both(self.last2), komi=self.komi, data=self.data)

    def flip_random(self):
        pos = self

        if random.random() < 0.5:
            pos = pos.flip_vert()
        if random.random() < 0.5:
            pos = pos.flip_horiz()

        return pos

    @staticmethod
    def print(pos, f=sys.stderr, owner_map=None):
        """ print visualization of the given board position, optionally also
        including an owner map statistic (probability of that area of board
        eventually becoming black/white) """

        # noinspection PyTypeChecker
        N = int(Board.N)
        W = int(Board.W)

        if pos.n % 2 == 0:  # to-play is black
            board = pos.board.replace('x', 'O')
            Xcap, Ocap = pos.cap
        else:  # to-play is white
            board = pos.board.replace('X', 'O').replace('x', 'X')
            Ocap, Xcap = pos.cap
        print('Move: %-3d   Black: %d caps   White: %d caps  Komi: %.1f' % (pos.n, Xcap, Ocap, pos.komi), file=f)
        pretty_board = ' '.join(board.rstrip()) + ' '
        if pos.last is not None:
            pretty_board = pretty_board[:pos.last * 2 - 1] + '(' + board[pos.last] + ')' + pretty_board[
                                                                                           pos.last * 2 + 2:]
        rowcounter = count()
        pretty_board = [' %-02d%s' % (N - i, row[2:]) for row, i in zip(pretty_board.split("\n")[1:], rowcounter)]
        if owner_map is not None:
            pretty_ownermap = ''
            for c in range(W * W):
                if board[c].isspace():
                    pretty_ownermap += board[c]
                elif owner_map[c] > 0.6:
                    pretty_ownermap += 'X'
                elif owner_map[c] > 0.3:
                    pretty_ownermap += 'x'
                elif owner_map[c] < -0.6:
                    pretty_ownermap += 'O'
                elif owner_map[c] < -0.3:
                    pretty_ownermap += 'o'
                else:
                    pretty_ownermap += '.'
            pretty_ownermap = ' '.join(pretty_ownermap.rstrip())
            pretty_board = ['%s   %s' % (brow, orow[2:]) for brow, orow in
                            zip(pretty_board, pretty_ownermap.split("\n")[1:])]
        print("\n".join(pretty_board), file=f)
        print('    ' + ' '.join(Position.col_str[:N]), file=f)
        print('', file=f)

    @staticmethod
    def parse_coord(s):
        # noinspection PyTypeChecker
        N = int(Board.N)
        W = int(Board.W)

        s = str(s).strip()

        if s == 'pass':
            return None

        return W + 1 + (N - int(s[1:])) * W + Position.col_str.index(s[0].upper())

    @staticmethod
    def str_coord(c):
        # noinspection PyTypeChecker
        N = int(Board.N)
        W = int(Board.W)

        if c is None:
            return 'pass'
        row, col = divmod(c - (W + 1), W)
        return '%c%d' % (Position.col_str[col], N - row)
