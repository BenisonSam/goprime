from __future__ import division

from twisted.python.formmethod import InputError


class Elo:
    """
    A class to calculate the elo rating of both players in given order
    """

    def __init__(self):
        self.player_names = ["Player 1", "Player 2"]
        self.players = {}
        self.score = 0
        self._i = _Implementation()

    def add_init_elo_ratings(self, elo_rating, opponent_elo_rating, result):
        assert elo_rating is not float
        assert opponent_elo_rating is not float
        assert result is not str

        result = result.upper()
        assert result in ['WIN', 'LOST', 'DRAW']

        self.score = 1.0 if result == 'WIN' else (0.0 if result == 'LOST' else 0.5)
        self.players[self.player_names[0]] = elo_rating
        self.players[self.player_names[1]] = opponent_elo_rating

        for player, rating in self.players.items():
            self._i.add_player(player, rating)

        if result == 'WIN':
            self._i.record_match(self.player_names[0], self.player_names[1], winner=self.player_names[0])
        elif result == 'LOST':
            self._i.record_match(self.player_names[0], self.player_names[1], winner=self.player_names[1])
        else:
            self._i.record_match(self.player_names[0], self.player_names[1], draw=True)

    def get_final_elo_ratings(self, k=32):
        elo1 = []
        elo2 = []

        exp1 = Elo.__expected(self.players[self.player_names[0]], self.players[self.player_names[1]])
        exp2 = Elo.__expected(self.players[self.player_names[1]], self.players[self.player_names[0]])

        elo1.append(Elo.__elo(self.players[self.player_names[0]], exp1, self.score, k))
        elo2.append(Elo.__elo(self.players[self.player_names[1]], exp2, 1 - self.score, k))

        for rating in self._i.get_rating_list():
            if rating[0] == self.player_names[0]:
                elo1.append(rating[1])
            elif rating[0] == self.player_names[1]:
                elo2.append(rating[1])

        return min(elo1), min(elo2)

    @staticmethod
    def __expected(a, b):
        """
        Calculate expected score of A in a match against B
        """
        return 1 / (1 + 10 ** ((b - a) / 400))

    @staticmethod
    def __elo(old, exp, score, k=32):
        """
        Calculate the new Elo rating for a player
        """
        return old + k * (score - exp)


class _Implementation:
    """
    A class that represents an implementation of the Elo Rating System
    """

    def __init__(self, base_rating=1000):
        """
        Runs at initialization of class object.
        """
        self.base_rating = base_rating
        self.players = []

    def __get_player_list(self):
        """
        Returns this implementation's player list.
        """
        return self.players

    def get_player(self, name):
        """
        Returns the player in the implementation with the given name.
        """
        for player in self.players:
            if player.name == name:
                return player
        return None

    def contains(self, name):
        """
        Returns true if this object contains a player with the given name.
        Otherwise returns false.
        """
        for player in self.players:
            if player.name == name:
                return True
        return False

    def add_player(self, name, rating=None):
        """
        Adds a new player to the implementation.
        """
        if rating is None:
            rating = self.base_rating

        self.players.append(_Player(name=name, rating=rating))

    def remove_player(self, name):
        """
        Adds a new player to the implementation.
        """
        self.__get_player_list().remove(self.get_player(name))

    def record_match(self, name1, name2, winner=None, draw=False, k=32):
        """
        Should be called after a game is played.
        """
        player1 = self.get_player(name1)
        player2 = self.get_player(name2)

        expected1 = player1.compare_rating(player2)
        expected2 = player2.compare_rating(player1)

        k = len(self.__get_player_list()) * (k / 2)

        rating1 = player1.rating
        rating2 = player2.rating

        if draw:
            score1 = 0.5
            score2 = 0.5
        elif winner == name1:
            score1 = 1.0
            score2 = 0.0
        elif winner == name2:
            score1 = 0.0
            score2 = 1.0
        else:
            raise InputError('One of the names must be the winner or draw must be True')

        new_rating_1 = rating1 + k * (score1 - expected1)
        new_rating_2 = rating2 + k * (score2 - expected2)

        if new_rating_1 < 0:
            new_rating_1 = 0
            new_rating_2 = rating2 - rating1

        elif new_rating_2 < 0:
            new_rating_2 = 0
            new_rating_1 = rating1 - rating2

        player1.rating = new_rating_1
        player2.rating = new_rating_2

    def get_player_rating(self, name):
        """
        Returns the rating of the player with the given name.
        :return The rating of the player with the given name.
        """
        player = self.get_player(name)
        return player.rating

    def get_rating_list(self):
        """
        Returns a list of tuples in the form of ({name},{rating})
        :return The list of tuples
        """
        lst = []
        for player in self.__get_player_list():
            lst.append((player.name, player.rating))
        return lst


class _Player:
    """
    A class to represent a player in the Elo Rating System
    """

    def __init__(self, name, rating):
        """
        Runs at initialization of class object.
        """
        self.name = name
        self.rating = rating

    def compare_rating(self, opponent):
        """
        Compares the two ratings of the this player and the opponent.
        :returns The expected score between the two players.
        """
        return (1 + 10 ** ((opponent.rating - self.rating) / 400.0)) ** -1
