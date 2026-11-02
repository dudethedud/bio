"""Measure the move-selection strategy on a faithful TriPeaks simulator.

This isolates the *strategy* (which legal move to pick) from the vision layer.
It plays classic TriPeaks -- the base mechanic of Disney Solitaire, minus
wilds/power-ups -- for many random deals and reports win rates for:

  * random  : pick a random legal move, else draw  (a "no strategy" baseline)
  * tool    : the tool's best_move rule -- play the free card that uncovers the
              most other cards, tie-break toward the lower rank, else draw

A win = every one of the 28 tableau cards cleared before the stock runs out.

Run:  python benchmark/simulate.py --deals 20000
"""

import argparse
import random

RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]

# Standard 28-card TriPeaks tableau: rows of 3, 6, 9, 10. For each card, the two
# cards in the row below that must both be gone before it can be played.
CHILDREN = {
    0: [3, 4], 1: [5, 6], 2: [7, 8],
    3: [9, 10], 4: [10, 11], 5: [12, 13], 6: [13, 14], 7: [15, 16], 8: [16, 17],
    9: [18, 19], 10: [19, 20], 11: [20, 21], 12: [21, 22], 13: [22, 23],
    14: [23, 24], 15: [24, 25], 16: [25, 26], 17: [26, 27],
}
PARENTS = {}
for _p, _kids in CHILDREN.items():
    for _k in _kids:
        PARENTS.setdefault(_k, []).append(_p)


def adjacent(a, b):
    d = abs(RANKS.index(a) - RANKS.index(b))
    return d == 1 or d == len(RANKS) - 1


def new_deal(rng):
    deck = [r for r in RANKS for _ in range(4)]  # ranks only; suits don't matter here
    rng.shuffle(deck)
    tableau = {i: deck[i] for i in range(28)}  # position -> rank (absent = removed)
    stock = deck[28:]                          # 24 cards
    waste = stock.pop()                        # one card starts face up
    return tableau, stock, waste


def is_free(pos, tableau):
    return all(c not in tableau for c in CHILDREN.get(pos, []))


def legal_moves(tableau, waste):
    return [p for p in tableau if is_free(p, tableau) and adjacent(tableau[p], waste)]


def unlock_count(pos, tableau):
    """How many parent cards would become free if `pos` were removed now."""
    n = 0
    for parent in PARENTS.get(pos, []):
        if parent not in tableau:
            continue
        others = [c for c in CHILDREN[parent] if c != pos]
        if all(c not in tableau for c in others):  # pos is the last thing covering it
            n += 1
    return n


def choose_random(moves, tableau, rng):
    return rng.choice(moves)


def choose_tool(moves, tableau, rng):
    # Mirrors best_move: most cards uncovered, then lower rank.
    return max(moves, key=lambda p: (unlock_count(p, tableau), -RANKS.index(tableau[p])))


def play_game(tableau, stock, waste, choose, rng):
    while tableau:
        moves = legal_moves(tableau, waste)
        if moves:
            pos = choose(moves, tableau, rng)
            waste = tableau.pop(pos)
        elif stock:
            waste = stock.pop()
        else:
            return False  # stuck with cards left
    return True  # cleared the board


def run(deals, seed):
    policies = {"random": choose_random, "tool": choose_tool}
    wins = {name: 0 for name in policies}
    for i in range(deals):
        base = random.Random(seed + i)
        deal = new_deal(base)  # same deal given to every policy
        for name, choose in policies.items():
            t, s, w = ({**deal[0]}, list(deal[1]), deal[2])
            if play_game(t, s, w, choose, random.Random(seed + i)):
                wins[name] += 1
    return wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    wins = run(args.deals, args.seed)
    print(f"TriPeaks simulation over {args.deals} identical deals per policy:\n")
    for name in ("random", "tool"):
        w = wins[name]
        print(f"  {name:7s}: {w:6d} wins  = {100 * w / args.deals:5.1f}%")
    lift = 100 * (wins["tool"] - wins["random"]) / args.deals
    print(f"\n  tool vs random: +{lift:.1f} percentage points")


if __name__ == "__main__":
    main()
