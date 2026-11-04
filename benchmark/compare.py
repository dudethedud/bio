"""Compare all strategies on identical deals, including the optimal ceiling.

  random    : random legal move, else draw            (no strategy)
  greedy    : uncover-most heuristic (the shipped best_move)
  immediate : best_immediate -- deeper live-game heuristic (stock unseen)
  optimal   : full-information solver (win iff the deal is winnable at all)

Only `optimal` is allowed to see the stock; it marks the theoretical ceiling.

    python benchmark/compare.py --deals 3000
"""

import argparse
import random

from simulate import choose_random, choose_tool, legal_moves, new_deal, play_game
from solver import best_immediate, solve


def play_immediate(tableau, stock, waste):
    tab, left, w = dict(tableau), len(stock), waste
    while tab:
        mv = best_immediate(tab, w)
        if mv[0] == "play":
            w = tab.pop(mv[1])
        elif left > 0:
            left -= 1
            w = stock[left]
        else:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    wins = {k: 0 for k in ("random", "greedy", "immediate", "optimal")}
    for i in range(args.deals):
        tableau, stock, waste = new_deal(random.Random(args.seed + i))

        t, s, w = dict(tableau), list(stock), waste
        if play_game(t, s, w, choose_random, random.Random(args.seed + i)):
            wins["random"] += 1

        t, s, w = dict(tableau), list(stock), waste
        if play_game(t, s, w, choose_tool, random.Random(args.seed + i)):
            wins["greedy"] += 1

        if play_immediate(tableau, stock, waste):
            wins["immediate"] += 1

        if solve(tableau, stock, waste, node_cap=15000):
            wins["optimal"] += 1

    print(f"TriPeaks, {args.deals} identical deals per strategy:\n")
    for k in ("random", "greedy", "immediate", "optimal"):
        print(f"  {k:10s}: {100 * wins[k] / args.deals:5.1f}%")
    print(f"\n  optimal is the ceiling (node-capped, so a floor on the true value): "
          f"at least {100 * wins['optimal'] / args.deals:.1f}% of deals are winnable with "
          f"perfect knowledge of the stock. The gap between it and the heuristics is "
          f"information you don't have on screen, not cleverness.")


if __name__ == "__main__":
    main()
