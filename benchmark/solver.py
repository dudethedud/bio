"""Full-information TriPeaks solver.

Given a complete board (tableau + the ordered stock + the current waste), search
for a sequence of moves that clears all 28 tableau cards. This is 'optimal play'
with perfect knowledge -- useful for (a) measuring the ceiling win rate and
(b) auto-completing a known deal.

It is NOT what the tool can do on the live game: there the stock is face-down,
so upcoming draws are unknown and only the best *immediate* move can be chosen.
See best_immediate() for the under-uncertainty version.

A move is ("play", pos) or ("draw",).
"""

from simulate import CHILDREN, PARENTS, RANKS, adjacent, is_free  # noqa: F401


def legal_plays(tableau, waste):
    return [p for p in tableau if is_free(p, tableau) and adjacent(tableau[p], waste)]


def _unlock(pos, tableau):
    n = 0
    for parent in PARENTS.get(pos, []):
        if parent in tableau and all(c not in tableau for c in CHILDREN[parent] if c != pos):
            n += 1
    return n


def solve(tableau, stock, waste, node_cap=200000):
    """Return a winning move list, or None if no win exists within node_cap.

    stock is a list with the top of the pile at the end (drawn via pop order);
    we track how many remain with `left`.
    """
    seen = set()
    nodes = [0]

    def dfs(tab, left, waste, path):
        if not tab:
            return list(path)
        nodes[0] += 1
        if nodes[0] > node_cap:
            raise _Budget()
        key = (frozenset(tab.items()), left, waste)
        if key in seen:
            return None
        seen.add(key)

        # Try plays first, best-uncovering ones first (finds wins faster).
        plays = legal_plays(tab, waste)
        plays.sort(key=lambda p: (_unlock(p, tab), -RANKS.index(tab[p])), reverse=True)
        for pos in plays:
            nxt = dict(tab)
            moved = nxt.pop(pos)
            path.append(("play", pos))
            got = dfs(nxt, left, moved, path)
            if got:
                return got
            path.pop()

        if left > 0:
            path.append(("draw",))
            got = dfs(tab, left - 1, stock[left - 1], path)
            if got:
                return got
            path.pop()
        return None

    try:
        return dfs(dict(tableau), len(stock), waste, [])
    except _Budget:
        return None


class _Budget(Exception):
    pass


def best_immediate(tableau, waste, rng=None):
    """Pick the best move WITHOUT seeing the stock (the live-game case).

    Heuristic: among legal plays prefer the one that uncovers the most cards,
    then the one that keeps the most future options open (frees a card adjacent
    to many others), then the lower rank. Returns ("play", pos) or ("draw",).
    """
    plays = legal_plays(tableau, waste)
    if not plays:
        return ("draw",)

    def score(pos):
        unlocked = _unlock(pos, tableau)
        # look one card deeper: ranks that become reachable after this play
        reach = 0
        for parent in PARENTS.get(pos, []):
            if parent in tableau and all(c not in tableau for c in CHILDREN[parent] if c != pos):
                reach += 1  # a newly-free card is a new option next turn
        return (unlocked, reach, -RANKS.index(tableau[pos]))

    return ("play", max(plays, key=score))
