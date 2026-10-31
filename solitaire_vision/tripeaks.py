"""Step 3: interpret the recognised board.

Disney Solitaire plays like TriPeaks: a face-up tableau card can go onto the
foundation (the card next to the draw pile) if its rank is one above or below
it, with King and Ace wrapping around.
"""

RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]


def adjacent(a, b):
    if a not in RANKS or b not in RANKS:
        return False
    d = abs(RANKS.index(a) - RANKS.index(b))
    return d in (1, len(RANKS) - 1)


def split_foundation(cards, frame_height, bottom_frac=0.25):
    """Guess which card is the foundation: the lowest face-up card in the
    bottom strip of the screen. Returns (foundation, tableau)."""
    bottom = [c for c in cards if c.face_up and c.y > frame_height * (1 - bottom_frac)]
    if not bottom:
        return None, cards
    foundation = max(bottom, key=lambda c: (c.y, c.x))
    return foundation, [c for c in cards if c is not foundation]


def playable(foundation, tableau):
    """Free, face-up tableau cards whose rank is adjacent to the foundation."""
    if foundation is None or foundation.rank is None:
        return []
    return [c for c in tableau
            if c.face_up and c.free and c.rank and adjacent(c.rank, foundation.rank)]


def best_move(foundation, tableau):
    """Pick the playable card that uncovers the most other cards.

    A card is 'worth more' if removing it frees cards currently blocked only
    by it. This is a shallow, one-ply heuristic -- it doesn't search deep
    chains -- but it beats picking an arbitrary legal move.
    """
    moves = playable(foundation, tableau)
    if not moves:
        return None

    def unlocks(card):
        count = 0
        for other in tableau:
            if other.free or other is card:
                continue
            # Would 'other' become free if only 'card' were removed? Check that
            # 'card' is the one covering it and nothing else does.
            covering = [c for c in tableau if c is not other and _covers(c, other)]
            if covering == [card]:
                count += 1
        return count

    # Prefer the move that frees the most cards; break ties by lower card
    # (clearing low cards early tends to keep streaks going).
    return max(moves, key=lambda c: (unlocks(c), -RANKS.index(c.rank)))


def _covers(top, bottom, vspan=1.25, overlap=0.35):
    """True if `top` is drawn in front of `bottom` (same rule as mark_coverage)."""
    if not top.face_up:
        return False
    if not (bottom.y + 0.05 * bottom.h < top.y < bottom.y + vspan * bottom.h):
        return False
    overlap_x = min(bottom.x + bottom.w, top.x + top.w) - max(bottom.x, top.x)
    return overlap_x > overlap * min(bottom.w, top.w)
