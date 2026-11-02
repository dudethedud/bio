# Disney Solitaire – screen recognition

A small OpenCV pipeline that reads a Disney Solitaire board from the screen
(or a screenshot), works out which cards are where, and lists which cards can
be played on the foundation. It only reads the board. It doesn't click or
send any input.

## How the method works

1. **Capture** (`solitaire_vision/capture.py`): grab the screen or a window
   region with `mss`, or load a screenshot.
2. **Detect cards** (`solitaire_vision/detect.py`)
   - Convert to HSV and mask pixels that are bright and unsaturated, which
     picks out the white card faces.
   - Find the outer contours and keep boxes with a card-like size and an
     aspect ratio of about 0.55–0.85.
   - Face-down cards use the same steps with a colour mask for the card back
     (`--back-hsv`, because the back depends on the deck you have selected).
   - Crop each face-up card's top-left corner, where the rank and suit are.
3. **Identify** (`solitaire_vision/identify.py`)
   - Binarise the corner with Otsu, then split it into a rank glyph and a
     suit glyph using horizontal ink bands.
   - Crop each glyph to its ink, resize it to 32×40 with a blank border, and
     compare it with labelled templates using normalised cross-correlation.
   - Count red ink against black ink to narrow the suit match to ♥♦ or ♠♣.
4. **Coverage** (`solitaire_vision/detect.py`, `mark_coverage`): a card is
   only playable if nothing is drawn in front of it. Each card is marked
   `free` or covered from the overlap geometry; face-down backs behind a card
   don't count as covering it.
5. **Interpret** (`solitaire_vision/tripeaks.py`): the lowest face-up card at
   the bottom of the screen is taken as the foundation (waste pile). A *free*
   tableau card is playable if its rank is ±1 from the foundation (K↔A wraps).
   `best_move()` then recommends the legal play that uncovers the most cards.

## Usage

```bash
pip install -r requirements.txt

# Recognise a screenshot with the bundled Disney deck templates + card-back colour
python main.py detect --image shot.png \
    --templates example_templates/disney \
    --back-hsv 95 120 120 125 255 255 \
    --out annotated.png

# Pick the game window once by dragging a box over it (saves to region.json)
python main.py pick

# Live: watch that region and print the best move whenever the board changes
python main.py watch --interval 1 \
    --templates example_templates/disney --back-hsv 95 120 120 125 255 255

# Teach it glyphs it doesn't know yet (see "Templates" below)
python main.py label --image shot.png --templates example_templates/disney
```

Colours in the annotated image: **blue** = foundation/waste,
**bright green** = recommended move, **dim green** = other legal move,
**yellow** = free face-up card, **dim** = covered face-up card,
**grey** = face-down card.

### Templates

`example_templates/disney/` ships with glyphs trained from a real level-51
board, covering ranks **A, 2, 5, 6, 8, 10, Q** and all four suits. The ranks
**3, 4, 7, 9, J, K** aren't in there yet — when you hit a board with one,
run `label` on a screenshot of it and type the card (e.g. `7H`, `KD`) to add
it. More samples per card only improves accuracy. The `--back-hsv` preset
above matches the blue Mickey card back; change it if you use another deck.

### Picking the capture region

For live play on an iPad, mirror the iPad to the computer first (QuickTime
"New Movie Recording" on a Mac, or an AirPlay receiver app), so the game shows
in a window. Then run `python main.py pick`: the screen freezes, you drag a box
over the mirror window, and press **ENTER**. The box is saved to `region.json`,
and `detect`/`watch` use it automatically (override any time with
`--region LEFT TOP W H`, or point at another file with `--region-file`).

## Tuning

If detection misses cards, change `DetectConfig` in `detect.py`:
`white_min_val` / `white_max_sat` set what counts as a card face, the
`*_area_frac` values set the size limits, and `corner_w` / `corner_h` set the
corner crop. Overlapping cards in the peaks may merge into a single contour.
Splitting those needs extra work, and that's a good place to start improving
this.

## Tests

```bash
python -m pytest -q
```

The test builds a synthetic board, labels it, and checks detection,
identification and the playable-move logic from start to finish.
