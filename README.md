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
4. **Interpret** (`solitaire_vision/tripeaks.py`): the lowest face-up card
   at the bottom of the screen is taken as the foundation. A tableau card is
   playable if its rank is ±1 from the foundation (K↔A wraps).

## Usage

```bash
pip install -r requirements.txt

# 1. Take a screenshot of the game and teach it the glyphs (once per deck style)
python main.py label --image shot.png          # type e.g. QH, 10S, AD for each card

# 2. Recognise a screenshot and save an annotated copy
python main.py detect --image shot.png --out annotated.png

# 3. Live: watch a screen region (left top width height) and print changes
python main.py watch --region 0 0 1080 1920 --interval 1
```

Colours in the annotated image: blue = foundation, green = playable,
yellow = other face-up cards, grey = face-down cards.

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
