# Diplomacy Fog-of-War mapper
Generates FOW maps for diplomacy games

## How to use:
1. Install the `requirements.txt`
2. Add asset files to assets:
   1. They must all be the same dimensions
   2. They should all be PNGs
   3. One file must contain just the background image (Test.png)
      1. Anything black (255, 255, 255) will be treated as a border
      2. Territories are regions of continuous color (diagonals are connected); sorry for anyone who likes to draw canals/whatever
      3. It should be able to handle borders up to 12px thickness just fine, but I've only tested with 6px. If you need to tune this just search for `expand_labels` in the code and change the distance (should be able to handle `2*distance`)
   4. One file must contain just supply centers (Testcenters.png)
      1. Anywhere the centers map is not invisible (alpha = 0), it will mark that territory as a center
   5. One file must contain just territory names (Testnames.png)
      1. Territory names should not be on borders; they should be entirely within a region
      2. It will attempt to use `pytesseract` to do image recognition and guess what your regions are named, but if you don't want to have to install pytesseract, comment that out
   6. One file must contain just units (Testunits/Testunitsalt.png)
      1. It's okay for regions to touch borders or even poke into other areas. If this happens it should take the region with the most unit on it
      2. Units should not touch. A unit is recognized by a connected (including diagonals) region of non-invisible pixels. If you have two units touching, they will be thought of as the same unit.
      3. It cannot tell boats from armies currently
3. Change hardcoded values in the code:
   1. Filenames in the bottom of main.py
   2. constants throughout the file used for mapping:
      1. `COLOR_REGION_TYPE_MAPPING` maps an rgb color for a territory to the player that owns it
      2. `COLOR_UNIT_MAPPING` maps an rgb color on a unit to the player that owns it (if units are multicolored, the most common color that has an existant matching will be used)
      3. `NON_PLAYER_REGIONS` will probably just contain `"neutral"` and `"ocean"` but you can add other items defined in a/b that are not to be treated as players here
      4. `PLAYER_PREVIOUSLY_SEEN_COLOR_MAPPING` maps players to what rgb color should be used to represent regions they own players used to see but currently do not. I personally like to just use Paint.Net's "brightness/contrast" tool to turn down brightness by 50% for this
4. `python main.py`
5. The first time it runs, it will take longer because it needs to generate region information. On later runs this information will be re-used so it should be faster
6. Output files should be generated in `outputs/`

## Troubleshooting
There's some commented out code in the `parse_names` function that may be useful for problems with name parsing.
The `plot_region_info` function can be used on any label graph (currently it's unused in the code; it's only here for debugging).
It will output a matplotlib graph that shows which region has which label as well as other info.

If you have very tiny borders between territories, you may find that territory labelling jumping over diagonals is a detriment. 
If this is the case, you can change the `structure` argument passed to `ndimage.label` in `load_images_and_bg_labels`.
The current setup has it look for neighbors like this: 
```
[[ 1, 1, 1],
 [ 1, 1, 1],
 [ 1, 1, 1]]
```
If you imagine the center of the matrix is your pixel, it will be connected everywhere there is a `1`.
The current setup has it look at diagonal spots but if you change those to 0's you can make it orthogonal only.

You don't need `pytesseract` really - I already had it installed from trying to use felixludos' digi-diplo months ago (shoutout to them; the name parsing part of this is based on the code in the jupyter notebooks of that project)
