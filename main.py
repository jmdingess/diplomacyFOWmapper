import json
import math
import sys
from json import JSONEncoder

import matplotlib.pyplot as plt
import numpy as np
import pytesseract
from PIL import Image
from scipy import ndimage
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries, expand_labels

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return list(obj)
        return JSONEncoder.default(self, obj)


def plot_region_info(labels):
    fig, ax = plt.subplots()
    ax.imshow(labels, cmap=plt.cm.gray)
    regions = regionprops(labels)
    for props in regions:
        y0, x0 = props.centroid
        orientation = props.orientation

        x1 = x0 + math.cos(orientation) * 0.5 * props.axis_minor_length
        y1 = y0 - math.sin(orientation) * 0.5 * props.axis_minor_length
        x2 = x0 - math.sin(orientation) * 0.5 * props.axis_major_length
        y2 = y0 - math.cos(orientation) * 0.5 * props.axis_major_length

        ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
        ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
        ax.plot(x0, y0, '.g', markersize=15)
        ax.text(x0 + 5, y0 + 5, f"{props.label}", fontsize='x-large',
                bbox={"facecolor": "red", "alpha": 0.5})

        minr, minc, maxr, maxc = props.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        ax.plot(bx, by, '-b', linewidth=2.5)

    ax.axis((0, 2000, 1200, 0))


def load_images_and_bg_labels(background_fname, names_fname, centers_fname, units_fname):
    background_image = np.asarray(Image.open(background_fname).convert('RGB'))
    names_image = np.asarray(Image.open(names_fname).convert('RGBA'))
    centers_image = np.asarray(Image.open(centers_fname).convert('RGBA'))
    units_image = np.asarray(Image.open(units_fname).convert('RGBA'))
    assert background_image is not None, "bg file could not be read"
    assert names_image is not None, "names file could not be read"
    assert centers_image is not None, "centers file could not be read"
    assert units_image is not None, "units file could not be read"
    bg_labels, ndlabels = ndimage.label((background_image != [0, 0, 0]).all(-1), structure=np.ones((3, 3)))
    return background_image, names_image, centers_image, units_image, bg_labels, ndlabels


def parse_names(labeled_names, ndlabels, names_image):
    name_regions = regionprops(labeled_names)
    # columns = 6
    # fig, ax = plt.subplots(math.ceil(ndlabels/columns), min(columns, ndlabels))
    parsed_names = {}
    for props in name_regions:
        # ax_current = ax[(props.label - 1)//columns][(props.label - 1)%columns]
        minr, minc, maxr, maxc = props.bbox
        name_image = names_image[minr - 5:maxr + 5, minc - 5:maxc + 5, ...]
        name = pytesseract.image_to_string(name_image, lang='eng')
        parsed_names[props.label] = name.strip()
        # ax_current.imshow(name_image)
        # ax_current.set_title(name)
        # ax_current.set_xlabel(props.label)
        # ax_current.set_xticks([])
        # ax_current.set_yticks([])
    # fig.tight_layout()
    # fig.set_dpi(200)
    return parsed_names


def find_neighbors(labels, ndlabels):
    expanded_labels = expand_labels(labels, distance=6)
    neighbors = {}
    for label_number in range(1, ndlabels+1):
        boundaries = find_boundaries(expanded_labels == label_number, mode='outer', background=False)
        neighbors[label_number] = np.setdiff1d(np.unique(expanded_labels * boundaries), [0])
    return neighbors


COLOR_REGION_TYPE_MAPPING = {
    (197, 223, 234): "ocean",
    (226, 198, 158): "neutral",
    (121, 175, 198): "blue player",
}
def identify_region_types(background_image, bg_labels, ndlabels):
    color_mapping = {}
    for label_number in range(1, ndlabels+1):
        colors = np.unique(background_image[bg_labels == label_number], axis=0)
        assert len(colors) == 1, f"{label_number} is not one solid color: {colors}"
        color_mapping[label_number] = COLOR_REGION_TYPE_MAPPING[tuple(colors[0])]
    return color_mapping


COLOR_UNIT_MAPPING = {
    (204, 236, 193): "green player",
    (201, 255, 255): "blue player",
}
def parse_player_units(bg_labels, units_image, unit_labels, ndunits):
    region_units = {}
    player_unit_regions = {}
    for unit_number in range(1, ndunits + 1):
        labels, counts = np.unique(bg_labels[unit_labels == unit_number], return_counts=True)
        region_label = labels[counts.argmax()]

        unit_colors = np.unique(units_image[unit_labels == unit_number], axis=0)
        unit_player = None
        for color in unit_colors:
            if tuple(color[:3]) in COLOR_UNIT_MAPPING:
                unit_player = COLOR_UNIT_MAPPING[tuple(color[:3])]
                break
        assert unit_player is not None, f"Could not find player for unit {unit_number} with colors {unit_colors}"

        if unit_player not in player_unit_regions:
            player_unit_regions[unit_player] = []
        player_unit_regions[unit_player].append(region_label)
        region_units[region_label] = {"unit_number": unit_number, "player": unit_player}

    return region_units, player_unit_regions


def parse_centers(bg_labels, centers_image):
    return np.setdiff1d(np.unique(bg_labels * (centers_image[:, :, 3] != 0)), [0])


NON_PLAYER_REGIONS = ["ocean", "neutral"]
def calculate_current_player_vision(region_information):
    player_vision = {}
    for label in region_information:
        region = region_information[label]

        def add_region_for_player(player):
            if player not in player_vision:
                player_vision[player] = set()
            player_vision[player].add(int(label))
            player_vision[player].update(region["adjacency"])

        if region["is_supply_center"] and region["type"] not in NON_PLAYER_REGIONS:
            add_region_for_player(region["type"])
        if region["unit"] is not None:
            add_region_for_player(region["unit"]["player"])

    return player_vision


def calculate_visible_units(region_information, player_vision):
    player_visible_units = {}
    for player in player_vision:
        player_visible_units[player] = set()
        for label in player_vision[player]:
            region = region_information[str(label)]
            if region["unit"] is not None:
                player_visible_units[player].add(region["unit"]["unit_number"])

    return player_visible_units


PLAYER_PREVIOUSLY_SEEN_COLOR_MAPPING = {
    "blue player": (71, 125, 148),
    "green player": (114, 146, 103),
    "neutral": (176, 148, 108),
    "ocean": (147, 173, 184),
}
def calculate_previously_seen(current_player_vision, old_player_vision):
    if old_player_vision is None:
        return {player: dict() for player in current_player_vision}
    previously_seen_regions = {}
    for player in current_player_vision:
        previously_seen_regions[player] = {}
        no_longer_seen = set(map(int, old_player_vision.get(player, dict()).keys())) - current_player_vision[player]
        for region in no_longer_seen:
            region_player = old_player_vision[player][str(region)]
            region_color = PLAYER_PREVIOUSLY_SEEN_COLOR_MAPPING[region_player]
            previously_seen_regions[player][region] = {"player": region_player, "color": region_color}

    return previously_seen_regions


def generate_user_maps(player_vision, previously_seen_regions,
                       player_visible_units, bg_labels,
                       unit_labels, background_image,
                       names_image, centers_image, units_image):
    names_mask = (names_image[:, :, 3] != 0)
    centers_mask = (centers_image[:, :, 3] != 0)
    units_mask = (units_image[:, :, 3] != 0)
    for player in player_vision:
        print(f"generating user map for {player}")
        visible_regions = player_vision[player]
        previously_seen_regions_for_player = previously_seen_regions[player]
        visible_units = player_visible_units[player]

        region_mask = np.isin(bg_labels, list(visible_regions))
        player_image = background_image * region_mask[:, :, None].repeat(3, axis=2)

        previously_visible_region_mask = np.isin(bg_labels, list(previously_seen_regions_for_player.keys()))
        previously_seen_image = bg_labels[:, :, None].repeat(3, axis=2)
        for label in previously_seen_regions_for_player:
            previously_seen_image[previously_seen_image==label] = (np.ones_like(bg_labels[bg_labels==label])[None].T * previously_seen_regions_for_player[label]["color"]).flatten()
        player_image[previously_visible_region_mask] = previously_seen_image[previously_visible_region_mask]

        player_image[names_mask & (region_mask | previously_visible_region_mask)] = names_image[..., :3][names_mask & (region_mask | previously_visible_region_mask)]
        player_image[centers_mask & region_mask] = centers_image[..., :3][centers_mask & region_mask]

        units_visible_mask = np.isin(unit_labels, list(visible_units))
        player_image[units_mask & units_visible_mask] = units_image[..., :3][units_mask & units_visible_mask]

        y_indices, x_indices = np.where(previously_visible_region_mask | region_mask)
        y_min = min(y_indices)
        y_max = max(y_indices)
        x_min = min(x_indices)
        x_max = max(x_indices)

        plt.imshow(player_image)
        plt.show()
        Image.fromarray(player_image[y_min:y_max+1, x_min:x_max+1, :]).save(f"outputs/{player}.png")


def save_player_vision(region_information, current_player_vision, previously_seen_regions):
    total_player_vision = {}
    for player in current_player_vision:
        total_player_vision[player] = {str(region): region_information[str(region)]["type"] for region in current_player_vision[player]}
        total_player_vision[player].update({str(region): previously_seen_regions[player][region]["player"] for region in previously_seen_regions.get(player, dict())})

    with open("outputs/player_vision.json", "w") as json_file:
        json.dump(total_player_vision, json_file, indent=4, cls=NumpyArrayEncoder)


def generate_map_json(background_image, names_image, centers_image, units_image, bg_labels, ndlabels):
    name_labels = bg_labels * (names_image[:, :, 3] != 0)
    unit_labels, ndunits = ndimage.label((units_image[:, :, 3] != 0), structure=np.ones((3, 3)))

    region_neighbors = find_neighbors(bg_labels, ndlabels)
    region_names = parse_names(name_labels, ndlabels, names_image)
    region_centers = parse_centers(bg_labels, centers_image)
    region_types = identify_region_types(background_image, bg_labels, ndlabels)
    region_units, player_unit_regions = parse_player_units(bg_labels, units_image, unit_labels, ndunits)

    combined_region_information = {
        str(label): {
            "name": region_names[label],
            "adjacency": region_neighbors[label],
            "type": region_types[label],
            "is_supply_center": label in region_centers,
            "unit": region_units.get(label, None)
        } for label in range(1, ndlabels+1)}

    print(combined_region_information)

    with open('outputs/region_info.json', 'w') as json_file:
        json.dump(combined_region_information, json_file, indent=4, cls=NumpyArrayEncoder)

    return combined_region_information, unit_labels


def open_json_file(filename):
    try:
        with open(filename, 'r') as json_file:
            json_information = json.load(json_file)
    except FileNotFoundError as e:
        print(f"File {filename} not found; it will be generated", file=sys.stderr)
        return None
    return json_information


def update_unit_locations(region_information, bg_labels, units_image):
    unit_labels, ndunits = ndimage.label((units_image[:, :, 3] != 0), structure=np.ones((3, 3)))
    region_units, player_unit_regions = parse_player_units(bg_labels, units_image, unit_labels, ndunits)
    for label in region_information:
        region_information[label]["unit"] = region_units.get(int(label), None)
    return unit_labels


if __name__ == '__main__':
    background_image, names_image, centers_image, units_image, bg_labels, ndlabels = load_images_and_bg_labels("assets/Test.png", "assets/Testnames.png", "assets/Testcenters.png", "assets/Testunitsalt.png")
    region_information = open_json_file("outputs/region_info.json")
    if region_information is None:
        region_information, unit_labels = generate_map_json(background_image,
                                               names_image,
                                               centers_image,
                                               units_image,
                                               bg_labels,
                                               ndlabels)
    else:
        unit_labels = update_unit_locations(region_information, bg_labels, units_image)

    current_player_vision = calculate_current_player_vision(region_information)
    old_player_vision = open_json_file("outputs/player_vision.json")
    previously_seen_regions = calculate_previously_seen(current_player_vision, old_player_vision)

    player_visible_units = calculate_visible_units(region_information, current_player_vision)

    generate_user_maps(current_player_vision, previously_seen_regions, player_visible_units, bg_labels, unit_labels, background_image, names_image, centers_image, units_image)
    save_player_vision(region_information, current_player_vision, previously_seen_regions)
    print("Success")