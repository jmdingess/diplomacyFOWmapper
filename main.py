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

        # ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
        # ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
        ax.plot(x0, y0, '.g', markersize=15)
        ax.text(x0 + 5, y0 + 5, f"{props.label}", fontsize='x-large',
                bbox={"facecolor": "red", "alpha": 0.5})

        minr, minc, maxr, maxc = props.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        # ax.plot(bx, by, '-b', linewidth=2.5)

    fig.set_dpi(200)
    ax.axis((0, labels.shape[1], labels.shape[0], 0))
    plt.show()


def load_images_and_bg_labels(background_fname, names_fname, centers_fname, units_fname, retreats_fname):
    background_image = np.asarray(Image.open(background_fname).convert('RGB'))
    names_image = np.asarray(Image.open(names_fname).convert('RGBA'))
    centers_image = np.asarray(Image.open(centers_fname).convert('RGBA'))
    units_image = np.asarray(Image.open(units_fname).convert('RGBA'))
    retreats_image = np.asarray(Image.open(retreats_fname).convert('RGBA'))
    assert background_image is not None, "bg file could not be read"
    assert names_image is not None, "names file could not be read"
    assert centers_image is not None, "centers file could not be read"
    assert units_image is not None, "units file could not be read"
    bg_labels, ndlabels = ndimage.label((background_image != [0, 0, 0]).any(-1), structure=np.ones((3, 3)))
    return background_image, names_image, centers_image, units_image, retreats_image, bg_labels, ndlabels


def world_wrap_labels(bg_labels, ndlabels):
    # plot_region_info(bg_labels)
    wrapped_regions = ((bg_labels[:, 0] != 0) & (bg_labels[:, -1] != 0))
    connected_labels = np.unique(
        np.column_stack((bg_labels[wrapped_regions, 0], bg_labels[wrapped_regions, -1])),
        axis=0)
    assert np.unique(connected_labels).size == connected_labels.size, f"Some labels are wrapping to multiple regions: {connected_labels}"
    print(connected_labels)

    missing_labels = []
    for connection in connected_labels:
        smaller = min(connection)
        larger = max(connection)
        bg_labels[bg_labels == larger] = np.ones_like(bg_labels[bg_labels == larger]) * smaller
        missing_labels.append(larger)

    for missing_label in sorted(missing_labels, reverse=True):
        bg_labels[missing_label < bg_labels] -= 1

    return bg_labels, ndlabels - len(missing_labels)


def parse_names(labeled_names, ndlabels, names_image):
    name_regions = regionprops(labeled_names)
    columns = 6
    names_to_show = min(ndlabels, 30)
    fig, ax = plt.subplots(math.ceil(names_to_show/columns), min(columns, names_to_show))
    parsed_names = {}
    for props in name_regions:
        minr, minc, maxr, maxc = props.bbox
        name_image = names_image[minr - 5:maxr + 5, minc - 5:maxc + 5, ...]
        name = pytesseract.image_to_string(name_image, lang='eng')
        parsed_names[props.label] = name.strip()
        if props.label - 1 < names_to_show:
            ax_current = ax[(props.label - 1) // columns][(props.label - 1) % columns]
            ax_current.imshow(name_image)
            ax_current.set_title(name)
            ax_current.set_xlabel(props.label)
            ax_current.set_xticks([])
            ax_current.set_yticks([])
    fig.tight_layout()
    fig.set_dpi(200)
    plt.show()
    return parsed_names


def find_neighbors(labels, ndlabels):
    expanded_labels = expand_labels(labels, distance=6)
    neighbors = {}
    for label_number in range(1, ndlabels+1):
        boundaries = find_boundaries(expanded_labels == label_number, mode='outer', background=False)
        neighbors[label_number] = np.setdiff1d(np.unique(expanded_labels * boundaries), [0])
    # Wrap adjacency along X axis
    adjacent_on_edge = expanded_labels[:, 0] != expanded_labels[:, -1]
    connected_regions = np.unique(np.column_stack((expanded_labels[adjacent_on_edge, 0], expanded_labels[adjacent_on_edge, -1])), axis=0)
    print("Connected regions", connected_regions)
    for connection in connected_regions:
        if 0 in connection:
            continue
        if (connection[1] not in neighbors[connection[0]]):
            neighbors[connection[0]] = np.append(neighbors[connection[0]], connection[1])
        if (connection[0] not in neighbors[connection[1]]):
            neighbors[connection[1]] = np.append(neighbors[connection[1]], connection[0])
    return neighbors


COLOR_REGION_TYPE_MAPPING = {
    (197, 223, 234): "ocean",
    (226, 198, 158): "neutral",
    (96, 96, 96): "edge",
    (0, 148, 255): "the fallen",
    (255, 0, 220): "mungus",
    (255, 133, 20): "greenland",
    (239, 196, 228): "pink submarine",
    (0, 255, 255): "The Count of the Lands of Supercalifragilisticexpialadocioustic Peoples from Death",
    (50, 214, 17): "italia",
    (26, 188, 156): "tonga",
    (196, 143, 133): "abjaria",
    (204, 204, 255): "arktonid republic",
    (59, 137, 44): "italy",
    (255, 203, 91): "abukan",
    (245, 245, 245): "icenia",
    (255, 56, 50): "legadonia",
    (102, 51, 153): "urugay",
    (139, 29, 29): "northwestern gnome land",
}
def identify_region_types(background_image, bg_labels, ndlabels):
    color_mapping = {}
    for label_number in range(1, ndlabels+1):
        colors = np.unique(background_image[bg_labels == label_number], axis=0)
        assert len(colors) == 1, f"{label_number} is not one solid color: {colors}"
        assert tuple(colors[0]) in COLOR_REGION_TYPE_MAPPING, f"{tuple(colors[0])} is not in mapping (region {label_number})"
        color_mapping[label_number] = COLOR_REGION_TYPE_MAPPING[tuple(colors[0])]
    return color_mapping


COLOR_UNIT_MAPPING = {
    (25, 173, 255): "the fallen",
    (255, 25, 245): "mungus",
    (255, 158, 45): "greenland",
    (255, 221, 253): "pink submarine",
    (25, 255, 255): "The Count of the Lands of Supercalifragilisticexpialadocioustic Peoples from Death",
    (75, 239, 42): "italia",
    (51, 213, 181): "tonga",
    (221, 168, 158): "abjaria",
    (229, 229, 255): "arktonid republic",
    (84, 162, 69): "italy",
    (255, 228, 116): "abukan",
    (255, 255, 255): "icenia",
    (255, 81, 75): "legadonia",
    (127, 76, 178): "urugay",
    (164, 54, 54): "northwestern gnome land",
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


NON_PLAYER_REGIONS = ["ocean", "neutral", "edge"]
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


def give_nearby_center_information(current_player_vision, region_information):
    vision = {}
    for player in current_player_vision:
        vision[player] = {}
        for region in current_player_vision[player]:
            if (region_information[str(region)]["type"] == "edge"):
                continue
            vision[player][str(region)] = region_information[str(region)]["type"]
            for adjacent_region in region_information[str(region)]["adjacency"]:
                if region_information[str(adjacent_region)]["is_supply_center"]:
                    vision[player][str(adjacent_region)] = region_information[str(adjacent_region)]["type"]
    return vision


def calculate_visible_units(unit_regions, player_vision):
    player_visible_units = {}
    for player in player_vision:
        player_visible_units[player] = set()
        for label in player_vision[player]:
            if label in unit_regions:
                player_visible_units[player].add(unit_regions[label]["unit_number"])

    return player_visible_units


PLAYER_PREVIOUSLY_SEEN_COLOR_MAPPING = {
    "neutral": (176, 148, 108),
    "ocean": (147, 173, 184),
    "edge": (96, 96, 96),
    "the fallen": (0, 98, 205),
    "mungus": (205, 0, 170),
    "greenland": (205, 83, 0),
    "pink submarine": (189, 146, 178),
    "The Count of the Lands of Supercalifragilisticexpialadocioustic Peoples from Death": (0, 205, 205),
    "italia": (0, 164, 0),
    "tonga": (0, 138, 106),
    "abjaria": (146, 93, 83),
    "arktonid republic": (154, 154, 205),
    "italy": (9, 87, 0),
    "abukan": (205, 153, 41),
    "icenia": (195, 195, 195),
    "legadonia": (205, 6, 0),
    "urugay": (52, 1, 103),
    "northwestern gnome land": (89, 0, 0),
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


SHOW_EXTENDED_MAP_THRESHOLD = 30
SHOW_EXTENDED_MAP_SCROLL = 990
def generate_user_maps(player_vision, previously_seen_regions,
                       player_visible_units, player_visible_retreats, bg_labels,
                       unit_labels, retreats_labels, background_image,
                       names_image, centers_image, units_image, retreats_image):
    names_mask = (names_image[:, :, 3] != 0)
    centers_mask = (centers_image[:, :, 3] != 0)
    units_mask = (units_image[:, :, 3] != 0)
    retreats_mask = (retreats_image[:, :, 3] != 0)
    for player in player_vision:
        print(f"generating user map for {player}")
        visible_regions = player_vision[player]
        previously_seen_regions_for_player = previously_seen_regions[player]
        visible_units = player_visible_units[player]
        visible_retreats = player_visible_retreats[player]

        region_mask = np.isin(bg_labels, list(visible_regions))
        player_image = background_image * region_mask[:, :, None].repeat(3, axis=2)

        previously_visible_region_mask = np.isin(bg_labels, list(previously_seen_regions_for_player.keys()))
        previously_seen_image = bg_labels[:, :, None].repeat(3, axis=2)
        for label in previously_seen_regions_for_player:
            previously_seen_image[(previously_seen_image==[label, label, label]).all(-1)] = np.ones_like(bg_labels[bg_labels==label])[None].T * previously_seen_regions_for_player[label]["color"]
        player_image[previously_visible_region_mask] = previously_seen_image[previously_visible_region_mask]

        player_image[names_mask & (region_mask | previously_visible_region_mask)] = names_image[..., :3][names_mask & (region_mask | previously_visible_region_mask)]
        player_image[centers_mask & (region_mask | previously_visible_region_mask)] = centers_image[..., :3][centers_mask & (region_mask | previously_visible_region_mask)]

        units_visible_mask = np.isin(unit_labels, list(visible_units))
        player_image[units_mask & units_visible_mask] = units_image[..., :3][units_mask & units_visible_mask]

        retreats_visible_mask = np.isin(retreats_labels, list(visible_retreats))
        player_image[retreats_mask & retreats_visible_mask] = retreats_image[..., :3][retreats_mask & retreats_visible_mask]

        y_indices, x_indices = np.where(previously_visible_region_mask | region_mask)
        y_min = min(y_indices)
        y_max = max(y_indices)
        x_min = min(x_indices)
        x_max = max(x_indices)

        plt.imshow(player_image)
        plt.show()
        if x_min < SHOW_EXTENDED_MAP_THRESHOLD//2 and bg_labels.shape[1] - 1 - SHOW_EXTENDED_MAP_SCROLL//2 < x_max:
            x_sorted = np.sort(np.unique(x_indices))
            differences = x_sorted[1:] - x_sorted[:-1]
            x_left = x_sorted[np.argmax(differences)]
            x_right = x_sorted[np.argmax(differences) + 1]
            if x_right - x_left < SHOW_EXTENDED_MAP_THRESHOLD:
                scrolled_image = np.concatenate((player_image, player_image[:, :SHOW_EXTENDED_MAP_SCROLL, :]), axis=1)
                Image.fromarray(scrolled_image[y_min:y_max+1, :, :]).save(f"outputs/{player}.png")
                scrolled_image_alternate = np.concatenate((player_image[:, SHOW_EXTENDED_MAP_SCROLL:, :], player_image), axis=1)
                Image.fromarray(scrolled_image_alternate[y_min:y_max+1, :, :]).save(f"outputs/{player}_alt.png")
                continue
            else:
                player_image = np.concatenate((player_image[:, x_right:, :], player_image[:, :x_left+1, :]), axis=1)
                x_max -= (x_right - (x_left + 1))
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

    print("Finding neighbors")
    region_neighbors = find_neighbors(bg_labels, ndlabels)
    print("Finding region names")
    region_names = parse_names(name_labels, ndlabels, names_image)
    print("Finding centers")
    region_centers = parse_centers(bg_labels, centers_image)
    print("Finding region types")
    region_types = identify_region_types(background_image, bg_labels, ndlabels)
    print("Finding units")
    region_units, player_unit_regions = parse_player_units(bg_labels, units_image, unit_labels, ndunits)

    print("Wrapping information")
    combined_region_information = {
        str(label): {
            "name": region_names.get(label, "not named"),
            "adjacency": region_neighbors[label],
            "type": region_types[label],
            "is_supply_center": label in region_centers,
            "unit": region_units.get(label, None)
        } for label in range(1, ndlabels+1)}

    print(combined_region_information)

    with open('outputs/region_info.json', 'w') as json_file:
        json.dump(combined_region_information, json_file, indent=4, cls=NumpyArrayEncoder)

    return combined_region_information, unit_labels, region_units


def get_retreat_locations(retreats_image, bg_labels):
    retreat_labels, ndretreats = ndimage.label((retreats_image[:, :, 3] != 0), structure=np.ones((3, 3)))
    region_retreats, player_retreat_regions = parse_player_units(bg_labels, retreats_image, retreat_labels, ndretreats)
    return region_retreats, retreat_labels


def open_json_file(filename):
    try:
        with open(filename, 'r') as json_file:
            json_information = json.load(json_file)
    except FileNotFoundError:
        print(f"File {filename} not found; it will be generated", file=sys.stderr)
        return None
    return json_information


def update_unit_locations(region_information, bg_labels, units_image):
    unit_labels, ndunits = ndimage.label((units_image[:, :, 3] != 0), structure=np.ones((3, 3)))
    unit_labels, ndunits = world_wrap_labels(unit_labels, ndunits)
    region_units, player_unit_regions = parse_player_units(bg_labels, units_image, unit_labels, ndunits)
    for label in region_information:
        region_information[label]["unit"] = region_units.get(int(label), None)
    return unit_labels, region_units


def update_region_types(background_image, bg_labels, ndlabels, region_information):
    region_types = identify_region_types(background_image, bg_labels, ndlabels)
    for label in region_information:
        region_information[label]["type"] = region_types[int(label)]


def print_country_stats(region_information):
    country_stats = {}
    for region_label in region_information:
        region = region_information[region_label]
        if region["is_supply_center"]:
            if region["type"] not in country_stats:
                country_stats[region["type"]] = { "units": 0, "centers": 0 }
            country_stats[region["type"]]["centers"] += 1
        if region["unit"] is not None:
            if region["unit"]["player"] not in country_stats:
                country_stats[region["unit"]["player"]] = { "units": 0, "centers": 0 }
            country_stats[region["unit"]["player"]]["units"] += 1
    print(country_stats)

if __name__ == '__main__':
    background_image, names_image, centers_image, units_image, retreats_image, bg_labels, ndlabels = \
        load_images_and_bg_labels("assets/fow_map.png",
                                  "assets/fow_names.png",
                                  "assets/fow_centers.png",
                                  "assets/fow_units.png",
                                  "assets/fow_retreats.png")
    bg_labels, ndlabels = world_wrap_labels(bg_labels, ndlabels)
    plot_region_info(bg_labels)
    # region_centers = parse_centers(bg_labels, centers_image)
    # print(ndlabels)
    # print(region_centers, len(region_centers))
    # exit()
    region_information = open_json_file("outputs/region_info.json")
    if region_information is None:
        region_information, unit_labels, region_units = generate_map_json(background_image,
                                               names_image,
                                               centers_image,
                                               units_image,
                                               bg_labels,
                                               ndlabels)
    else:
        unit_labels, region_units = update_unit_locations(region_information, bg_labels, units_image)
        update_region_types(background_image, bg_labels, ndlabels, region_information)

    retreats_regions, retreats_labels = get_retreat_locations(retreats_image, bg_labels)

    current_player_vision = calculate_current_player_vision(region_information)
    old_player_vision = open_json_file("outputs/player_vision.json")
    if old_player_vision is None:
        old_player_vision = give_nearby_center_information(current_player_vision, region_information)

    for retreat_region in retreats_regions:
        print(retreats_regions[retreat_region]["player"])
        if retreats_regions[retreat_region]["player"] not in current_player_vision:
            current_player_vision[retreats_regions[retreat_region]["player"]] = set()
        current_player_vision[retreats_regions[retreat_region]["player"]].add(retreat_region)
        current_player_vision[retreats_regions[retreat_region]["player"]].update(region_information[str(retreat_region)]["adjacency"])

    previously_seen_regions = calculate_previously_seen(current_player_vision, old_player_vision)

    player_visible_units = calculate_visible_units(region_units, current_player_vision)
    player_visible_retreats = calculate_visible_units(retreats_regions, current_player_vision)

    print_country_stats(region_information)
    generate_user_maps(current_player_vision, previously_seen_regions, player_visible_units, player_visible_retreats, bg_labels, unit_labels, retreats_labels, background_image, names_image, centers_image, units_image, retreats_image)
    save_player_vision(region_information, current_player_vision, previously_seen_regions)
    print("Success")