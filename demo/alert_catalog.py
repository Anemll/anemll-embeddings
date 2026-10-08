"""Pinned camera-alert samples and the four preset rules.

Images and clips are not in git. ``demo/scripts/fetch_alert.py`` downloads
these Wikimedia Commons files (CC0, CC BY, or CC BY-SA, attribution stored
in the manifest) into ``ANEMLL_DEMO_ALERT``.

Sparky's reference and the Sparky test frame are two photographs of the
same black cat: Nikolai Bulykin's series at Medeo (Almaty),
"лестница здоровья, кошка" (1) on the bench and (5) on the step.
"""

from __future__ import annotations

from typing import Any

# Placeholder thresholds until every item in the rule's scope has a score.
# Text–image cosines are modest; 0.15 is only a starting line. The baseline
# rule uses a margin over the empty street, so 0.02 means "a little more
# like the sentence than the empty street." Photo matching starts at 0.35.
# Score everything replaces these with a threshold fit to this set.
TEXT_PLACEHOLDER = 0.15
MARGIN_PLACEHOLDER = 0.02
PHOTO_PLACEHOLDER = 0.35

FRAMES: list[dict[str, Any]] = [
    {
        "id": "ups",
        "caption": "Front door cam: UPS truck",
        "modality": "image",
        "kind": "scene",
        "commons": "File:UPS package car.jpg",
        "file": "frames/ups.jpg",
    },
    {
        "id": "fedex",
        "caption": "Front door cam: FedEx truck",
        "modality": "image",
        "kind": "scene",
        "commons": "File:FedEx Truck.jpg",
        "file": "frames/fedex.jpg",
    },
    {
        "id": "ginger",
        "caption": "Neighbor's cat",
        "modality": "image",
        "kind": "cat",
        "commons": "File:Ginger european cat.jpg",
        "file": "frames/ginger.jpg",
    },
    {
        "id": "sparky",
        "caption": "Sparky (test photo)",
        "modality": "image",
        "kind": "cat",
        "commons": "File:Медео, лестница здоровья, кошка (5).jpg",
        "file": "frames/sparky.jpg",
    },
    {
        "id": "door",
        "caption": "Front door cam: person at the door",
        "modality": "image",
        "kind": "scene",
        "commons": "File:Knocking on doors (50212006202).jpg",
        "file": "frames/door.jpg",
    },
    {
        "id": "street",
        "caption": "Front door cam: empty street",
        "modality": "image",
        "kind": "scene",
        "commons": "File:Empty Street - geograph.org.uk - 2719534.jpg",
        "file": "frames/street.jpg",
        "baseline": True,
    },
]

SOUNDS: list[dict[str, Any]] = [
    {
        "id": "bark",
        "caption": "Sound: dog barking",
        "modality": "audio",
        "kind": "sound",
        "commons": "File:Barking of a dog.ogg",
        "file": "sounds/bark.wav",
    },
    {
        "id": "meow",
        "caption": "Sound: cat meowing",
        "modality": "audio",
        "kind": "sound",
        "commons": "File:Meow.ogg",
        "file": "sounds/meow.wav",
    },
]

REFERENCES: list[dict[str, Any]] = [
    {
        "id": "sparky-ref",
        "rule_id": "sparky",
        "caption": "Sparky reference photo",
        "modality": "image",
        "kind": "reference",
        "commons": "File:Медео, лестница здоровья, кошка (1).jpg",
        "file": "references/sparky.jpg",
    },
]

DEFAULT_RULES: list[dict[str, Any]] = [
    {
        "id": "significant",
        "type": "text",
        "name": "Anything significant",
        "chip": "Significant",
        "text": "a person, animal, or vehicle in front of a house",
        "baseline_id": "street",
        "scope": "image",
        "threshold": MARGIN_PLACEHOLDER,
        "color": "#7dbea8",
    },
    {
        "id": "ups",
        "type": "text",
        "name": "UPS truck",
        "chip": "UPS truck",
        "text": "a brown UPS delivery truck",
        "baseline_id": None,
        "scope": "image",
        "threshold": TEXT_PLACEHOLDER,
        "color": "#e39a45",
    },
    {
        "id": "sparky",
        "type": "photo",
        "name": "Sparky",
        "chip": "Sparky",
        "text": None,
        "baseline_id": None,
        "scope": "image",
        "threshold": PHOTO_PLACEHOLDER,
        "color": "#e2c57a",
    },
    {
        "id": "dog",
        "type": "text",
        "name": "Dog barking",
        "chip": "Dog barking",
        "text": "a dog barking",
        "baseline_id": None,
        "scope": "audio",
        "threshold": TEXT_PLACEHOLDER,
        "color": "#e09a8a",
    },
]

# Not one of the preset alerts. Sound tiles also show this comparison so a
# meow and a bark can be told apart the same way as the alert meters.
MEOW_COMPARE: dict[str, Any] = {
    "id": "meow-query",
    "text": "a cat meowing",
    "label": "a cat meowing",
    "scope": "audio",
    "threshold": TEXT_PLACEHOLDER,
}


def catalog_items() -> list[dict[str, Any]]:
    return [*FRAMES, *SOUNDS, *REFERENCES]


def default_rules() -> list[dict[str, Any]]:
    return [dict(rule) for rule in DEFAULT_RULES]
