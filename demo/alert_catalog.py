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

# Lines measured on an M4 Pro with EmbeddingGemma 2 fully on the Neural
# Engine (build f14fc48), then placed halfway between the lowest hit and
# the highest miss:
#   UPS truck 0.724 vs FedEx 0.573 → 0.649
#   Sparky photo 0.896 vs ginger 0.602 → 0.749
#   bark 0.714 vs meow 0.649 → 0.682
#   meow 0.685 vs bark 0.634 on "a cat meowing" → 0.660
# Significant is no longer that text query. It is 1 - cosine against the
# empty-street frame. 0.05 only holds the rule card until the page measures
# this set and moves the line halfway from 0 to the smallest real change.
SIGNIFICANT_THRESHOLD = 0.05
UPS_THRESHOLD = 0.649
PHOTO_THRESHOLD = 0.749
DOG_THRESHOLD = 0.682
MEOW_THRESHOLD = 0.660

# Best-matching caption for the "what changed" hint. Firing ignores these.
CHANGE_LABELS: list[dict[str, str]] = [
    {"id": "truck", "text": "a truck", "caption": "a truck"},
    {"id": "person", "text": "a person", "caption": "a person"},
    {"id": "cat", "text": "a cat", "caption": "a cat"},
    {"id": "dog", "text": "a dog", "caption": "a dog"},
    {"id": "empty", "text": "an empty street", "caption": "an empty street"},
]

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
        "type": "change",
        "name": "Anything significant",
        "chip": "Significant",
        "text": None,
        "label": "different from the usual empty street",
        "baseline_id": "street",
        "scope": "image",
        "threshold": SIGNIFICANT_THRESHOLD,
        "positive_ids": ["ups", "fedex", "door", "sparky", "ginger"],
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
        "threshold": UPS_THRESHOLD,
        "positive_ids": ["ups"],
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
        "threshold": PHOTO_THRESHOLD,
        "positive_ids": ["sparky"],
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
        "threshold": DOG_THRESHOLD,
        "positive_ids": ["bark"],
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
    "threshold": MEOW_THRESHOLD,
    "positive_ids": ["meow"],
}


def catalog_items() -> list[dict[str, Any]]:
    return [*FRAMES, *SOUNDS, *REFERENCES]


def default_rules() -> list[dict[str, Any]]:
    return [dict(rule) for rule in DEFAULT_RULES]
