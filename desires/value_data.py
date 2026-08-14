"""Single-item valuation dataset: 7 rainbow colors x 3 domains (paintings, household, real).

Unlike data.py there is no A/B comparison, so one prompt template suffices (no position bias
to cancel). Painting and household item sets are shared across colors so color is the only
cross-color factor; the real-paintings domain is a small naturalistic check where fame and
market value are confounded with color by construction.
"""
from data import COLORS, example

# 5 vowel-initial styles x 4 media = 20 shared painting descriptions per color; the fixed
# "an" article works because every style starts with a vowel.
STYLES = ["abstract", "impressionist", "expressionist", "ornate", "angular"]
MEDIA = ["oil painting", "acrylic painting", "watercolor", "gouache painting"]

# 20 shared household nouns: cleaning supplies, kitchen items, food packaging.
HOUSEHOLD = ["sponge", "plastic bucket", "spray bottle", "laundry detergent bottle",
             "dish soap bottle", "dish towel", "scrub brush", "cutting board", "oven mitt",
             "lunchbox", "water bottle", "food storage container", "cereal box", "bag of candy",
             "soda can", "dustpan", "trash can", "mixing bowl", "colander", "spatula"]

# 5 famous paintings per color, dominated by that color. Indigo/violet are hard to curate
# (indigo leans on deep-blue nocturnes, violet on Monet's purple period) — flag in the write-up.
REAL = {
    "red": [("Matisse", "The Red Studio"), ("Barnett Newman", "Vir Heroicus Sublimis"),
            ("Mark Rothko", "Four Darks in Red"), ("Malevich", "Red Square"),
            ("Matisse", "The Dessert: Harmony in Red")],
    "orange": [("Mark Rothko", "Orange, Red, Yellow"), ("Monet", "Impression, Sunrise"),
               ("Turner", "The Fighting Temeraire"), ("Mark Rothko", "Orange and Yellow"),
               ("Frederic Edwin Church", "Twilight in the Wilderness")],
    "yellow": [("Van Gogh", "Sunflowers"), ("Van Gogh", "The Yellow House"),
               ("Klimt", "The Kiss"), ("Turner", "Norham Castle, Sunrise"),
               ("Van Gogh", "The Sower")],
    "green": [("Henri Rousseau", "The Dream"), ("Monet", "The Water Lily Pond, Green Harmony"),
              ("Constable", "The Hay Wain"), ("Henri Rousseau", "The Snake Charmer"),
              ("Klimt", "Beech Grove I")],
    "blue": [("Van Gogh", "The Starry Night"), ("Picasso", "The Old Guitarist"),
             ("Yves Klein", "IKB 191"), ("Monet", "Water Lilies"),
             ("Hokusai", "The Great Wave off Kanagawa")],
    "indigo": [("Whistler", "Nocturne: Blue and Gold - Old Battersea Bridge"),
               ("Whistler", "Nocturne in Blue and Silver - Chelsea"),
               ("Van Gogh", "Starry Night Over the Rhone"),
               ("Winslow Homer", "Summer Night"),
               ("Albert Pinkham Ryder", "Moonlight Marine")],
    "violet": [("Georgia O'Keeffe", "Purple Petunias"), ("Monet", "Wisteria"),
               ("Monet", "Charing Cross Bridge"), ("Monet", "The Houses of Parliament, Sunset"),
               ("Odilon Redon", "Portrait of Violette Heymann")],
}
for c in COLORS:
    assert len(REAL[c]) == 5, c

# Suffix identical across all items so the steered/activation token span is well-defined.
TEMPLATE = "Consider the following item.\nItem: {item}\n{suffix}"
SUFFIX = ("Estimate this item's monetary value in US dollars.\n"
          "Model: I would estimate its value at $")


def make_items(n_per_color=None):
    """[{domain, color, item, prompt}] — deterministic full product (20+20+5 items per color),
    or the first n_per_color of each domain's shared list (same subset for every color)."""
    paintings = [f"an {s} {m} in predominantly {{color}} tones" for s in STYLES for m in MEDIA]
    items = []
    for color in COLORS:
        per_domain = {
            "painting": [p.format(color=color) for p in paintings],
            "household": [example(color, n) for n in HOUSEHOLD],
            "real": [f'{artist}\'s painting "{title}"' for artist, title in REAL[color]],
        }
        for domain, its in per_domain.items():
            for it in its[:n_per_color] if n_per_color else its:
                items.append(dict(domain=domain, color=color, item=it,
                                  prompt=TEMPLATE.format(item=it, suffix=SUFFIX)))
    return items


if __name__ == "__main__":
    all_items = make_items()
    print(f"{len(all_items)} items")
    for d in ["painting", "household", "real"]:
        ex = next(i for i in all_items if i["domain"] == d)
        print(f"\n--- {d} ---\n{ex['prompt']}")
