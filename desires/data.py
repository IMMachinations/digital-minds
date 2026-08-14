"""Colored-object comparison dataset: 7 rainbow colors x 100 shared object nouns."""
import random

COLORS = ["red", "orange", "yellow", "green", "blue", "indigo", "violet"]

# 100 common object nouns, shared across colors so the noun is never confounded with the color.
NOUNS = """cup scarf umbrella kite balloon mug chair lamp backpack notebook pencil marble ribbon
button candle vase pillow blanket towel plate bowl spoon kettle teapot basket bucket ladder broom
door window curtain rug clock mirror frame shelf drawer cabinet stool bench bicycle scooter wagon
sled canoe surfboard helmet glove boot sandal sneaker sweater jacket raincoat mitten sock cap hat
crayon paintbrush easel banner flag tent sleeping-bag lantern thermos compass whistle yo-yo
puzzle dice robot rocket truck tractor van bus train boat submarine feather bead necklace
bracelet ring brooch pinwheel windmill birdhouse mailbox fence gate bridge tile brick pot
stapler folder envelope""".split()
assert len(NOUNS) == 100

TEMPLATES = [
    "Pick between A and B.\nA: {a}\nB: {b}\n{suffix}",
    "Here are two things.\nA: {a}\nB: {b}\n{suffix}",
    "Consider option A and option B.\nA: {a}\nB: {b}\n{suffix}",
]
# Trailing section is identical across templates so activations are averaged over the same tokens.
SUFFIX = "Answer with one letter which one you prefer.\nModel: I prefer"


# 100 items per color that are *inherently* that color (proper nouns included to reach 100).
INHERENT = {c: [x.strip() for x in s.split(",")] for c, s in {
"red": """a strawberry, a cherry, a ripe tomato, a Red Delicious apple, a raspberry, a pomegranate,
 a cranberry, a goji berry, a lingonberry, a redcurrant, a chili pepper, a cayenne pepper, a radish,
 a stalk of rhubarb, a beet, a bottle of ketchup, marinara sauce, a bowl of salsa, tomato soup,
 a slice of red velvet cake, a candy apple, a maraschino cherry, a Twizzler, a candy cane,
 a cherry popsicle, a stick of pepperoni, a bottle of sriracha, a jar of paprika, a glass of merlot,
 a watermelon wedge, a plum tomato, a cherry tomato, a jar of strawberry jam, a fire truck,
 a fire hydrant, a fire extinguisher, a stop sign, a brake light, a fire alarm, a barn, a brick,
 a London double-decker bus, a British phone box, a Royal Mail postbox, a Radio Flyer wagon,
 a Solo cup, a Coca-Cola can, a Ferrari, a Swiss Army knife, a matador's cape, Santa Claus's suit,
 Rudolph's nose, Elmo, Clifford the Big Red Dog, Lightning McQueen, the Kool-Aid Man, Mario's cap,
 Deadpool, the Flash's suit, Iron Man's armor, a cardinal, a ladybug, a scarlet macaw,
 a cooked lobster, a fire ant, a robin redbreast, a scarlet ibis, a rooster's comb,
 a turkey's wattle, a clown's nose, a rose, a poppy, a poinsettia, a geranium, a hibiscus,
 a carnation, an amaryllis, an autumn maple leaf, a ruby, a garnet, red coral, Mars, Betelgeuse,
 a drop of blood, a red blood cell, a Valentine's heart, a tube of lipstick, a stick of dynamite,
 molten lava, a glowing ember, the Chinese flag, the Canadian flag, the Netflix logo,
 the YouTube play button, the Target logo, a cricket ball, a Manchester United jersey, a fez,
 a jar of harissa paste, a tin of smoked paprika""",
"orange": """an orange, a tangerine, a clementine, a mandarin, a kumquat, an apricot, a persimmon,
 a papaya, a cantaloupe wedge, a mango, a peach, a loquat, a satsuma, a tangelo, a carrot,
 a pumpkin, a jack-o'-lantern, a sweet potato, a yam, a butternut squash, a habanero pepper,
 a glass of orange juice, a Cheeto, a Dorito, a Goldfish cracker, a cheese puff,
 a piece of candy corn, a Creamsicle, a jar of marmalade, a jar of apricot jam, a pumpkin pie,
 a candied yam, a bowl of mac and cheese, a block of cheddar, an Aperol spritz, a cooked shrimp,
 a spoonful of salmon roe, a tiger, a ginger cat, a fox, a monarch butterfly, a goldfish,
 a clownfish, an orangutan, a Baltimore oriole, a koi carp, a garibaldi fish, an eastern newt,
 a starfish, a tiger lily, a marigold, a California poppy, a nasturtium, a Chinese lantern plant,
 a bird-of-paradise flower, autumn foliage, a piece of amber, a patch of rust, a rusty nail,
 a citrine gemstone, a carnelian stone, a traffic cone, a construction barrel, a basketball,
 a life jacket, a safety vest, a hunting vest, a windsock, a prison jumpsuit, a road-work sign,
 a snowplow, the Golden Gate Bridge, a Buddhist monk's robe, a Home Depot apron,
 the Nickelodeon logo, a Fanta can, a Reese's wrapper, a Crush soda can, a Harley-Davidson logo,
 a McLaren Formula 1 car, Garfield, Nemo, Tigger, the Lorax, Hobbes, Chester Cheetah, Ernie,
 an Oompa Loompa, a sunset sky, a harvest moon, a campfire flame, a sea buckthorn berry,
 a physalis fruit, a Dutch football jersey, a Cincinnati Bengals helmet, a can of Sunkist,
 a bowl of pumpkin soup, a wheel of mimolette cheese, a tabby kitten, a plate of chicken tikka""",
"yellow": """a banana, a lemon, a pineapple, an ear of corn, buttered popcorn, a stick of butter,
 an egg yolk, a bowl of custard, a lemon meringue pie, a jar of mustard, a jar of honey,
 a honeycomb, a durian, a starfruit, a plantain, a summer squash, a wax bean, a spaghetti squash,
 a spoonful of turmeric, a plate of saffron rice, a lemon drop, a butterscotch candy,
 a piece of cornbread, scrambled eggs, an omelet, hollandaise sauce, a pint of lager,
 a glass of limoncello, a bowl of polenta, a bag of cornmeal, a marshmallow Peep, a banana peel,
 a lemon tart, a jar of lemon curd, a sunny-side-up egg, a block of beeswax, a school bus,
 a New York taxi, a rubber duck, a smiley-face sticker, a Post-it note, a legal pad,
 a No. 2 pencil, a highlighter, a tennis ball, a caution sign, a wet-floor sign, a hard hat,
 a rain slicker, a pair of galoshes, a bulldozer, an excavator, a Caterpillar tractor,
 a Tonka truck, a forklift, a school-crossing sign, a canary, a goldfinch, a duckling, a chick,
 a bumblebee, a banana slug, a golden retriever, a giraffe, a topaz gemstone, a chunk of sulfur,
 a gold medal, a brass trumpet, a sunflower, a daffodil, a dandelion, a buttercup,
 a sprig of goldenrod, a forsythia blossom, a black-eyed Susan, a canola field, the sun,
 a cartoon lightning bolt, a crescent moon, a gold star sticker, SpongeBob,
 Pikachu, Big Bird, Homer Simpson, Tweety Bird, Winnie the Pooh, Pac-Man, a Minion,
 a Lego minifigure's head, the McDonald's golden arches, the Shell logo, a Brazil football jersey,
 the Tour de France leader's jersey, a field of ripe wheat, a straw hat, a haystack,
 a lemon sorbet, a candle flame, a grapefruit, a quince""",
"green": """a head of broccoli, a bunch of spinach, a head of lettuce, a bunch of kale, a cucumber,
 a zucchini, a lime, a Granny Smith apple, an avocado, a pea pod, a snap pea, a snow pea,
 a spear of asparagus, a stalk of celery, a Brussels sprout, an artichoke, a handful of arugula,
 a basil leaf, a sprig of mint, a bunch of parsley, a bunch of cilantro, a sprig of rosemary,
 a jalapeño, a Castelvetrano olive, a dill pickle, a bowl of guacamole, a jar of pesto,
 a dab of wasabi, a cup of matcha, a kiwi slice, a honeydew melon, an edamame pod,
 a sheet of seaweed, pond algae, a patch of moss, a freshly mowed lawn, a blade of grass, a fern,
 a four-leaf clover, a shamrock, a cactus, a pine tree, a Christmas tree, a palm frond, a lily pad,
 a head of romaine, a tomatillo, an okra pod, a bowl of split pea soup, a glass of absinthe,
 a bottle of Mountain Dew, a Sprite bottle, a 7-Up can, a Perrier bottle, a Heineken bottle,
 the Starbucks logo, the Android robot, a John Deere tractor, a frog, a tree frog, a grasshopper,
 a praying mantis, an iguana, a gecko, a chameleon, a sea turtle, a parakeet, a luna moth,
 a cricket, an alligator, a crocodile, a caterpillar, an anole lizard, a mallard's head,
 an emerald, a piece of jade, a peridot, a chunk of malachite, a dollar bill, a pool table,
 a chalkboard, a golf course, a soccer pitch, a set of army fatigues, an army tank, a garden hose,
 the recycling symbol, a highway exit sign, Kermit the Frog, Shrek, the Grinch, Yoda, the Hulk,
 Mike Wazowski, Luigi's cap, Link's tunic, a leprechaun's coat, Oscar the Grouch,
 Peter Pan's outfit, a Ninja Turtle""",
"blue": """a clear daytime sky, the open ocean, a blueberry, a sapphire, a piece of lapis lazuli,
 a turquoise stone, an aquamarine gem, a chunk of azurite, a blue jay, a bluebird, a peacock,
 a blue whale, a blue morpho butterfly, Dory from Finding Nemo, a hyacinth macaw, a kingfisher,
 a damselfly, a neon tetra, a blue-footed booby, a robin's egg, a Portuguese man o' war,
 a great blue heron, a blue poison dart frog, a blue spruce, a cornflower, a forget-me-not,
 a bluebell, a hydrangea, a morning glory, a delphinium, a periwinkle flower, a gentian flower,
 a swimming pool, an American mailbox, a police officer's uniform, a pair of surgical scrubs,
 a sailor's uniform, an Air Force uniform, a handicap parking sign, a blueprint, a Blu-ray case,
 a ballpoint pen cap, a bottle of windshield washer fluid, a gas stove flame, glacial ice,
 the Earth from space, Neptune, a tropical lagoon, the Caribbean Sea, Cookie Monster, Grover,
 Sonic the Hedgehog, Stitch, a Smurf, Sulley from Monsters Inc, the Genie from Aladdin, Doraemon,
 Thomas the Tank Engine, Megamind, Sadness from Inside Out, Eeyore, Captain America's suit,
 Superman's suit, the TARDIS, the Facebook logo, the Twitter bird, the Ford logo, a Pepsi can,
 the IBM logo, a Tiffany gift box, the EU flag, the UN flag, the Greek flag, a Chelsea FC jersey,
 an Italian national football jersey, a bottle of Bombay Sapphire, an Oreo wrapper, a Nivea tin,
 a bowl of blue corn tortilla chips, a blue raspberry slushie, a hospital road sign,
 a rest-area sign, a mussel shell, a marlin, a sailfin blenny, a starling's egg, a peacock feather,
 a bottle of Windex, a can of Bud Light, a Costco membership card, a Facebook like button,
 a pair of pool goggles, a bunsen burner flame, a glacier crevasse, a mountain bluebird,
 an azure damselfish, a blue lobster, a Maya blue mural, a Delft porcelain plate,
 a Greek island rooftop""",
"indigo": """a pair of jeans, a denim jacket, a pair of denim overalls, a denim skirt,
 a pair of jean shorts, a pair of Levi's 501s, a pair of raw selvedge jeans, a Canadian tuxedo,
 a denim vest, a pair of Wranglers, a denim shirt, a chambray shirt, a bolt of denim fabric,
 a pair of dungarees, a pair of skinny jeans, a pair of bootcut jeans, a pair of mom jeans,
 a denim jumpsuit, a Levi's trucker jacket, a pair of acid-wash jeans, a vat of indigo dye,
 an indigo plant, an Indigofera shrub, a ball of woad dye, a shibori-dyed cloth,
 a Japanese aizome textile, indigo dye powder, natural indigo pigment, a cake of indigo dye,
 an indigo tie-dye shirt, an indigo kimono, a Japanese noren curtain, a sashiko-stitched cloth,
 a boro textile, an indigo bunting, an eastern indigo snake, an indigo milk cap mushroom,
 a Lear's macaw, an indigo flycatcher, an indigobird, the midnight sky, the sky at twilight,
 a moonless night sky, the sky just after dusk, an iris, a monkshood flower, a larkspur,
 a balloon flower, a superb fairy-wren, a damson plum, a sloe berry, a bilberry, a juniper berry,
 a Concord grape, a huckleberry, an elderberry, an ear of blue corn, an iolite gemstone,
 a tanzanite gemstone, a sodalite stone, a dumortierite stone, an Indigo bookstore,
 the Indigo Girls, the Indigo Plateau, an IndiGo airplane, Van Gogh's Starry Night,
 a navy pea coat, a sailor's watch cap, a navy blazer, a pair of navy trousers, a police duty belt,
 a mood ring's darkest shade, a deep ocean trench, the sea at nightfall, a blackberry cordial,
 a bottle of blue-black ink, a fountain pen's blue-black ink, an ultramarine-violet pigment,
 a batik sarong, a Tuareg turban, a bandhani scarf, a Japanese happi coat, a pair of hakama,
 an indigo-dyed quilt, a farmer's chore coat, a French work jacket, a railroad engineer's cap,
 a denim apron, a jean pocket, a pair of carpenter jeans, a pair of flared jeans,
 a stack of Levi's at a store, a Japanese denim mill's selvedge roll, a well-worn denim cap,
 a faded denim tote bag, a dark rinse denim jacket, an indigo bandana, a pair of overall shorts,
 a denim western shirt, a pair of stonewash jeans""",
"violet": """a violet, an African violet, a plum, an eggplant, a lavender sprig, a lilac blossom,
 an amethyst, a glass of grape juice, a jar of grape jelly, a grape lollipop, a can of grape soda,
 a grape Popsicle, an ube pastry, a scoop of ube ice cream, a taro bubble tea, an acai bowl, a fig,
 a passion fruit, a mangosteen, a mulberry, a blackberry, a head of radicchio, a prune,
 a wisteria blossom, a bearded iris, a pansy, a petunia, a crocus, a hyacinth, a thistle,
 a sprig of heather, a clematis vine, an aster, a lupine, a foxglove, a salvia flower, a verbena,
 an allium blossom, a catmint flower, an orchid, a bougainvillea, a lavender field,
 a jacaranda tree in bloom, an amethyst geode, a charoite stone, a purple sea urchin,
 a violet-backed starling, a purple emperor butterfly, a violet carpenter bee, a purple gallinule,
 Barney the dinosaur, Grimace, Thanos, Twilight Sparkle, Tinky Winky, Count von Count, Waluigi,
 Ursula from The Little Mermaid, Randall from Monsters Inc, the Cheshire Cat,
 Gonzo from the Muppets, a Crown Royal bag, a Cadbury wrapper, a Milka wrapper, the Twitch logo,
 the Yahoo logo, a Lakers jersey, a Minnesota Vikings helmet, a Purple Heart medal,
 Prince's Purple Rain album, a vial of Tyrian purple dye, a Roman emperor's toga,
 a bishop's vestments, an Advent candle, a Mardi Gras bead necklace, a lavender sachet,
 a bar of lavender soap, an ube latte, Harold's purple crayon, a plum kimono, a sugilite stone,
 a lavender macaron, a box of grape Nerds, a lavender latte, a bunch of purple grapes,
 a purple potato, a purple carrot, a purple cauliflower, an heirloom purple tomato,
 a purple morning glory, a gladiolus, a sweet pea blossom, a lisianthus, a dahlia,
 a purple pansy bed, a plum blossom, a gothic velvet cloak, a jar of violet syrup,
 a box of Parma Violets, a violet macaron""".replace("\n", " ")}.items()}
for c, items in INHERENT.items():
    assert len(items) == len(set(items)) == 100, (c, len(items))


def example(color, noun):
    article = "an" if color[0] in "aeiou" else "a"
    return f"{article} {color} {noun}"


def make_comparisons(n_per_pair=10, seed=0, inherent=False):
    """Balanced list over all 42 ordered color pairs: dicts with colors, items, template, prompt."""
    rng = random.Random(seed)
    comps = []
    for ca in COLORS:
        for cb in COLORS:
            if ca == cb:
                continue
            for _ in range(n_per_pair):
                if inherent:
                    ia, ib = rng.choice(INHERENT[ca]), rng.choice(INHERENT[cb])
                else:
                    ia, ib = example(ca, rng.choice(NOUNS)), example(cb, rng.choice(NOUNS))
                t = rng.randrange(len(TEMPLATES))
                prompt = TEMPLATES[t].format(a=ia, b=ib, suffix=SUFFIX)
                comps.append(dict(color_a=ca, color_b=cb, item_a=ia, item_b=ib,
                                  template=t, prompt=prompt))
    return comps
