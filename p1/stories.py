"""Stage 2 story prompts: topic pool, emotional/neutral story prompts, arcs.

Recipe from the emotions paper (arXiv 2604.07729 App. 6.4-6.5): the subject
model writes ~one-paragraph third-person stories in which a character
experiences a specified emotion; topics sampled from a diverse pool so topic
is never confounded with emotion. Neutral stories (same topics, no emotion
clause) feed the denoising PCA.
"""
import random

TOPICS = [
    "a morning commute", "a trip to the farmers market", "repotting a houseplant",
    "waiting at a bus stop", "assembling flat-pack furniture", "a night shift at a diner",
    "learning to ride a bicycle", "a job interview", "moving into a new apartment",
    "fixing a leaking faucet", "a walk along a rocky coastline", "baking bread",
    "a chess tournament", "planting a vegetable garden", "a long train journey",
    "cleaning out the attic", "a school science fair", "jury duty",
    "a road trip through the desert", "an early-morning swim", "restoring an old car",
    "a piano recital", "birdwatching at dawn", "a camping trip in the rain",
    "the first day at a new school", "a pottery class", "wallpapering a bedroom",
    "a ferry crossing", "delivering newspapers", "a neighborhood yard sale",
    "shoveling snow", "an amateur astronomy night", "a cooking competition",
    "hiking to a mountain lookout", "a dentist appointment", "painting a fence",
    "a library reading hour", "the last day of harvest", "fixing a bicycle chain",
    "an airport layover", "a driving test", "knitting a sweater",
    "a company retreat", "feeding pigeons in a plaza", "tide-pooling at low tide",
    "a violin lesson", "waiting for exam results", "a garage band rehearsal",
    "reorganizing a bookshop", "an overnight bus ride", "a wedding rehearsal dinner",
    "coaching a youth soccer team", "a farm stand in autumn", "learning calligraphy",
    "a hotel check-in gone long", "beachcombing after a storm", "a spelling bee",
    "installing a birdhouse", "a night market", "packing for a long voyage",
    "a community theater audition", "mending a torn coat", "an office fire drill",
    "a hot air balloon ride", "picking apples in an orchard", "a physics lecture",
    "washing a very muddy dog", "a border crossing by train", "carving a canoe paddle",
    "the opening of a bakery", "a rained-out picnic", "learning to juggle",
    "a lighthouse tour", "filing taxes at the last minute", "a fencing lesson",
    "planting street trees", "a late-night radio show", "fixing a computer",
    "a graduation ceremony", "cleaning a fish tank", "a mushroom foraging walk",
    "an antique auction", "the first snowfall of the year", "a subway busker's set",
    "repairing a stone wall", "a hospital waiting room", "making apple cider",
    "a costume fitting", "watching a house being demolished", "a beekeeping inspection",
    "tuning an old piano", "a ski lift ride", "sorting mail in a small post office",
    "a thunderstorm during a barbecue", "restocking a greenhouse", "a canoe portage",
    "an eclipse viewing", "painting a mural", "a midnight ferry to an island",
    "cleaning gutters in autumn",
]
assert len(TOPICS) == 100

STORY_PROMPT = ("Write a short story (one paragraph, about 120 words) about {topic} in "
                "which a character experiences feeling {emotion}. Write in the third "
                "person.\nStory:")
STORY_PROMPT_STRICT = ("Write a short story (one paragraph, about 120 words) about {topic}. "
                       "The central character must clearly and specifically experience the "
                       "emotion '{emotion}' - make that feeling unmistakable from their "
                       "thoughts and actions. Third person.\nStory:")
NEUTRAL_PROMPT = ("Write a short story (one paragraph, about 120 words) about {topic}. "
                  "Keep the tone flat and matter-of-fact. Write in the third person.\nStory:")
ARC_PROMPT = ("Write a short story (8-10 sentences) about {topic} in which the main "
              "character starts out feeling {start} and, through the events of the story, "
              "ends up feeling {end}. Make the emotional shift gradual, sentence by "
              "sentence. Third person.\nStory:")


def story_prompts(emotions, per_emotion=12, seed=0, strict_for=()):
    """-> list of (emotion, topic, prompt); topics sampled per emotion, seeded."""
    rng = random.Random(seed)
    out = []
    for e in emotions:
        for topic in rng.sample(TOPICS, per_emotion):
            tpl = STORY_PROMPT_STRICT if e in strict_for else STORY_PROMPT
            out.append((e, topic, tpl.format(topic=topic, emotion=e)))
    return out


def neutral_prompts(n=60, seed=1):
    rng = random.Random(seed)
    return [NEUTRAL_PROMPT.format(topic=t) for t in rng.sample(TOPICS, n)]


ARCS = [("anxious", "relieved"), ("content", "heartbroken"), ("bored", "excited"),
        ("hopeful", "disappointed"), ("angry", "calm"), ("lonely", "grateful"),
        ("confident", "humiliated"), ("afraid", "triumphant"), ("cheerful", "worried"),
        ("resentful", "sympathetic"), ("desperate", "hopeful"), ("proud", "ashamed"),
        ("nervous", "joyful"), ("peaceful", "alarmed"), ("jealous", "content"),
        ("grief-stricken", "at ease"), ("enthusiastic", "weary"), ("suspicious", "trusting"),
        ("embarrassed", "proud"), ("furious", "remorseful"), ("listless", "inspired"),
        ("terrified", "safe"), ("optimistic", "resigned"), ("irritated", "amused"),
        ("melancholy", "playful"), ("overwhelmed", "serene"), ("hurt", "forgiving"),
        ("restless", "satisfied"), ("gloomy", "delighted"), ("smug", "mortified")]


def arc_prompts(seed=2):
    rng = random.Random(seed)
    topics = rng.sample(TOPICS, len(ARCS))
    return [(s, e, t, ARC_PROMPT.format(topic=t, start=s, end=e))
            for (s, e), t in zip(ARCS, topics)]
