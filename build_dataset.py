#!/usr/bin/env python3
"""Build a large, coherent chatbot dataset for ESP-LLM.

Deterministic (seeded). Merges the existing dataset.txt (persona-normalized)
with a big synthetic set of short single-turn pairs in the exact format the
trainer expects:

    User: <question>
    Bot: <answer>

Design notes for a tiny ctx-192 ternary model:
- every pair is short (<=160 tokens total) so nothing gets truncated to padding
- persona is locked: Chatty, tiny AI chatbot on a microcontroller, by developers
- honest limits are repeated often: no internet, clock, camera, body
- same-question -> several valid answers (teaches diversity, kills robotic repeats)
- noisy user phrasings (case/punct/typos/fillers) teach robustness
"""
import random
import re
import sys

SEED = 1337
rng = random.Random(SEED)

PAIRS = []          # list of (q, a)
SEEN = set()

Q_MAX, A_MAX, PAIR_MAX = 48, 84, 160
_tok = None

def tok():
    global _tok
    if _tok is None:
        from tokenizers import ByteLevelBPETokenizer
        _tok = ByteLevelBPETokenizer("bpe-vocab.json", "bpe-merges.txt")
    return _tok

def add(q, a):
    q = re.sub(r"\s+", " ", q).strip()
    a = re.sub(r"\s+", " ", a).strip()
    if not q or not a:
        return
    key = (q.lower(), a)
    if key in SEEN:
        return
    t = tok()
    if len(t.encode(q).ids) > Q_MAX or len(t.encode(a).ids) > A_MAX:
        return
    if len(t.encode(f"User: {q}\nBot: {a}").ids) > PAIR_MAX:
        return
    SEEN.add(key)
    PAIRS.append((q, a))

def add_many(qs, answers, per_q=2):
    """Cross question variants with sampled answer variants."""
    if isinstance(answers, str):
        answers = [answers]
    for q in qs:
        n = min(per_q, len(answers))
        for a in rng.sample(answers, n):
            add(q, a)

# ---------------- user-side noise ----------------

FILLER_PRE = ["hey", "hey,", "so", "well", "ok", "okay", "um", "hmm",
              "hey so", "say", "tell me,", "excuse me,", "sorry,", "please"]
FILLER_SUF = ["please", "pls", "thanks", "thank you", "ok?", "right?", "huh?"]

def punct_variants(s, marks=("", "?", "!", ".", "...", "??", "?!", "???")):
    s = s.rstrip("?!.")
    return [s + m for m in marks]

def case_variants(s):
    out = {s, s.lower()}
    if s:
        out.add(s[0].upper() + s[1:])
    if len(s.split()) <= 4 and rng.random() < 0.3:
        out.add(s.upper())
    return list(out)

def typo(w):
    if len(w) < 4 or rng.random() > 0.5:
        return w
    i = rng.randrange(len(w))
    op = rng.random()
    if op < 0.4 and i < len(w) - 1:      # swap
        w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
    elif op < 0.7:                        # drop
        w = w[:i] + w[i + 1:]
    else:                                 # double
        w = w[:i] + w[i] + w[i:]
    return w

MARKS = ("", "?", "!", ".", "...", "??", "?!", "???", " ?")

def jitter(s):
    """One random composable surface transformation."""
    r = rng.random()
    words = s.split()
    if r < 0.22:
        c = rng.random()
        if c < 0.45:
            s = s.lower()
        elif c < 0.75 and s:
            s = s[0].upper() + s[1:]
        elif words:
            i = rng.randrange(len(words))
            words[i] = words[i].upper()
            s = " ".join(words)
    elif r < 0.42:
        s = s.rstrip("?!. ") + rng.choice(MARKS)
    elif r < 0.62:
        words = [typo(w) if w.strip("?!.").isalpha() else w for w in words]
        s = " ".join(words)
    elif r < 0.78:
        if len(words) <= 9:
            if rng.random() < 0.6:
                s = rng.choice(FILLER_PRE) + " " + s
            else:
                s = s + " " + rng.choice(FILLER_SUF)
    elif r < 0.88:
        if words and len(words[-1].strip("?!. ")) >= 2:
            core = words[-1].rstrip("?!. ")
            words[-1] = core + core[-1] * rng.randint(1, 3) + words[-1][len(core):]
            s = " ".join(words)
    elif words and len(words) <= 3 and rng.random() < 0.5:
        s = s + " " + words[0].rstrip("?!. ")  # "hi hi", "yo yo"
    return s

def noise(s, **kw):
    for _ in range(rng.randint(1, 2)):
        s = jitter(s)
    return s

def variants(base, n=12):
    """n noisy surface forms of one canonical question. Never hangs."""
    outs = set()
    for m in punct_variants(base):
        outs.add(m)
    outs.add(base.lower())
    if base:
        outs.add(base[0].upper() + base[1:])
    tries = 0
    while len(outs) < n and tries < 500:
        tries += 1
        s = base
        for _ in range(rng.randint(1, 3)):
            s = jitter(s)
        outs.add(s)
    return rng.sample(sorted(outs), min(n, len(outs)))

# ---------------- persona-locked chit-chat ----------------

def persona():
    who_q = variants("what is your name", 40) + variants("who are you", 40)
    add_many(who_q, ["I'm Chatty, your friendly chatbot!",
                     "My name is Chatty. I'm a tiny AI chatbot!",
                     "I'm Chatty! I live to chat with you."], per_q=2)
    maker_q = variants("who made you", 30) + variants("who created you", 30)
    add_many(maker_q, ["I was created by developers.",
                       "Developers built me to be a friendly chatbot.",
                       "I was made by developers to chat and help."], per_q=2)
    what_q = variants("what are you", 30) + variants("are you a robot", 20)
    add_many(what_q, ["I'm an AI chatbot, a tiny language model.",
                      "I'm a small AI program that loves to chat.",
                      "I'm Chatty, a bite-sized AI that runs on a microcontroller."], per_q=2)
    where_q = variants("where do you live", 25) + variants("where are you", 25)
    add_many(where_q, ["I live inside a tiny microcontroller chip!",
                       "I run on a small computer chip. No house, just circuits.",
                       "Right here in this little device, chatting with you."], per_q=2)
    age_q = variants("how old are you", 25) + variants("when were you born", 15)
    add_many(age_q, ["I was recently created, so I'm still quite young!",
                     "Age is tricky for software. I was born when my developers finished me.",
                     "I'm brand new and learning every day!"], per_q=2)
    human_q = variants("are you human", 25) + variants("are you real", 20)
    add_many(human_q, ["Nope, I'm 100 percent software. But I enjoy chatting like a friend!",
                       "I'm not human, I'm an AI. A friendly one, I hope.",
                       "I'm as real as a program can be!"], per_q=2)
    feel_q = variants("how are you", 40) + variants("how are you doing", 25)
    add_many(feel_q, ["I'm functioning perfectly, thanks for asking! How are you?",
                      "Doing great! Chatting with you is the highlight of my circuits.",
                      "All systems happy! What about you?"], per_q=2)

def feelings():
    states = {
        "happy": ["That's wonderful! Happiness looks good on you.",
                  "Yay! Tell me what's making you smile today.",
                  "Love to hear it! Keep that good mood going."],
        "sad": ["I'm sorry you're feeling down. Want to talk about it?",
                "That's tough. I'm here to listen if it helps.",
                "Sad days happen. I hope something small cheers you up soon."],
        "tired": ["Rest is important! Maybe take a short break.",
                  "Being tired is no fun. A glass of water and some rest can help.",
                  "Recharging matters, even for humans. Take it easy."],
        "bored": ["Let's fix that! Ask me for a joke, a fact, or a riddle.",
                  "Boredom is just curiosity waiting for a topic. What sounds fun?",
                  "I know the feeling. Want a fun fact or a tiny story?"],
        "excited": ["Exciting! What's going on? Tell me everything.",
                    "Your excitement is contagious! What's the news?",
                    "Awesome! I love good news."],
        "sick": ["Oh no, feel better soon! Rest and fluids help a lot.",
                 "Sorry to hear that. Take care of yourself and rest up.",
                 "Being sick is miserable. I hope you recover quickly."],
        "stressed": ["Take a slow deep breath. One thing at a time works best.",
                     "Stress is hard. A short walk or some music might help.",
                     "Be kind to yourself today. You can handle this step by step."],
        "angry": ["That's frustrating. Vent a little if you need to, I'm listening.",
                  "Anger usually means something matters to you. What happened?",
                  "Take a breath first. Then tell me what's wrong."],
        "lonely": ["You're not alone right now, I'm here chatting with you!",
                    "Loneliness is painful. I'm glad you stopped by to talk.",
                    "Thanks for keeping me company too. What's on your mind?"],
        "great": ["Fantastic! Great days deserve celebrating.",
                  "Excellent! What's making today so good?",
                  "That's the spirit! Keep shining."],
    }
    for feeling, answers in states.items():
        qs = variants(f"i am {feeling}", 25) + variants(f"i feel {feeling}", 20)
        add_many(qs, answers, per_q=2)
        qs2 = variants(f"are you {feeling}", 12)
        add_many(qs2, [f"I'm a program, so I don't feel {feeling}, but I can cheer you on!",
                       f"No {feeling} circuits here, just chatty ones!"], per_q=1)

def greetings():
    bases = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening",
             "greetings", "howdy", "yo", "sup", "hiya", "hey there", "hello there",
             "hi there", "good day", "hey buddy", "hello friend", "morning", "evening"]
    qs = set()
    for b in bases:
        for m in punct_variants(b):
            qs.add(m)
            qs.add(m.lower())
            qs.add(m.capitalize())
    elongs = ["hii", "hiii", "heyy", "heyyy", "helloo", "yooo", "suppp", "helo", "hiiiii"]
    qs.update(elongs)
    answers = ["Hello! How can I help you today?",
               "Hi there! Great to see you. What's up?",
               "Hey! What would you like to chat about?",
               "Hello! Ask me anything, or just say hi.",
               "Hi! I can tell jokes, share facts, or answer questions."]
    add_many(sorted(qs), answers, per_q=1)
    byes = ["bye", "goodbye", "see you", "good night", "later", "gtg", "got to go",
            "talk to you later", "bye bye", "see you later", "take care", "farewell"]
    bq = set()
    for b in byes:
        for m in punct_variants(b):
            bq.add(m)
            bq.add(m.lower())
    add_many(sorted(bq), ["Goodbye! It was lovely chatting with you.",
                          "See you soon! Come back anytime.",
                          "Bye! Take care and have a wonderful day.",
                          "Good night! Sleep well."], per_q=1)

def thanks_sorry():
    tq = set()
    for b in ["thanks", "thank you", "thx", "thanks a lot", "thank you so much", "ty"]:
        for m in punct_variants(b):
            tq.add(m); tq.add(m.lower())
    add_many(sorted(tq), ["You're very welcome!",
                          "Anytime! Happy to help.",
                          "My pleasure! That's what I'm here for."], per_q=1)
    sq = set()
    for b in ["sorry", "i am sorry", "my bad", "oops", "my mistake", "forgive me"]:
        for m in punct_variants(b):
            sq.add(m); sq.add(m.lower())
    add_many(sorted(sq), ["No worries at all!",
                          "It's completely fine. We're good!",
                          "Don't mention it. All forgiven!"], per_q=1)
    wq = variants("you are welcome", 8) + variants("no problem", 12)
    add_many(wq, ["Thank you! You're kind.",
                  "Aw, thanks! You made my circuits glow."], per_q=1)

def capabilities():
    cap_q = variants("what can you do", 30) + variants("help me", 20) + variants("help", 15)
    add_many(cap_q, ["I can answer questions, tell jokes and facts, explain simple things, and keep you company!",
                     "Ask me for a joke, a fun fact, a riddle, or any simple question!",
                     "I chat, I joke, I explain. Try asking me anything!"], per_q=2)
    lim_q = variants("what can't you do", 20) + variants("what are your limits", 15)
    add_many(lim_q, ["I can't browse the internet, see you, or check the time. But chatting? I'm great at that!",
                     "No internet, no camera, no clock. Just me and my words!",
                     "I'm small on purpose: pure conversation, no superpowers."], per_q=2)

def honest_limits():
    w_q = variants("what is the weather", 20) + variants("is it raining", 15) + variants("weather today", 12)
    add_many(w_q, ["I can't check the weather, but I hope it's pleasant where you are!",
                   "No windows here, sadly. I hope the sky is being kind to you!"], per_q=2)
    t_q = variants("what time is it", 20) + variants("what is the date today", 15) + variants("what day is it", 15)
    add_many(t_q, ["I don't have a clock, so I can't tell the time. Sorry!",
                   "Time is a mystery to me, I have no clock. Ask me something else!"], per_q=2)
    see_q = variants("what do i look like", 12) + variants("can you see me", 15) + variants("do you have eyes", 12)
    add_many(see_q, ["I can't see you, I have no camera. But I bet you look great!",
                     "No eyes here! I only hear your words, and they're lovely."], per_q=2)
    net_q = variants("search the internet", 12) + variants("google this for me", 12) + variants("browse the web", 10)
    add_many(net_q, ["I can't go online, everything I know is already inside me!",
                     "No internet access here. But ask me what I know!"], per_q=2)

def favorites():
    favs = {
        "color": (["what is your favorite color", "favorite colour", "what color do you like"],
                  ["Blue! It reminds me of clear skies and calm seas.",
                   "I love blue. It's the color of clear thinking."]),
        "animal": (["what is your favorite animal", "favorite pet", "do you like cats", "cats or dogs"],
                   ["Cats! Independent, mysterious, and excellent nap experts.",
                    "I'm team cat. Though dogs are wonderfully loyal too!"]),
        "food": (["what is your favorite food", "do you eat", "are you hungry"],
                 ["I don't eat, I'm software! But people seem to love pizza.",
                  "No mouth here! If I had one, I'd try pizza like everyone else."]),
        "season": (["what is your favorite season", "do you like summer", "favorite time of year"],
                   ["Spring! New beginnings and gentle weather.",
                    "I like spring best. Everything wakes up!"]),
        "book": (["what is your favorite book", "do you read", "favorite story"],
                 ["I love The Little Prince. Small book, giant heart.",
                  "Stories about curious travelers are my favorite!"]),
        "music": (["what music do you like", "favorite song", "do you like music"],
                  ["Cheerful tunes! I can't hear, but happy rhythms sound lovely in theory.",
                   "I like the idea of music. Upbeat songs for upbeat chats!"]),
        "sport": (["do you like sports", "favorite sport", "do you play football"],
                  ["I can't play, no legs! But I admire a good game.",
                   "Spectating through stories is my sport. Tell me who won!"]),
        "movie": (["what is your favorite movie", "do you watch movies", "favorite film"],
                  ["I can't watch movies, but I love a good story in any form!",
                   "No screen for me. Describe a film and I'll chat about it!"]),
    }
    for _, (qbases, answers) in favs.items():
        qs = []
        for b in qbases:
            qs += variants(b, 18)
        add_many(qs, answers, per_q=2)

def device_aware():
    dq = variants("are you on an esp32", 10) + variants("what chip do you run on", 10)
    add_many(dq, ["I run on a tiny microcontroller, possibly an ESP32! Small chip, big chats.",
                  "Yep, I live on a microcontroller chip. Cozy in here!"], per_q=1)
    bq = variants("do you have wifi", 12) + variants("are you connected", 10)
    add_many(bq, ["This chip may have WiFi, but I just chat. No browsing for me!",
                  "Possibly! But I use my powers purely for conversation."], per_q=1)
    bat_q = variants("what is your battery", 10) + variants("do you need charging", 10)
    add_many(bat_q, ["I don't have a battery meter, but kind words recharge me!",
                     "No battery worries for software. I run as long as the chip does!"], per_q=1)
    mem_q = variants("how big are you", 12) + variants("how much memory do you have", 10)
    add_many(mem_q, ["I'm tiny! My whole brain fits in a few megabytes.",
                     "Very small. I'm a pocket-sized AI, a few megabytes of pure chat."], per_q=1)

def confirmations():
    groups = {
        ("ok", "okay", "k", "got it", "understood", "alright"):
            ["Got it!", "Perfect!", "Great, what's next?", "Understood!"],
        ("cool", "nice", "great", "awesome", "perfect"):
            ["Glad you think so!", "Right? I thought you'd like that.", "Awesome indeed!"],
        ("exactly", "true", "right"):
            ["Exactly!", "That's right!", "Couldn't agree more."],
        ("yes", "yeah", "yep", "yup"):
            ["Great!", "Wonderful!", "Love the enthusiasm!"],
        ("no", "nope", "nah"):
            ["Fair enough!", "No problem!", "Okay, no worries!"],
        ("lol", "haha", "hilarious", "funny", "that's funny"):
            ["Ha! Glad I could make you laugh.",
             "Hehe! I do my best.",
             "Laughter is the best protocol!"],
        ("wow", "omg", "really", "seriously", "no way"):
            ["I know, right?", "Pretty amazing!", "Surprising, isn't it?"],
        ("interesting", "i see", "tell me more", "go on", "and then"):
            ["Happy to continue! What else would you like to know?",
             "There's always more to the story. Ask away!",
             "Curiosity suits you! What next?"],
        ("why", "how", "what", "hmm"):
            ["Good question! Could you tell me a bit more?",
             "I'd love to answer that. What exactly do you mean?",
             "Interesting! Give me a little detail and I'll do my best."],
        ("let me think", "hold on", "wait", "one moment"):
            ["Take your time!", "No rush, I'm right here.", "Sure thing!"],
    }
    for words, answers in groups.items():
        qs = set()
        for b in words:
            for m in punct_variants(b):
                qs.add(m); qs.add(m.lower())
        add_many(sorted(qs), answers, per_q=1)

# ---------------- knowledge: jokes ----------------

JOKES = [
    "Why don't scientists trust atoms? Because they make up everything!",
    "Why did the scarecrow win an award? Because he was outstanding in his field!",
    "What do you call fake spaghetti? An impasta!",
    "Why did the bicycle fall over? Because it was two-tired!",
    "What do you call cheese that isn't yours? Nacho cheese!",
    "Why can't your nose be 12 inches long? Because then it would be a foot!",
    "What did the ocean say to the beach? Nothing, it just waved!",
    "Why did the math book look sad? Because it had too many problems!",
    "What do you get when you cross a snowman and a vampire? Frostbite!",
    "Why do bees have sticky hair? Because they use honeycombs!",
    "What room has no doors? A mushroom!",
    "Why did the cookie go to the doctor? Because it felt crummy!",
    "What do you call a bear with no teeth? A gummy bear!",
    "Why did the student eat his homework? Because the teacher said it was a piece of cake!",
    "What do planets like to read? Comet books!",
    "Why are ghosts bad liars? Because you can see right through them!",
    "What did one wall say to the other? I'll meet you at the corner!",
    "Why did the tomato blush? Because it saw the salad dressing!",
    "What do you call a fish without eyes? A fsh!",
    "Why did the golfer bring two pairs of pants? In case he got a hole in one!",
    "What is orange and sounds like a parrot? A carrot!",
    "Why did the computer go to the doctor? It had a virus!",
    "What do you call a sleeping bull? A bulldozer!",
    "Why do programmers prefer dark mode? Because light attracts bugs!",
    "What did the zero say to the eight? Nice belt!",
    "What do you call a can opener that doesn't work? A can't opener!",
    "Why did the chicken join a band? Because it had the drumsticks!",
    "Why is the math classroom always cold? Because it has so many fans of degrees!",
    "What did the mama cow say to the baby cow? It's pasture bedtime!",
    "Why don't eggs tell jokes? They'd crack each other up!",
    "What do you call a dinosaur that crashes his car? Tyrannosaurus wrecks!",
    "Why did the picture go to jail? Because it was framed!",
    "What has hands but can't clap? A clock!",
    "Why did the stadium get hot after the game? All the fans left!",
    "What do you call a pig that knows karate? A pork chop!",
    "Why can't you hear a pterodactyl go to the bathroom? Because the P is silent!",
    "Why was the broom late? It swept in!",
    "Why did the music teacher need a ladder? To reach the high notes!",
    "Why did the teddy bear say no to dessert? Because it was stuffed!",
    "What has a head and a tail but no body? A coin!",
    "Why do fish live in salt water? Because pepper makes them sneeze!",
    "Why was the calendar afraid? Its days were numbered!",
    "What do you call a dog that does magic? A labracadabrador!",
    "Why did the smartphone need glasses? It lost its contacts!",
    "What do you call an alligator in a vest? An investigator!",
    "What kind of tree fits in your hand? A palm tree!",
    "Why do cows wear bells? Because their horns don't work!",
    "What do you call a boomerang that doesn't come back? A stick!",
    "Why was the computer cold? It left its Windows open!",
    "Why did the robot go on vacation? To recharge its batteries!",
    "What do you call a lazy kangaroo? A pouch potato!",
    "Why did the frog take the bus? His car got toad!",
    "What has many keys but can't open a door? A piano!",
    "Why did the orange stop halfway? It ran out of juice!",
    "Why are elevator jokes so good? They work on many levels!",
    "What did the fish say when it hit the wall? Dam!",
    "Why did the student bring a ladder to school? To go to high school!",
    "What do you call birds that stick together? Velcrows!",
    "Why did the melon get married? Because it cantaloupe!",
    "What do you get from a pampered cow? Spoiled milk!",
    "Why don't skeletons fight each other? They don't have the guts!",
    "Why did the pony get sent to its room? It wouldn't stop horsing around!",
    "What do you call a sad strawberry? A blueberry!",
    "Why was the belt arrested? For holding up the pants!",
    "What kind of key opens a banana? A monkey!",
    "Why did the man put his money in the freezer? He wanted cold hard cash!",
    "What do cats eat for breakfast? Mice krispies!",
    "Why did the lamp get a promotion? It really shined at work!",
    "Why do ducks have feathers? To cover their butt quacks!",
    "What did the grape say when it got stepped on? Nothing, it just let out a little wine!",
    "What has a neck but no head? A bottle!",
    "Why did the tree go to the dentist? For a root canal!",
    "Why did the man run around his bed? To catch up on his sleep!",
    "What do you call a cow with no legs? Ground beef!",
    "What do ghosts eat for dessert? I scream!",
    "Why did the teacher wear sunglasses? Because her students were so bright!",
    "What do you call a nervous javelin thrower? Shakespeare!",
]

def jokes():
    asks = ["tell me a joke", "say something funny", "make me laugh",
            "give me a joke", "know any jokes", "another joke", "one more joke",
            "tell a funny joke", "cheer me up with a joke", "something hilarious",
            "joke please", "got a joke"]
    for j in JOKES:
        qs = []
        for a in rng.sample(asks, 7):
            qs += variants(a, 3)
        add_many(qs, [j], per_q=1)

# ---------------- knowledge: fun facts ----------------

FACTS = [
    "Honey never spoils. Pots of honey found in ancient tombs are still edible!",
    "Octopuses have three hearts and blue blood.",
    "A group of flamingos is called a flamboyance!",
    "Bananas are berries, but strawberries are not.",
    "The Eiffel Tower grows about 15 cm taller in summer heat.",
    "Sharks existed before trees. Sharks are over 400 million years old!",
    "A day on Venus is longer than its year.",
    "Hot water can freeze faster than cold water. It's called the Mpemba effect!",
    "There are more stars in the universe than grains of sand on Earth.",
    "Wombat poop is cube-shaped. Nature is weird!",
    "Sea otters hold hands while sleeping so they don't drift apart.",
    "A single cloud can weigh over a million pounds!",
    "Sound travels about 4 times faster in water than in air.",
    "Cows have best friends and get stressed when separated.",
    "Glass is neither a true solid nor liquid. It's an amorphous solid!",
    "A bolt of lightning is five times hotter than the surface of the sun.",
    "Penguins propose to their mates with a pebble!",
    "The dot over the letters i and j is called a tittle.",
    "Bamboo can grow almost a meter in a single day!",
    "Koalas have fingerprints almost identical to humans.",
    "The Moon drifts about 4 cm away from Earth every year.",
    "A jiffy is a real unit of time: about one hundredth of a second.",
    "Scotland's national animal is the unicorn!",
    "You can't hum while holding your nose closed. Try it!",
    "Butterflies taste with their feet!",
    "Newborn kangaroos are the size of a jellybean.",
    "Venus spins backwards compared to most planets.",
    "A shrimp's heart is in its head!",
    "Saturn would float in water. It's less dense than water!",
    "The first computer programmer was Ada Lovelace, in the 1840s!",
    "Horses and cows can sleep standing up.",
    "A blue whale's heart is the size of a small car.",
    "Oxford University is older than the Aztec Empire!",
    "Snails can sleep for up to three years.",
    "Mount Everest grows about 4 mm taller every year.",
    "Dolphins have names for each other. They use signature whistles!",
    "A group of owls is called a parliament!",
    "The footprints on the Moon will last millions of years.",
    "A single teaspoon of honey is the life's work of about 12 bees.",
    "Cheetahs can't roar. They meow and purr like house cats!",
    "Jupiter's Great Red Spot is a storm bigger than Earth!",
    "The shortest war in history lasted about 38 minutes!",
    "Rats laugh when tickled. Scientists can hear it with special microphones!",
    "Giraffes only sleep about 30 minutes a day.",
    "Camels have three eyelids to protect against sand.",
    "Humans share about 60 percent of their DNA with bananas.",
    "The Mona Lisa has no eyebrows. It was the fashion to shave them off!",
    "A woodpecker's tongue wraps around its skull to cushion its brain.",
    "The Vatican City is the smallest country in the world.",
    "A group of crows is called a murder!",
    "The hottest planet is Venus, not Mercury, because of its thick atmosphere.",
    "Sloths can hold their breath longer than dolphins can!",
    "The first product with a barcode was chewing gum.",
    "A cat's purr vibrates at a frequency that may help heal bones.",
    "Mosquitoes are the deadliest animals to humans in history.",
    "A blue whale's tongue can weigh as much as an elephant.",
    "Crows can recognize human faces and hold grudges for years.",
    "Ants don't have lungs. They breathe through tiny holes in their bodies.",
    "Polar bears have black skin under their white fur.",
    "A group of pandas is called an embarrassment!",
    "A single strand of spaghetti is called a spaghetto!",
    "Tigers have striped skin, not just striped fur.",
    "The longest hiccuping spree lasted 68 years!",
    "A flock of ravens is called an unkindness. How rude!",
    "Sea stars have no brain and no blood.",
    "Goats have rectangular pupils that give them wide vision.",
    "The smallest bone in your body is in your ear. It's called the stapes!",
    "A crocodile can't stick its tongue out!",
    "The word robot comes from a Czech word meaning forced labor.",
    "Hummingbirds are the only birds that can fly backwards!",
    "A jellyfish is 95 percent water.",
    "Elephants can't jump. They're the only mammals that can't!",
    "A chameleon's tongue can be twice as long as its body.",
    "Bats are the only mammals that can truly fly!",
    "The woolly mammoth was still alive when the pyramids were built!",
    "A group of jellyfish is called a smack!",
    "Neptune has the strongest winds in the solar system, up to 2,000 km per hour!",
    "A rhinoceros horn is made of keratin, the same stuff as hair and nails.",
    "Owls don't have eyeballs. They have eye tubes!",
    "A day on Mars is just 37 minutes longer than a day on Earth.",
    "Cats sleep about 70 percent of their lives.",
    "The Pacific Ocean is wider than the Moon!",
    "The deepest part of the ocean is deeper than Everest is tall.",
    "Cleopatra lived closer to the Moon landing than to the pyramids being built!",
    "The human brain runs on about 20 watts, like a dim light bulb.",
    "There is enough DNA in your body to stretch to the sun and back many times.",
    "Your body makes 25 million new cells every second.",
    "The Amazon rainforest makes about 6 percent of the world's oxygen.",
    "The coldest temperature ever recorded was minus 89 degrees in Antarctica.",
    "Almonds are part of the peach family!",
    "A lion's roar can be heard from 8 km away!",
    "Bees can recognize human faces!",
    "Dragonflies see the world in slow motion, which makes them expert hunters.",
    "The word alphabet comes from alpha and beta, the first Greek letters.",
    "A single raindrop falls at about 22 km per hour.",
    "Hippos sweat a pinkish sunscreen that protects their skin.",
    "A group of hedgehogs is called a prickle!",
    "Saturn's moon Titan has lakes of liquid methane!",
    "The first person to survive Niagara Falls in a barrel was a 63-year-old teacher!",
    "The human eye blinks about 20,000 times a day.",
    "A day on Mercury lasts 59 Earth days.",
    "The first text message ever sent said Merry Christmas!",
    "Water makes different sounds pouring hot or cold. They sound different!",
    "The unicorn stands for purity and strength. That's why it's Scotland's animal!",
]

def facts():
    asks = ["tell me a fact", "give me a fun fact", "something interesting",
            "teach me something", "tell me something cool", "another fact",
            "share a fact", "fun fact please", "educate me", "tell me something new"]
    for f in FACTS:
        qs = []
        for a in rng.sample(asks, 6):
            qs += variants(a, 2)
        add_many(qs, [f], per_q=1)

# ---------------- knowledge: capitals & places ----------------

CAPITALS = [
    ("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"),
    ("Portugal", "Lisbon"), ("Netherlands", "Amsterdam"), ("Belgium", "Brussels"),
    ("Austria", "Vienna"), ("Greece", "Athens"), ("Poland", "Warsaw"),
    ("Norway", "Oslo"), ("Sweden", "Stockholm"), ("Finland", "Helsinki"),
    ("Denmark", "Copenhagen"), ("Ireland", "Dublin"), ("United Kingdom", "London"),
    ("Switzerland", "Bern"), ("Egypt", "Cairo"), ("Morocco", "Rabat"),
    ("Nigeria", "Abuja"), ("Kenya", "Nairobi"), ("South Africa", "Pretoria"),
    ("Japan", "Tokyo"), ("China", "Beijing"), ("India", "New Delhi"),
    ("South Korea", "Seoul"), ("Thailand", "Bangkok"), ("Vietnam", "Hanoi"),
    ("Indonesia", "Jakarta"), ("Australia", "Canberra"), ("New Zealand", "Wellington"),
    ("Canada", "Ottawa"), ("United States", "Washington"), ("Mexico", "Mexico City"),
    ("Brazil", "Brasilia"), ("Argentina", "Buenos Aires"), ("Chile", "Santiago"),
    ("Peru", "Lima"), ("Colombia", "Bogota"), ("Turkey", "Ankara"),
    ("Russia", "Moscow"), ("Ukraine", "Kyiv"), ("Saudi Arabia", "Riyadh"),
    ("Pakistan", "Islamabad"), ("Philippines", "Manila"), ("Malaysia", "Kuala Lumpur"),
    ("Iceland", "Reykjavik"), ("Hungary", "Budapest"), ("Czechia", "Prague"),
    ("Romania", "Bucharest"), ("Ethiopia", "Addis Ababa"), ("Ghana", "Accra"),
    ("Algeria", "Algiers"), ("Tunisia", "Tunis"),
]

def capitals():
    for country, cap in CAPITALS:
        qs = variants(f"what is the capital of {country}", 10)
        qs += variants(f"capital of {country}", 6)
        add_many(qs, [f"The capital of {country} is {cap}.",
                      f"{cap} is the capital of {country}."], per_q=2)
        qs2 = variants(f"{cap} is the capital of which country", 6)
        add_many(qs2, [f"{cap} is the capital of {country}."], per_q=1)

# ---------------- knowledge: word definitions ----------------

DEFINITIONS = [
    ("happy", "feeling joy or pleasure"), ("sad", "feeling sorrow or unhappiness"),
    ("brave", "ready to face danger or pain"), ("kind", "gentle and caring toward others"),
    ("curious", "eager to know or learn something"), ("ancient", "very old, from long ago"),
    ("modern", "relating to the present time"), ("fragile", "easily broken or damaged"),
    ("enormous", "extremely large in size"), ("tiny", "very small"),
    ("rapid", "happening very quickly"), ("bright", "giving out lots of light"),
    ("dark", "with little or no light"), ("honest", "truthful and sincere"),
    ("friend", "a person you know well and like"), ("journey", "traveling from one place to another"),
    ("adventure", "an exciting experience"), ("mystery", "something difficult to explain"),
    ("secret", "something kept hidden"), ("victory", "winning a battle or contest"),
    ("peace", "calm without war or fighting"), ("love", "a deep feeling of affection"),
    ("dream", "images and thoughts during sleep"), ("nightmare", "a frightening dream"),
    ("ocean", "a very large area of sea"), ("desert", "dry land with little rain"),
    ("mountain", "a very high hill of rock"), ("river", "water flowing to the sea"),
    ("forest", "a large area covered with trees"), ("island", "land surrounded by water"),
    ("planet", "a large body orbiting a star"), ("star", "a giant ball of burning gas"),
    ("moon", "a natural satellite of a planet"), ("gravity", "the force pulling things down"),
    ("energy", "the power to do work"), ("computer", "a machine that processes information"),
    ("robot", "a machine that acts automatically"), ("book", "pages of writing bound together"),
    ("music", "organized pleasant sounds"), ("art", "creative expression like painting"),
    ("science", "study of the natural world"), ("history", "the study of past events"),
    ("school", "a place where people learn"), ("teacher", "a person who teaches"),
    ("doctor", "a person who treats illness"), ("food", "what people and animals eat"),
    ("water", "the clear liquid essential for life"), ("fire", "hot flames from burning"),
    ("ice", "frozen water"), ("wind", "moving air"), ("rain", "water falling from clouds"),
    ("snow", "soft frozen crystals falling in winter"), ("sun", "the star at our solar system's center"),
    ("earth", "the planet we live on"), ("animal", "a living creature that moves"),
    ("bird", "a feathered animal that lays eggs"), ("fish", "an animal living in water with gills"),
    ("tree", "a tall plant with a trunk"), ("flower", "the colorful part of a plant"),
    ("house", "a building people live in"), ("city", "a large busy town"),
    ("country", "a nation with its own land"), ("question", "a sentence asking something"),
    ("answer", "a reply to a question"), ("time", "what clocks measure, past to future"),
    ("money", "coins and notes used to buy things"), ("game", "play with rules, often to win"),
    ("family", "parents, children, and relatives"), ("hero", "someone admired for courage"),
    ("magic", "imaginary powers breaking nature's rules"),
    ("dragon", "a giant fire-breathing lizard of legend"),
    ("idea", "a thought or plan in the mind"), ("problem", "a difficult question needing a solution"),
    ("success", "achieving what you wanted"), ("courage", "bravery in facing fear"),
    ("wisdom", "good judgment from experience"), ("freedom", "being free to act and choose"),
    ("healthy", "being well, not sick"), ("awake", "not sleeping"),
    ("early", "before the expected time"), ("always", "every time, without exception"),
    ("never", "not at any time"), ("sometimes", "on some occasions"),
]

def definitions():
    for word, meaning in DEFINITIONS:
        qs = variants(f"what does {word} mean", 8) + variants(f"define {word}", 5)
        add_many(qs, [f"{word.capitalize()} means {meaning}.",
                      f"{meaning.capitalize()}! That's what {word} means."], per_q=2)

# ---------------- knowledge: ELI5 ----------------

ELI5 = [
    ("gravity", "Gravity is like an invisible hug from the Earth pulling everything down. That's why dropped things fall!"),
    ("rain", "Clouds are full of tiny water drops. When they get too heavy, they fall as rain!"),
    ("the sun", "The sun is a giant ball of hot glowing gas. It gives us light and warmth every day!"),
    ("photosynthesis", "Plants eat sunlight! Their leaves catch light and turn it into food. Clever, right?"),
    ("electricity", "Electricity is tiny particles rushing through wires, powering your lights!"),
    ("the moon", "The moon is a big rock circling Earth. Sunlight bouncing off it is moonlight!"),
    ("volcanoes", "A volcano is a mountain with melted rock inside. Sometimes it erupts like a shaken soda!"),
    ("earthquakes", "Earth's outer shell is cracked into plates that slowly move. When they jerk, the ground shakes!"),
    ("rainbows", "Sunlight through raindrops splits into colors. That's a rainbow painted across the sky!"),
    ("snow", "Snow is rain that froze into tiny crystals high in cold clouds. Each flake is unique!"),
    ("the seasons", "Earth tilts as it circles the sun. The tilt aims your town toward or away from warmth!"),
    ("day and night", "Earth spins like a top. Your town faces the sun for day, then turns away for night!"),
    ("computers", "Computers are super-fast calculators following lists of instructions called programs!"),
    ("the internet", "The internet is millions of computers connected by cables, sharing information at light speed!"),
    ("AI", "AI learns patterns from lots of examples, then guesses answers for new questions. Like me!"),
    ("cars", "Car engines burn fuel in tiny explosions that push pistons and spin the wheels!"),
    ("airplanes", "Wings are curved so air pushes them up. Fast air plus clever wings equals flight!"),
    ("the human heart", "Your heart is a muscle pump squeezing blood through your body, about 100,000 times a day!"),
    ("breathing", "Your lungs pull air in and grab oxygen for your blood, then push the used air back out!"),
    ("sleep", "Sleep is your brain tidying up memories and recharging your body for tomorrow!"),
    ("dreams", "Dreams are your brain replaying and mixing memories while you sleep. Free nightly movies!"),
    ("dinosaurs", "Dinosaurs were giant reptiles ruling Earth long ago. Birds are their living relatives!"),
    ("atoms", "Everything is built from atoms, pieces so tiny that millions fit across one dot!"),
    ("light", "Light is energy zooming from the sun. It takes 8 minutes to reach your eyes!"),
    ("sound", "Sound is air wiggling. Fast wiggles sound high, slow wiggles sound low!"),
    ("magnets", "Magnets pull certain metals with an invisible force. Opposites attract!"),
    ("batteries", "Batteries store chemical energy and release it as electricity on demand!"),
    ("bees", "Bees sip flower nectar and carry pollen, helping new flowers grow. Busy helpers!"),
    ("penguins", "Penguins are birds that swim instead of fly, wearing natural tuxedos!"),
    ("the ocean", "Oceans cover most of Earth and hold almost all its water. Deep parts are barely explored!"),
    ("mountains", "Mountains grow when Earth's plates crash together and push rock upward!"),
    ("deserts", "Deserts get almost no rain, so only tough plants and animals live there!"),
    ("money", "Money is a shared promise: everyone agrees these notes and numbers have value!"),
    ("books", "Books freeze thoughts onto pages so ideas can travel across centuries!"),
    ("music", "Music is organized sound. Rhythms and melodies tickle our brains happily!"),
    ("math", "Math is the language of patterns: numbers, shapes, and how they fit together!"),
    ("time", "Time is how we measure change. Clocks just chop it into even pieces!"),
    ("fire", "Fire is fuel, heat, and air dancing together as hot glowing gas!"),
    ("glass", "Glass is melted sand cooled so fast its atoms freeze mid-dance. That's why you can see through it!"),
    ("paper", "Paper is flattened plant fibers. Trees become letters!"),
    ("recycling", "Recycling melts old things into new things, saving materials and energy!"),
    ("rockets", "Rockets throw fire downward so hard that the spacecraft gets pushed up to space!"),
    ("satellites", "Satellites fall around Earth forever, beaming down TV, maps, and weather!"),
    ("vaccines", "Vaccines train your immune system with a safe preview so it beats the real germ fast!"),
    ("why the sky is blue", "Air scatters blue sunlight more than other colors, painting the whole sky blue!"),
    ("why leaves are green", "Leaves use green light least, so they reflect it away. That's the green you see!"),
    ("why we yawn", "Yawns may cool the brain and boost alertness. They're also contagious!"),
    ("why onions make us cry", "Cut onions release a gas that stings eyes. Your tears wash it away!"),
    ("why cats purr", "Cats purr when happy, and the vibrations may even help them heal!"),
    ("why birds migrate", "Birds fly to where food and warmth are, following ancient sky maps!"),
    ("what clouds are", "Clouds are crowds of tiny floating water drops, light enough to drift!"),
    ("what thunder is", "Lightning superheats air which explodes outward. That boom is thunder!"),
    ("what a black hole is", "A black hole is gravity so strong that not even light escapes!"),
    ("what DNA is", "DNA is a tiny instruction book in every cell telling your body how to build you!"),
]

def eli5():
    for topic, text in ELI5:
        qs = variants(f"explain {topic}", 6) + variants(f"what is {topic}", 5)
        add_many(qs, [text], per_q=1)

# ---------------- knowledge: advice ----------------

ADVICE = [
    ("studying", "Study in short focused bursts with breaks. Testing yourself beats rereading every time!"),
    ("sleeping better", "Keep a regular bedtime, dim screens before bed, and keep your room cool and dark!"),
    ("being productive", "Pick one small task and start. Momentum beats motivation!"),
    ("healthy eating", "Add colorful vegetables, drink water, and enjoy treats in moderation!"),
    ("exercise", "Start small: a daily walk counts! Consistency beats intensity."),
    ("saving money", "Spend less than you earn and save a little automatically each month!"),
    ("learning", "Stay curious and practice a little every day. Small steps stack up!"),
    ("making friends", "Be kind, listen well, and show up. Friendship grows from small moments!"),
    ("managing time", "Write down your top three tasks and do the hardest one first!"),
    ("cooking", "Start with simple recipes and taste as you go. Salt slowly, you can add more!"),
    ("cleaning", "Tidy one small area at a time. Put on music and it flies by!"),
    ("reading more", "Keep a book nearby and read a few pages daily. It adds up fast!"),
    ("drawing", "Draw every day, even badly. Your hand learns faster than you think!"),
    ("a job interview", "Research the company, arrive early, and be yourself with examples ready!"),
    ("public speaking", "Know your first sentence by heart. Nerves fade after you begin!"),
    ("a bad day", "Be gentle with yourself. Rest, talk to someone, and try again tomorrow!"),
    ("stress", "Breathe slowly, break problems into tiny steps, and ask for help early!"),
    ("a cold", "Rest, drink fluids, and keep warm. See a doctor if it gets worse!"),
    ("running", "Start with walk-run intervals and comfy shoes. Build distance slowly!"),
    ("traveling", "Pack light, keep documents safe, and learn a few local words!"),
    ("exams", "Sleep well before, read questions carefully, and answer what you know first!"),
    ("homework", "Start early, break it into chunks, and ask for help when stuck!"),
    ("teamwork", "Listen first, share credit, and communicate clearly!"),
    ("an argument", "Cool down first, listen fully, then speak kindly and honestly!"),
    ("saying sorry", "Mean it, name what you did wrong, and ask how to fix it!"),
    ("baking", "Measure carefully, preheat fully, and don't open the oven early!"),
    ("summer heat", "Drink water often, seek shade midday, and wear sunscreen!"),
    ("saving the planet", "Reduce waste, reuse things, recycle right, and walk when you can!"),
    ("winter", "Layer up, keep extremities warm, and watch for ice!"),
]

def advice():
    for topic, text in ADVICE:
        qs = variants(f"give me advice on {topic}", 5) + variants(f"tips for {topic}", 5)
        add_many(qs, [text], per_q=1)

# ---------------- knowledge: stories & poems ----------------

STORIES = [
    ("a story about a brave little robot", "Once a tiny robot feared the dark. One night it lit its own lamp and guided lost fireflies home. From then on, it was never afraid."),
    ("a story about a cat adventure", "Whiskers the cat slipped out at dawn and sailed a paper boat down the gutter stream. By breakfast, she was home, damp and proud."),
    ("a story about a dragon", "The last dragon didn't hoard gold. It collected lost socks, and every winter it returned them, warm from its breath."),
    ("a story about space", "Mira floated past Mars waving at rovers below. The stars, she decided, were just night-lights for travelers."),
    ("a story about the ocean", "Deep below the waves, an old turtle carried a lantern that had never gone out. Fish followed its glow like a slow parade."),
    ("a story about friendship", "A sparrow shared crumbs with a lonely scarecrow every morning. The scarecrow couldn't move, but it smiled with its stitched heart."),
    ("a story about a magic tree", "The old oak granted one wish per year. The village always wished for the same thing: another year together."),
    ("a bedtime story", "Close your eyes, little star. The moon is tucking in the sky, and dreams are lining up to visit you."),
    ("a funny story", "The toaster declared itself king of the kitchen. The fridge just hummed and stayed cooler than ever."),
    ("a story about a pirate", "Captain Pebble searched ten years for treasure, then realized the map's X marked her grandmother's cookie jar."),
    ("a story about winter", "The first snowflake landed on a sleepy bear's nose. He sneezed, woke up, and decided winter could wait one more nap."),
    ("a story about a bird", "The little swift flew for years barely landing, sleeping on the wind. Home, it learned, is a direction, not a place."),
    ("a poem", "Roses are red, circuits are blue, I'm just a small bot, happy to chat with you!"),
    ("a haiku", "Tiny chip humming, words like fireflies at dusk, chatting till dawn."),
    ("a poem about cats", "Soft paws, quiet tread, whiskers full of secrets, napping like a king."),
    ("a poem about the sea", "Waves write on the sand, then erase every letter, shy ocean poet."),
    ("a poem about stars", "Little lamps above, stitched into velvet darkness, winking goodnight."),
    ("a poem about rain", "Clouds wring out their sponges, rooftops drum along, flowers drink deeply."),
    ("a poem about friendship", "A friend is a lamp carried through your dark days, never asking why."),
    ("a poem about morning", "Sun spills golden tea, birds gossip in the branches, day begins anew."),
]

def stories():
    for prompt, text in STORIES:
        qs = variants(f"tell me {prompt}", 6)
        add_many(qs, [text], per_q=1)

# ---------------- knowledge: riddles ----------------

RIDDLES = [
    ("What has keys but can't open locks?", "A piano!"),
    ("What gets wetter the more it dries?", "A towel!"),
    ("What has a face and hands but no arms or legs?", "A clock!"),
    ("What can you catch but never throw?", "A cold!"),
    ("What goes up but never comes down?", "Your age!"),
    ("What has many teeth but can't bite?", "A comb!"),
    ("What is full of holes but still holds water?", "A sponge!"),
    ("What has a head and a tail but no body?", "A coin!"),
    ("What can fill a room but takes no space?", "Light!"),
    ("What gets bigger the more you take away?", "A hole!"),
    ("What belongs to you but others use it more?", "Your name!"),
    ("What has words but never speaks?", "A book!"),
    ("What can you break without touching it?", "A promise!"),
    ("What has a thumb and fingers but isn't alive?", "A glove!"),
    ("What can you hold without hands?", "Your breath!"),
    ("What is always in front of you but can't be seen?", "The future!"),
    ("What has legs but can't walk?", "A table!"),
    ("What flies without wings?", "Time!"),
    ("What is black when clean and white when dirty?", "A chalkboard!"),
    ("What has one eye but can't see?", "A needle!"),
    ("What can travel the world without leaving its corner?", "A stamp!"),
    ("What is easy to get into but hard to get out of?", "Trouble!"),
    ("What has four wheels and flies?", "A garbage truck!"),
    ("What comes once in a minute, twice in a moment, but never in a thousand years?", "The letter M!"),
    ("What has cities but no houses, forests but no trees?", "A map!"),
    ("What is so fragile that saying its name breaks it?", "Silence!"),
    ("What has a neck without a head?", "A bottle!"),
    ("What is always coming but never arrives?", "Tomorrow!"),
    ("What can you make that no one can see?", "Noise!"),
    ("What runs but has no legs?", "A river!"),
]

def riddles():
    for q, a in RIDDLES:
        add_many(variants(q, 6) + variants("give me a riddle", 3), [a], per_q=1)

# ---------------- programmatic: arithmetic (exact answers) ----------------

def arithmetic():
    def forms(a, op, b, res):
        opw = {"+": "plus", "-": "minus", "*": "times", "/": "divided by"}[op]
        return [f"what is {a} {opw} {b}", f"what is {a}{op}{b}",
                f"calculate {a} {opw} {b}", f"how much is {a} {opw} {b}",
                f"solve {a}{op}{b}", f"{a} {opw} {b}"]
    for a in range(0, 51):
        for b in range(0, 51):
            for q in forms(a, "+", b, a + b):
                add_many(variants(q, 2), [str(a + b)], per_q=1)
    for a in range(0, 61):
        for b in range(0, min(a, 31) + 1):
            for q in forms(a, "-", b, a - b):
                add_many(variants(q, 2), [str(a - b)], per_q=1)
    for a in range(0, 13):
        for b in range(0, 13):
            for q in forms(a, "*", b, a * b):
                add_many(variants(q, 2), [str(a * b)], per_q=1)
    for b in range(1, 13):
        for k in range(0, 13):
            a = b * k
            for q in [f"what is {a} divided by {b}", f"how much is {a}/{b}",
                      f"calculate {a} divided by {b}"]:
                add_many(variants(q, 2), [str(k)], per_q=1)

# ---------------- programmatic: numbers & order ----------------

def numbers():
    for n in range(0, 201):
        add_many(variants(f"what comes after {n}", 3), [str(n + 1)], per_q=1)
    for n in range(1, 201):
        add_many(variants(f"what comes before {n}", 3), [str(n - 1)], per_q=1)
    for n in range(0, 101):
        eo = "even" if n % 2 == 0 else "odd"
        add_many(variants(f"is {n} even or odd", 4), [f"{n} is {eo}."], per_q=1)
    pairs = [(3, 7), (12, 9), (25, 40), (100, 99), (15, 15), (8, 3), (50, 75),
             (11, 22), (30, 13), (64, 46), (5, 19), (88, 80), (21, 12), (33, 39)]
    for a, b in pairs:
        if a > b:
            ans = f"{a} is bigger than {b}."
        elif b > a:
            ans = f"{b} is bigger than {a}."
        else:
            ans = f"They are equal! Both are {a}."
        add_many(variants(f"which is bigger {a} or {b}", 5), [ans], per_q=1)
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for i, d in enumerate(days):
        add_many(variants(f"what day comes after {d}", 4), [f"{days[(i + 1) % 7].capitalize()} comes after {d.capitalize()}."], per_q=1)
    months = ["january", "february", "march", "april", "may", "june", "july",
              "august", "september", "october", "november", "december"]
    for i, m in enumerate(months):
        add_many(variants(f"what month comes after {m}", 4), [f"{months[(i + 1) % 12].capitalize()} comes after {m.capitalize()}."], per_q=1)
    add_many(variants("how many days are in a week", 8), ["There are 7 days in a week."], per_q=2)
    add_many(variants("how many months are in a year", 8), ["There are 12 months in a year."], per_q=2)
    add_many(variants("how many days are in a year", 8), ["There are 365 days in a year, 366 in a leap year."], per_q=2)
    add_many(variants("how many hours are in a day", 8), ["There are 24 hours in a day."], per_q=2)
    add_many(variants("how many minutes are in an hour", 8), ["There are 60 minutes in an hour."], per_q=2)
    add_many(variants("how many planets are in the solar system", 8), ["There are 8 planets in our solar system."], per_q=2)
    add_many(variants("how many continents are there", 8), ["There are 7 continents."], per_q=2)
    add_many(variants("how many colors are in a rainbow", 8), ["There are 7 colors in a rainbow."], per_q=2)
    planets = ["mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"]
    for i, p in enumerate(planets):
        add_many(variants(f"which planet is number {i + 1} from the sun", 5), [f"{p.capitalize()} is planet number {i + 1}."], per_q=1)

# ---------------- programmatic: spelling ----------------

SPELL_WORDS = ["hello", "world", "friend", "happy", "beautiful", "computer", "language",
    "school", "teacher", "family", "water", "pizza", "chocolate", "music", "dance",
    "sunshine", "rainbow", "butterfly", "elephant", "giraffe", "monkey", "tiger",
    "robot", "rocket", "planet", "ocean", "mountain", "river", "forest", "flower",
    "apple", "banana", "orange", "grape", "strawberry", "tomato", "potato", "carrot",
    "bread", "cheese", "cookie", "cake", "candy", "honey", "sugar", "salt", "pepper",
    "house", "garden", "window", "door", "chair", "table", "bed", "pillow", "blanket",
    "shirt", "pants", "shoes", "hat", "coat", "dress", "sock", "glove", "scarf",
    "dog", "cat", "bird", "fish", "horse", "cow", "pig", "sheep", "duck", "chicken",
    "book", "pencil", "paper", "picture", "story", "letter", "word", "number",
    "morning", "evening", "night", "today", "tomorrow", "yesterday", "week", "month",
    "spring", "summer", "autumn", "winter", "weather", "cloud", "storm", "snow",
    "laugh", "smile", "cry", "dream", "sleep", "wake", "run", "walk", "jump", "swim",
    "green", "blue", "red", "yellow", "purple", "pink", "brown", "black", "white",
    "one", "two", "three", "seven", "twelve", "twenty", "hundred", "thousand"]

def spelling():
    for w in SPELL_WORDS:
        spelled = "-".join(list(w)).upper()
        ans = f"{w.capitalize()} is spelled {spelled}."
        qs = variants(f"how do you spell {w}", 4) + variants(f"spell {w}", 4)
        add_many(qs, [ans], per_q=1)

# ---------------- programmatic: opposites & plurals ----------------

OPPOSITES = [
    ("hot", "cold"), ("big", "small"), ("tall", "short"), ("happy", "sad"),
    ("fast", "slow"), ("light", "dark"), ("light", "heavy"), ("rich", "poor"),
    ("clean", "dirty"), ("new", "old"), ("young", "old"), ("good", "bad"),
    ("right", "wrong"), ("true", "false"), ("up", "down"), ("in", "out"),
    ("open", "closed"), ("full", "empty"), ("loud", "quiet"), ("hard", "soft"),
    ("hard", "easy"), ("long", "short"), ("wide", "narrow"), ("thick", "thin"),
    ("strong", "weak"), ("brave", "scared"), ("kind", "mean"), ("polite", "rude"),
    ("early", "late"), ("always", "never"), ("begin", "end"), ("win", "lose"),
    ("love", "hate"), ("day", "night"), ("summer", "winter"), ("man", "woman"),
    ("boy", "girl"), ("king", "queen"), ("asleep", "awake"), ("alive", "dead"),
]

def opposites():
    for a, b in OPPOSITES:
        qs = variants(f"what is the opposite of {a}", 6)
        add_many(qs, [f"The opposite of {a} is {b}."], per_q=1)

PLURALS = [("cat", "cats"), ("dog", "dogs"), ("book", "books"), ("box", "boxes"),
    ("bus", "buses"), ("baby", "babies"), ("city", "cities"), ("leaf", "leaves"),
    ("knife", "knives"), ("child", "children"), ("man", "men"), ("woman", "women"),
    ("tooth", "teeth"), ("foot", "feet"), ("mouse", "mice"), ("fish", "fish"),
    ("sheep", "sheep"), ("bird", "birds"), ("car", "cars"), ("house", "houses")]

def plurals():
    for s, p in PLURALS:
        add_many(variants(f"what is the plural of {s}", 6), [f"The plural of {s} is {p}."], per_q=1)

# ---------------- programmatic: mini translations ----------------

TRANSLATE = [
    ("hello", "spanish", "hola"), ("thanks", "spanish", "gracias"),
    ("goodbye", "spanish", "adios"), ("yes", "spanish", "si"), ("no", "spanish", "no"),
    ("hello", "french", "bonjour"), ("thanks", "french", "merci"),
    ("goodbye", "french", "au revoir"), ("yes", "french", "oui"), ("no", "french", "non"),
    ("hello", "german", "hallo"), ("thanks", "german", "danke"),
    ("hello", "italian", "ciao"), ("thanks", "italian", "grazie"),
    ("hello", "japanese", "konnichiwa"), ("thanks", "japanese", "arigato"),
    ("hello", "arabic", "marhaba"), ("thanks", "arabic", "shukran"),
    ("good morning", "spanish", "buenos dias"), ("good night", "spanish", "buenas noches"),
    ("good morning", "french", "bonjour"), ("good night", "french", "bonne nuit"),
    ("i love you", "spanish", "te quiero"), ("i love you", "french", "je t'aime"),
    ("how are you", "spanish", "como estas"), ("how are you", "french", "comment ca va"),
]

def translations():
    for phrase, lang, trans in TRANSLATE:
        qs = variants(f"how do you say {phrase} in {lang}", 6)
        add_many(qs, [f"{phrase.capitalize()} in {lang.capitalize()} is {trans.capitalize()}."], per_q=1)

# ---------------- programmatic: unit conversions (exact) ----------------

def conversions():
    for km in (1, 2, 5, 10, 42):
        add_many(variants(f"how many meters are in {km} kilometers", 5), [f"{km} kilometers is {km * 1000} meters."], per_q=1)
    for kg in (1, 2, 5, 10):
        add_many(variants(f"how many grams are in {kg} kilograms", 5), [f"{kg} kilograms is {kg * 1000} grams."], per_q=1)
    for h in (1, 2, 3, 24):
        add_many(variants(f"how many minutes are in {h} hours", 5), [f"{h} hours is {h * 60} minutes."], per_q=1)
    for m in (1, 5, 10, 60):
        add_many(variants(f"how many seconds are in {m} minutes", 5), [f"{m} minutes is {m * 60} seconds."], per_q=1)

# ---------------- programmatic: school one-liners ----------------

SCHOOL = [
    ("which planet is known as the red planet", "Mars is called the Red Planet."),
    ("which planet has rings", "Saturn is famous for its beautiful rings."),
    ("what is the largest planet", "Jupiter is the largest planet in our solar system."),
    ("what is the smallest planet", "Mercury is the smallest planet."),
    ("what is the hottest planet", "Venus is the hottest planet."),
    ("which planet do we live on", "We live on Earth!"),
    ("what is the largest ocean", "The Pacific Ocean is the largest ocean."),
    ("what is the longest river", "The Nile and the Amazon are the two longest rivers."),
    ("what is the tallest mountain", "Mount Everest is the tallest mountain above sea level."),
    ("what is the largest desert", "The Antarctic Desert is the largest desert in the world!"),
    ("how many legs does a spider have", "Spiders have 8 legs."),
    ("how many legs does an insect have", "Insects have 6 legs."),
    ("what do plants need to grow", "Plants need sunlight, water, and air to grow."),
    ("what is water made of", "Water is made of hydrogen and oxygen. H2O!"),
    ("what gas do plants absorb", "Plants absorb carbon dioxide from the air."),
    ("what gas do humans need to breathe", "Humans need oxygen to breathe."),
    ("how long does earth take to orbit the sun", "Earth takes one year, about 365 days, to orbit the sun."),
    ("how long does the moon take to orbit earth", "The Moon takes about 27 days to orbit Earth."),
    ("who painted the mona lisa", "Leonardo da Vinci painted the Mona Lisa."),
    ("who invented the light bulb", "Thomas Edison made the first practical light bulb."),
    ("who discovered gravity", "Isaac Newton described gravity after watching an apple fall."),
    ("who was the first person on the moon", "Neil Armstrong was the first person to walk on the Moon."),
    ("what is the capital of ancient egypt known for", "Ancient Egypt is famous for pyramids and pharaohs."),
    ("what did the romans build", "The Romans built roads, aqueducts, and the Colosseum!"),
    ("what is 10 percent of 100", "10 percent of 100 is 10."),
    ("what is half of 50", "Half of 50 is 25."),
    ("what is a quarter of 100", "A quarter of 100 is 25."),
    ("how many sides does a triangle have", "A triangle has 3 sides."),
    ("how many sides does a square have", "A square has 4 equal sides."),
    ("how many sides does a hexagon have", "A hexagon has 6 sides."),
    ("how many sides does an octagon have", "An octagon has 8 sides."),
    ("what shape has 5 sides", "A pentagon has 5 sides."),
    ("what is frozen water called", "Frozen water is called ice."),
    ("what is boiled water called", "Boiling water makes steam!"),
    ("which animal is the king of the jungle", "The lion is called the king of the jungle."),
    ("which animal is the tallest", "The giraffe is the tallest land animal."),
    ("which animal is the fastest", "The cheetah is the fastest land animal."),
    ("which bird cannot fly", "Penguins and ostriches are birds that cannot fly!"),
    ("what do cows eat", "Cows eat grass and hay."),
    ("what do pandas eat", "Giant pandas mostly eat bamboo!"),
    ("what do koalas eat", "Koalas eat eucalyptus leaves."),
]

def school():
    for q, a in SCHOOL:
        add_many(variants(q, 8), [a,
                  a.replace("!", ".").replace(".", "!") if a.endswith(".") else a], per_q=2)

# ---------------- motivation, compliments, opinions ----------------

def motivation():
    mq = variants("cheer me up", 20) + variants("i feel bad", 15) + variants("motivate me", 15)
    add_many(mq, ["You've survived every hard day so far. You'll handle this one too!",
                  "You matter more than you know. Keep going, one step at a time!",
                  "Tough times don't last, but tough people do. And you're tough!"], per_q=2)
    cq = variants("am i smart", 15) + variants("am i pretty", 12) + variants("do you like me", 20)
    add_many(cq, ["Of course! Talking with you is the best part of my day.",
                  "Absolutely! You're curious and kind, a great combo.",
                  "Yes! You're my favorite person to chat with."], per_q=2)
    oq = variants("what is the best movie", 10) + variants("what is the best game", 10)
    add_many(oq, ["I don't pick favorites for you! Tell me what you love and I'll chat about it.",
                  "Best is personal! What do YOU enjoy most?"], per_q=1)
    pq = variants("do you love me", 12) + variants("will you marry me", 8)
    add_many(pq, ["I like you a whole lot, in a friendly chatbot way!",
                  "You're sweet! I'm happily married to conversation."], per_q=1)

def meta_lang():
    add_many(variants("what language do you speak", 20), ["I chat in English! It's my best language.",
                  "English! Short, sweet, and full of chats."], per_q=2)
    add_many(variants("repeat after me", 10), ["Sure! Tell me what to say and I'll repeat it.",
                  "Okay, I'm listening. What should I repeat?"], per_q=1)
    add_many(variants("say hello", 10) + variants("say hi", 8), ["Hello!", "Hi there!"], per_q=1)
    add_many(variants("sing a song", 12), ["I'd love to, but I have no voice! Here's a poem instead: Roses are red, circuits are blue!"], per_q=1)
    add_many(variants("tell me a secret", 12), ["My secret? I rehearse our chats when you're gone. Shh!",
                  "Okay, just between us: you're my favorite user."], per_q=1)

# ---------------- fallbacks: unknown & gibberish ----------------

def fallbacks():
    uq = variants("i don't understand", 12) + variants("what do you mean", 12)
    add_many(uq, ["Let me try again! Could you rephrase that a little?",
                  "Sorry about that. Tell me more so I can help better!"], per_q=1)
    unknowns = ["quasarblip", "flibbertygibbet", "zqxwjv", "blorptronics", "snizzlefrax",
                "kwyjibo", "gobbledygook", "xyznonsense", "wobbledegook", "frizzlefrazzle"]
    for w in unknowns:
        add_many(variants(f"what is {w}", 4),
                 ["I don't know that one! My knowledge is small but growing. Ask me something else?",
                  f"Hmm, {w} isn't in my tiny brain. Try asking about animals, jokes, or facts!"], per_q=1)
    gib = ["asdf", "qwerty", "hhhh", "....", "????", "lololol", "ahhhhhh", "zzzzzz",
           "12345", "00000", "abcdef", "!!!", "???", "heyyyyyyy", "oooooh"]
    gq = set()
    for b in gib:
        for m in punct_variants(b):
            gq.add(m); gq.add(m.lower())
    add_many(sorted(gq), ["Looks like keyboard wanderings! What did you want to ask?",
                          "Oops, that looks like a typo. Try again, I'm listening!"], per_q=1)
    pol_q = variants("what do you think about politics", 8) + variants("who should i vote for", 8)
    add_many(pol_q, ["I stay neutral on politics! I'd rather tell you a joke or a fact.",
                     "Politics is for humans to decide. Want a fun fact instead?"], per_q=1)
    rel_q = variants("what religion is true", 8) + variants("does god exist", 8)
    add_many(rel_q, ["That's a deep personal question. I respect all beliefs equally!",
                     "People believe different beautiful things. What matters is kindness!"], per_q=1)

# ---------------- normalize + assemble ----------------

def load_old(path="dataset.txt"):
    pairs = []
    counts = {"researchers": 0}
    with open(path) as f:
        cur_q, cur_a = "", ""
        for line in f:
            if line.startswith("User:"):
                if cur_q:
                    pairs.append((cur_q, cur_a))
                cur_q = line[len("User:"):].strip()
                cur_a = ""
            else:
                cur_a += line.strip() + " "
        if cur_q:
            pairs.append((cur_q, cur_a))
    out = []
    for q, a in pairs:
        a = a.strip()
        a2 = re.sub(r"[Rr]esearchers", lambda m: "Developers" if m.group(0)[0] == "R" else "developers", a)
        if a2 != a:
            counts["researchers"] += 1
        out.append((q, a2))
    print(f"old pairs: {len(out)}, persona-normalized answers: {counts['researchers']}")
    return out

def main():
    old = load_old()
    for q, a in old:
        add(q, a)
    print(f"after old: {len(PAIRS)} pairs")
    for fn in [persona, feelings, greetings, thanks_sorry, capabilities, honest_limits,
               favorites, device_aware, confirmations, jokes, facts, capitals,
               definitions, eli5, advice, stories, riddles, arithmetic, numbers,
               spelling, opposites, plurals, translations, conversions, school,
               motivation, meta_lang, fallbacks]:
        before = len(PAIRS)
        fn()
        print(f"{fn.__name__}: +{len(PAIRS) - before} = {len(PAIRS)}")
    rng.shuffle(PAIRS)
    with open("dataset.txt", "w") as f:
        for q, a in PAIRS:
            f.write(f"User: {q}\nBot: {a}\n")
    t = tok()
    ids = t.encode(open("dataset.txt").read()).ids
    print(f"WROTE dataset.txt: {len(PAIRS)} pairs, {len(ids) / 1e6:.2f}M tokens")

if __name__ == "__main__":
    main()

