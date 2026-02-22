ROLES = [
    "🔪 Mafia",
    "🛡 Doctor",
    "🕵 Detective",
    "🧙 Witch",
    "🤫 Silencer",
    "👑 Mayor",
    "💣 Bomber",
    "🛡 Guardian",
    "🎯 Sniper",
    "🔮 Oracle",
    "🧛 Vampire",
    "🧟 Necromancer",
    "🎭 Trickster",
    "⚖ Judge",
    "🔥 Arsonist",
]

# Internal role ids used by the game logic.
CORE_ROLE_POOL = [
    "mafia",
    "doctor",
    "detective",
    "witch",
    "silencer",
    "mayor",
    "villager",
]

EXTRA_ROLE_POOL = [
    "bomber",
    "guardian",
    "sniper",
    "oracle",
    "vampire",
    "necromancer",
    "trickster",
    "judge",
    "arsonist",
]

ROLE_LABEL = {
    "mafia": "🔪 Mafia",
    "doctor": "🛡 Doctor",
    "detective": "🕵 Detective",
    "witch": "🧙 Witch",
    "silencer": "🤫 Silencer",
    "mayor": "👑 Mayor",
    "villager": "👤 Villager",
    "bomber": "💣 Bomber",
    "guardian": "🛡 Guardian",
    "sniper": "🎯 Sniper",
    "oracle": "🔮 Oracle",
    "vampire": "🧛 Vampire",
    "necromancer": "🧟 Necromancer",
    "trickster": "🎭 Trickster",
    "judge": "⚖ Judge",
    "arsonist": "🔥 Arsonist",
}

ROLE_INFO = {
    "mafia": "Kill one player every night.",
    "doctor": "Save one player every night.",
    "detective": "Check one player's role every night.",
    "witch": "Has 1 heal potion and 1 poison potion.",
    "silencer": "Mute one player next day.",
    "mayor": "Permanent double vote.",
    "villager": "No special power. Vote wisely!",
    "bomber": "Explosive wildcard role (flavor role).",
    "guardian": "Protective support role (flavor role).",
    "sniper": "Precision attacker role (flavor role).",
    "oracle": "Vision role (flavor role).",
    "vampire": "Dark role (flavor role).",
    "necromancer": "Revival role (flavor role).",
    "trickster": "Deception role (flavor role).",
    "judge": "Authority role (flavor role).",
    "arsonist": "Chaos role (flavor role).",
}


def role_label(role_id: str) -> str:
    return ROLE_LABEL.get(role_id, role_id.title())
