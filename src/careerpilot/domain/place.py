"""Where a posting actually is, and whether it can be worked from where you are.

The ``is_remote`` flag is not enough on its own. Ashby reports ``isRemote: true`` for
roles whose only stated location is an office, so filtering on the flag alone hands a
candidate in Dhaka two hundred "remote" jobs that are in fact San Francisco jobs.
Location text is treated as the fact and the flag as a claim about it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from careerpilot.domain.job import JobPosting
from careerpilot.profile.model import Profile

Mode = Literal["remote", "hybrid", "onsite", "remote-eligible", "unspecified"]

# The assumption made when the fact base names no country. This build is for a seeker
# in Dhaka; `contact.location` in the profile overrides it for anyone else.
DEFAULT_HOME = "Bangladesh"

# Bangladeshi districts and the Dhaka neighbourhoods that appear as a whole location.
_BD = (
    "bangladesh", "dhaka", "chattogram", "chittagong", "sylhet", "khulna", "rangpur",
    "barishal", "barisal", "mymensingh", "gazipur", "narayanganj", "cumilla", "kushtia",
    "uttara", "gulshan", "mirpur", "dhanmondi", "banani", "pallabi", "moghbazar",
    "farmgate", "bosilia", "savar", "tonipur", "chawkbazar", "kotwali",
)

# Canonical country -> the words a posting uses to name it. Cities are listed under
# the country they are in, because "Dublin, Ireland" and "Dublin" must agree.
_COUNTRIES: dict[str, tuple[str, ...]] = {
    "Bangladesh": _BD,
    "United States": (
        "united states", "usa", "u.s.a", "u.s.", "us", "washington, dc", "washington dc",
        "d.c.", "san francisco", "sf bay area", "bay area", "new york", "seattle",
        "california", "mountain view", "boston", "austin", "chicago", "denver", "atlanta",
        "portland", "minneapolis", "remote - us", "remote us", "san diego", "durham",
        "north carolina", "remote - canada and us",
    ),
    "Canada": ("canada", "toronto", "vancouver", "montreal", "ottawa", "calgary", "waterloo"),
    "United Kingdom": ("united kingdom", "uk", "london", "manchester", "edinburgh", "england"),
    "Ireland": ("ireland", "dublin", "cork"),
    "Germany": ("germany", "berlin", "munich", "frankfurt", "hamburg", "dach"),
    "France": ("france", "paris", "lyon"),
    "Netherlands": ("netherlands", "amsterdam", "the hague"),
    "Spain": ("spain", "madrid", "barcelona"),
    "Portugal": ("portugal", "lisbon"),
    "Greece": ("greece", "athens"),
    "Poland": ("poland", "warsaw", "krakow", "wroclaw", "gdansk"),
    "Romania": ("romania", "bucharest", "cluj"),
    "Norway": ("norway", "oslo"),
    "Sweden": ("sweden", "stockholm", "gothenburg"),
    "Denmark": ("denmark", "copenhagen"),
    "Finland": ("finland", "helsinki"),
    "Italy": ("italy", "milan", "rome", "turin"),
    "Switzerland": ("switzerland", "zurich", "geneva", "lausanne"),
    "Austria": ("austria", "vienna"),
    "Czechia": ("czech", "prague", "brno"),
    "Ukraine": ("ukraine", "kyiv", "lviv"),
    "Israel": ("israel", "tel aviv", "haifa"),
    "United Arab Emirates": ("united arab emirates", "uae", "abu dhabi", "dubai"),
    "Turkey": ("turkey", "istanbul", "türkiye"),
    "India": (
        "india", "gurugram", "gurgaon", "bengaluru", "bangalore", "hyderabad", "mumbai",
        "pune", "delhi", "noida", "chennai", "kolkata", "ahmedabad",
    ),
    "Pakistan": ("pakistan", "karachi", "lahore", "islamabad"),
    "Sri Lanka": ("sri lanka", "colombo"),
    "Singapore": ("singapore",),
    "Japan": ("japan", "tokyo", "osaka"),
    "South Korea": ("south korea", "seoul"),
    "China": ("china", "beijing", "shanghai", "shenzhen"),
    "Australia": ("australia", "sydney", "melbourne", "brisbane"),
    "New Zealand": ("new zealand", "auckland"),
    "Brazil": ("brazil", "são paulo", "sao paulo", "rio de janeiro"),
    "Mexico": ("mexico", "mexico city", "queretaro", "guadalajara"),
    "Argentina": ("argentina", "buenos aires"),
    "Colombia": ("colombia", "bogota", "bogotá", "medellin"),
    "Chile": ("chile", "santiago"),
    "Costa Rica": ("costa rica", "san josé", "san jose"),
    "Kenya": ("kenya", "nairobi"),
    "Nigeria": ("nigeria", "lagos", "abuja"),
    "South Africa": ("south africa", "cape town", "johannesburg"),
    "Egypt": ("egypt", "cairo"),
    "Philippines": ("philippines", "manila", "cebu"),
    "Indonesia": ("indonesia", "jakarta"),
    "Malaysia": ("malaysia", "kuala lumpur"),
    "Vietnam": ("vietnam", "hanoi", "ho chi minh"),
    "Thailand": ("thailand", "bangkok"),
    "Estonia": ("estonia", "tallinn"),
}

# Regional buckets name several countries without picking one, so they can only be read
# from the seeker's side: "Remote, AMER" is open to a candidate in Toronto and closed to
# one in Dhaka, and the phrase alone will not tell you which.
#
# Supabase's board is what forced this. It lists "Remote, Global" next to
# "Remote, AMER", and the first version of this module treated any named region as
# "worth checking" — so it called both workable from Bangladesh and handed the seeker a
# role the posting had explicitly scoped to the Americas.
_EUROPE = frozenset(
    {
        "Ireland", "Germany", "France", "Netherlands", "Spain", "Portugal", "Greece",
        "Poland", "Romania", "Norway", "Sweden", "Denmark", "Finland", "Italy",
        "Switzerland", "Austria", "Czechia", "Ukraine", "Estonia",
    }
)
_AFRICA = frozenset({"Egypt", "Kenya", "Nigeria", "South Africa"})
_WEST_ASIA = frozenset({"Israel", "United Arab Emirates", "Turkey"})
_SOUTH_ASIA = frozenset({"Bangladesh", "India", "Pakistan", "Sri Lanka"})
_EAST_ASIA = frozenset(
    {
        "Singapore", "Japan", "South Korea", "China", "Vietnam", "Thailand",
        "Philippines", "Indonesia", "Malaysia", "Australia", "New Zealand",
    }
)
_AMERICAS = frozenset(
    {
        "United States", "Canada", "Mexico", "Brazil", "Argentina", "Colombia",
        "Chile", "Costa Rica",
    }
)

_REGIONS: tuple[tuple[tuple[str, ...], frozenset[str]], ...] = (
    (("latam", "latin america", "amer", "americas", "north america", "south america"), _AMERICAS),
    (("emea", "emenat", "mena"), _EUROPE | _AFRICA | _WEST_ASIA | _SOUTH_ASIA),
    (("europe", "eu", "nordic"), _EUROPE),
    (("apac", "asia pacific", "southeast asia", "south asia", "asia"), _SOUTH_ASIA | _EAST_ASIA),
)

_WORDY = re.compile(r"^[a-z0-9. ]+$")
# Two- and three-letter country abbreviations need word boundaries; "us" otherwise
# matches inside "status", "focus" and every other suffix. Longer names are unambiguous
# enough to match as substrings.
_SHORT_TOKEN = 3


def _mentions(text: str, needle: str) -> bool:
    """Word-boundary match for short tokens, substring for phrases.

    ``us`` has to be bounded or it matches inside ``status``; ``san francisco`` does not.
    """
    token = needle.lower().strip()
    if _WORDY.match(token) and len(token) <= _SHORT_TOKEN:
        return re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", text) is not None
    return token in text


def named_countries(text: str) -> tuple[str, ...]:
    """Every country a piece of location text names, in the order of the table."""
    lowered = f" {text.lower().strip()} "
    found = [
        country
        for country, words in _COUNTRIES.items()
        if any(_mentions(lowered, word) for word in words)
    ]
    return tuple(found)


def named_regions(text: str) -> tuple[frozenset[str], ...]:
    """The country sets the regional words in `text` stand for."""
    lowered = f" {text.lower().strip()} "
    return tuple(
        members
        for words, members in _REGIONS
        if any(_mentions(lowered, word) for word in words)
    )


def home_country(profile: Profile) -> str:
    """The country the fact base says you are standing in.

    Location verdicts are relative, so this comes from the profile rather than a
    constant baked into the caller: the same posting is workable for a Dhaka candidate
    and blocked for one in Lisbon. Falls back to Bangladesh, because that is who this
    build is for, and the verdict note says when it guessed.
    """
    named = named_countries(profile.contact.get("location", ""))
    return named[0] if named else DEFAULT_HOME


def _mode_of(text: str, is_remote: bool) -> Mode:
    if _mentions(text, "remote") or _mentions(text, "work from home") or _mentions(text, "wfh"):
        return "remote"
    if _mentions(text, "hybrid"):
        return "hybrid"
    if _mentions(text, "in-office") or _mentions(text, "onsite") or _mentions(text, "on-site"):
        return "onsite"
    if is_remote:
        return "remote-eligible"
    return "unspecified"


@dataclass(frozen=True)
class Place:
    mode: Mode
    countries: tuple[str, ...]
    at_home: bool
    workable: bool
    note: str
    region_only: bool
    home: str

    @property
    def rank(self) -> int:
        """Sort order: remote at home, then at home at all, then open remote."""
        if self.at_home and self.mode in {"remote", "remote-eligible"}:
            return 0
        if self.at_home:
            return 1
        if self.workable:
            return 2
        return 3


def place_of(job: JobPosting, *, home: str = DEFAULT_HOME) -> Place:
    """Classify one posting against the place you are actually standing in."""
    text = f" {((job.location or '') + ' ' + (job.employment_type or '')).lower().strip()} "
    mode = _mode_of(text, job.is_remote)
    countries = named_countries(text)
    at_home = home in countries
    away = tuple(c for c in countries if c != home)
    regions = () if countries else named_regions(text)
    region_only = bool(regions)

    remote_claimed = mode in {"remote", "remote-eligible"} or job.is_remote
    if at_home:
        note = f"in {home}"
        workable = True
    elif mode == "remote" and away:
        note = f"remote, but lists {', '.join(away)}"
        workable = False
    elif mode == "remote-eligible":
        # The flag alone cannot grant permission. It is set for office jobs, and an
        # unlisted city ("Columbus, OH", "Pune") is not evidence of anywhere reachable,
        # so an unverified claim loses to the location text. Under-promising here is the
        # point: `--where all` still shows the posting, with this note on it.
        workable = False
        note = f"flagged remote, but lists {job.location or 'no place'}"
    elif mode == "hybrid":
        note = f"hybrid{', in ' + ', '.join(away) if away else ' with no office named'}"
        workable = False
    elif mode == "onsite":
        note = f"on-site{', in ' + ', '.join(away) if away else ''}"
        workable = False
    elif remote_claimed:
        # A named region is a restriction, not an invitation. It only grants
        # reachability if the country you are standing in falls inside it.
        if region_only:
            covers_home = any(home in members for members in regions)
            workable = covers_home
            note = (
                f"remote, restricted to a region that covers {home} — check the posting"
                if covers_home
                else f"remote, restricted to a region outside {home}"
            )
        else:
            workable = True
            note = "remote, no country named — check the posting"
    else:
        # Nothing in the row claims remote. Say what is missing rather than what is
        # present: "London" is a named location, it just is not a reachable one.
        note = f"not stated as remote{', in ' + ', '.join(away) if away else ''}"
        workable = False

    return Place(
        mode=mode,
        countries=countries,
        at_home=at_home,
        workable=workable,
        note=note,
        region_only=region_only,
        home=home,
    )
