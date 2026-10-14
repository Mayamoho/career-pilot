"""Location eligibility for a candidate who is in Bangladesh.

These tests exist because the `is_remote` flag lies. Ashby reports `isRemote: true` for
roles whose only stated location is an office, so a filter built on the flag alone hands
a Dhaka candidate two hundred "remote" jobs that are in fact San Francisco jobs. Every
case below is a shape that actually appeared in the collected data.
"""

from __future__ import annotations

from careerpilot.domain.job import JobPosting
from careerpilot.domain.place import home_country, named_countries, place_of
from careerpilot.profile.model import Profile


def _job(
    location: str | None,
    *,
    remote: bool = False,
    source_id: str = "1",
) -> JobPosting:
    return JobPosting(
        source="greenhouse",
        source_id=source_id,
        company="Example",
        title="Backend Engineer",
        location=location,
        is_remote=remote,
    )


def test_a_remote_flag_on_an_office_location_is_not_remote() -> None:
    """The exact row that made this filter necessary: 124 of them, in fact."""
    place = place_of(_job("San Francisco", remote=True))

    assert place.mode == "remote-eligible"
    assert place.workable is False
    assert "San Francisco" in place.note


def test_an_unlisted_city_cannot_be_talked_into_workable() -> None:
    """The country table cannot hold every town, so the flag must not fill the gap.

    "Columbus, OH" matches no pattern at all. A rule that read "no country named,
    therefore anywhere" would quietly promote a hundred Ohio jobs to leads.
    """
    place = place_of(_job("Columbus, OH", remote=True))

    assert place.countries == ()
    assert place.workable is False


def test_explicit_country_scoped_remote_is_still_scoped() -> None:
    for location in ("Remote - US", "USA - Remote", "US - Remote", "Remote - Ireland"):
        place = place_of(_job(location, remote=True))
        assert place.workable is False, location
        assert place.mode == "remote"


def test_an_unscoped_remote_role_is_workable_from_dhaka() -> None:
    place = place_of(_job("Remote", remote=True))

    assert place.workable is True
    assert place.mode == "remote"


def test_a_dhaka_posting_is_home_regardless_of_work_mode() -> None:
    for location in ("Dhaka, Bangladesh", "Gulshan-1, Dhaka", "Uttara, Dhaka", "Chattogram"):
        place = place_of(_job(location))
        assert place.at_home is True, location
        assert place.workable is True, location
        assert place.note == "in Bangladesh"


def test_hybrid_with_no_office_named_cannot_be_assumed() -> None:
    """Forty rows say only "Hybrid". They are a real lead for someone local and a
    guess for everyone else, so the honest answer is not-workable-with-a-note."""
    place = place_of(_job("Hybrid"))

    assert place.mode == "hybrid"
    assert place.workable is False


def test_a_neighbourhood_abbreviation_does_not_rename_a_country() -> None:
    """'Gulshan' is Dhaka; 'Washington, DC' is not a word you can substring-match."""
    assert named_countries("Washington, DC") == ("United States",)
    assert named_countries("Washington Avenue, Gulshan, Dhaka") == ("Bangladesh",)


def test_short_country_codes_cannot_hide_inside_ordinary_words() -> None:
    assert named_countries("Status: Senior Backend Focus") == ()
    assert named_countries("Remote — US") == ("United States",)


def test_home_is_relative_to_the_profile_not_a_constant() -> None:
    """The same posting is a lead in Dhaka and a dead end in Lisbon."""
    job = _job("Remote - Portugal", remote=True)

    assert place_of(job, home="Bangladesh").workable is False
    assert place_of(job, home="Portugal").at_home is True
    assert place_of(job, home="Portugal").workable is True


def test_home_country_comes_from_the_fact_base() -> None:
    assert home_country(_profile("Mirpur, Dhaka, Bangladesh")) == "Bangladesh"
    assert home_country(_profile("Lisbon, Portugal")) == "Portugal"


def test_an_placeholder_address_still_defaults_to_bangladesh() -> None:
    """The committed example profile carries a placeholder address. Falling back to
    Bangladesh is who this build is for, and the note says which country was used."""
    assert home_country(_profile("Your City, Your Country")) == "Bangladesh"


def test_ranking_prefers_remote_at_home_over_remote_abroad() -> None:
    home_remote = place_of(_job("Dhaka, Bangladesh", remote=True))
    home_onsite = place_of(_job("Dhaka, Bangladesh"))
    open_remote = place_of(_job("Remote", remote=True))

    assert home_remote.rank < home_onsite.rank < open_remote.rank


def test_a_region_only_remote_is_workable_but_flagged() -> None:
    place = place_of(_job("Remote — EMEA", remote=True))

    assert place.workable is True
    assert place.region_only is True
    assert "region" in place.note


def test_a_region_that_excludes_home_is_a_lock_not_a_maybe() -> None:
    """Supabase lists this right beside its Global roles, which is how it got caught.

    "Remote, AMER" is a restriction with a boundary, not a remote role whose geography
    was left out. Reading it as workable sends a Dhaka candidate to apply for a job the
    posting had already closed to them.
    """
    place = place_of(_job("Remote, AMER", remote=True))

    assert place.mode == "remote"
    assert place.region_only is True
    assert place.workable is False
    assert "outside Bangladesh" in place.note


def test_a_region_that_could_contain_home_is_still_only_a_maybe() -> None:
    """South Asia sits inside most APAC splits, so the door is open but unlatched."""
    place = place_of(_job("Remote, APAC", remote=True))

    assert place.workable is True
    assert place.region_only is True
    assert "check the posting" in place.note


def test_a_global_remote_is_not_a_restricted_one() -> None:
    place = place_of(_job("Remote, Global", remote=True))

    assert place.workable is True
    assert place.region_only is False


def test_the_same_region_is_read_from_the_seekers_side() -> None:
    """The region words do not change; only where the reader is standing does.

    A Dhaka candidate is locked out of "Remote, AMER" and a Toronto candidate is locked
    out of "Remote, APAC". Anything that scored both the same way had quietly turned
    Bangladesh into a constant instead of a fact about the reader.
    """
    amer = place_of(_job("Remote, AMER", remote=True), home="Canada")
    apac = place_of(_job("Remote, APAC", remote=True), home="Canada")

    assert amer.workable is True
    assert apac.workable is False


def test_an_empty_location_is_not_silently_workable() -> None:
    place = place_of(_job(None))

    assert place.mode == "unspecified"
    assert place.workable is False


def test_a_place_that_is_simply_not_remote_says_the_place() -> None:
    """The note names what is missing, not what is present — but it must not deny the
    location exists. "London" is a stated place; it is not a reachable one."""
    place = place_of(_job("London"))

    assert place.workable is False
    assert place.note == "not stated as remote, in United Kingdom"


def _profile(location: str) -> Profile:
    return Profile(
        name="Test",
        headline="Backend Engineer",
        contact={"location": location},
        atoms=[],
    )
