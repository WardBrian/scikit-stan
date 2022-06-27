"""Tests for validation behavior."""
from email.generator import Generator
import sys
from typing import Tuple
import pytest

from sk_stan_regression.utils.validation import (
    validate_family, 
    FAMILY_LINKS_MAP
)
# -> Generator[Tuple[str, str], None, None]
def incompatible_fam_link_gen() : 
    """Generator for family and link combinations that are incompatible."""
    for fam in ["gaussian", "binomial", "gamma", "poisson", "inverse_gaussian"]:
        for link in ["identity", "inverse", "log", "1/mu^2"]:
            if link not in FAMILY_LINKS_MAP[fam].keys():
                yield fam, link

def compatible_fam_link_gen(): 
    """Generator for family and link combinations that are compatible."""
    for fam in ["gaussian", "binomial", "gamma", "poisson", "inverse_gaussian"]:
        for link in ["identity", "inverse", "log", "1/mu^2"]:
            if link in FAMILY_LINKS_MAP[fam].keys():
                yield fam, link

@pytest.mark.parametrize("fam, link", compatible_fam_link_gen())
def test_valid_fam_valid_links(fam: str, link: str) -> None:
    """Test that valid family and link combinations are accepted."""
    validate_family(fam, link)


@pytest.mark.parametrize("fam, link", incompatible_fam_link_gen())
def test_valid_fam_invalid_links(fam:str, link:str) -> None:
    """Test that validate_family raises an error when the family is not supported."""
    with pytest.raises(ValueError):
        validate_family(fam, link)

if __name__ == "__main__":
    sys.exit(pytest.main(["-qq"]))
