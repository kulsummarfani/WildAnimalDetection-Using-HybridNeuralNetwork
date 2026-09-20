"""Threat tiers and alert thresholds, keyed by species name.

The web app uses the detector's own class names (normalised by
``normalize_species``); the classifier-only labels (Jaguar, Cheetah) are kept
for Main.py, which still runs the 5-cat classifier.
"""

THREAT_TIERS = {
    # Big cats (also the only species that get the cat classifier / Grad-CAM check)
    'Tiger': 'HIGH',
    'Lion': 'HIGH',
    'Leopard': 'HIGH',
    'Jaguar': 'HIGH',
    'Cheetah': 'MEDIUM',
    # Other dangerous wildlife
    'Elephant': 'HIGH',
    'Bear': 'HIGH',
    'Wolf': 'MEDIUM',
    'Rhinoceros': 'MEDIUM',
    'Hippo': 'MEDIUM',
    'Buffalo': 'MEDIUM',
    'Monkey': 'MEDIUM',
    'Deer': 'LOW',
    'Giraffe': 'LOW',
    # Domestic animals: shown on the dashboard, never alert.
    'Cattle': 'LOW',
    'Bull': 'LOW',
    'Goat': 'LOW',
}

BIG_CATS = frozenset({'Tiger', 'Lion', 'Leopard'})

# The detector's class list spells buffalo "buffaloe".
_DISPLAY_NAMES = {'buffaloe': 'Buffalo'}


def normalize_species(raw_name: str) -> str:
    """Turn a detector class name into the display/threat-table name."""
    key = str(raw_name).strip().lower()
    return _DISPLAY_NAMES.get(key, key.replace('_', ' ').title())


def get_threat_tier(species: str) -> str:
    return THREAT_TIERS.get(species, 'MEDIUM')


def should_alert(species: str, confidence: float, high_threshold: float = 0.60,
                 medium_threshold: float = 0.85) -> bool:
    """Return whether confidence exceeds the threshold for the species' tier.

    LOW-tier species never alert. The default thresholds were chosen for the
    classifier's softmax confidence (Main.py); webapp.py passes its own
    thresholds for detector confidence.
    """
    tier = get_threat_tier(species)
    if tier == 'HIGH':
        return confidence >= high_threshold
    if tier == 'MEDIUM':
        return confidence >= medium_threshold
    return False
