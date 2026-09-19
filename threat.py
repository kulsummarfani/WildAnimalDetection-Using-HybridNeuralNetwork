"""Threat levels and alert thresholds for classified wildlife detections."""

THREAT_TIERS = {
    'Lion': 'HIGH',
    'Tiger': 'HIGH',
    'Leopard': 'HIGH',
    'Jaguar': 'HIGH',
    'Cheetah': 'MEDIUM',
}


def get_threat_tier(species: str) -> str:
    return THREAT_TIERS.get(species, 'MEDIUM')


def should_alert(species: str, confidence: float, high_threshold: float = 0.60,
                 medium_threshold: float = 0.85) -> bool:
    """Return whether the species confidence exceeds its configured risk threshold."""
    threshold = high_threshold if get_threat_tier(species) == 'HIGH' else medium_threshold
    return confidence >= threshold
