import numpy as np


def calculate_risk_score(probability):
    """
    Convert return probability into a 0-100 risk score.
    """

    probability = float(np.clip(probability, 0, 1))

    return round(probability * 100, 2)


def classify_risk(score):
    """
    Convert numerical risk score into a risk category.
    """

    if score < 35:
        return "LOW"

    elif score < 65:
        return "MEDIUM"

    else:
        return "HIGH"


def get_decision(risk_level):
    """
    Convert risk category into a business decision.
    """

    if risk_level == "LOW":
        return "APPROVE"

    elif risk_level == "MEDIUM":
        return "VERIFY"

    else:
        return "MANUAL REVIEW"


def generate_risk_result(probability):
    """
    Complete ReturnGuard risk decision.
    """

    score = calculate_risk_score(probability)

    risk_level = classify_risk(score)

    decision = get_decision(risk_level)

    return {
        "return_probability": round(float(probability), 4),
        "risk_score": score,
        "risk_level": risk_level,
        "decision": decision
    }


if __name__ == "__main__":

    print("=" * 60)
    print("RETURNGUARD RISK ENGINE")
    print("=" * 60)

    test_probabilities = [
        0.20,
        0.45,
        0.72,
        0.90
    ]

    for probability in test_probabilities:

        result = generate_risk_result(probability)

        print("\nReturn Probability:",
              result["return_probability"])

        print("Risk Score:",
              result["risk_score"])

        print("Risk Level:",
              result["risk_level"])

        print("Decision:",
              result["decision"])