# Preference-CFR: Beyond Nash Equilibrium for Diverse Game Strategies

## overview
Preference-CFR (Pref-CFR) is a novel algorithm that extends traditional Counterfactual Regret Minimization (CFR) to address the limitation of strategy diversity in game theory. While existing AI approaches focus on finding Nash Equilibrium (NE) for optimal payoffs, they often produce overly rational strategies lacking adaptability to different play styles. Pref-CFR introduces two key parameters:
- **Preference degree (δ)**: Guides strategies toward specific action inclinations (e.g., aggressive or passive).
- **Vulnerability degree (β)**: Allows controlled trade-offs between optimality and strategy flexibility, enabling convergence to ε-NE.

This repository is the accompanying code for the paper PrefCFR. This method allows strategies to converge to different equilibria. The paper has been accepted by ICML 2025 [Paper Link](https://openreview.net/forum?id=beBNOZP5Tk&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICML.cc%2F2025%2FConference%2FAuthors%23your-submissions)).
## Quick Start
**Clone the repo:**
```bash
git clone 
cd xxx
```
**Install dependencies:**
```bash
pip install -r requirements.txt
```

## Training Examples