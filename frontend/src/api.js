const BASE_URL = process.env.REACT_APP_API_URL || 'http://5.42.105.7:8000';

// GET /matches → [{p1, p2, odds_a, odds_b}]
export async function fetchMatches() {
    const res = await fetch(`${BASE_URL}/matches`);
    if (!res.ok) throw new Error('Failed to fetch matches');
    return res.json();
}

// POST /predict → {p1, p2, prob_a, odds_a, odds_b}
export async function predictMatch(p1, p2) {
    const res = await fetch(`${BASE_URL}/predict`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({p1, p2}),
    });
    if (!res.ok) throw new Error('Prediction failed');
    return res.json();
}
