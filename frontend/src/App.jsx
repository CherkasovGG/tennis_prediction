import {useEffect, useRef, useState} from "react";
import * as C from "./colors";
import {fetchMatches, predictMatch} from "./api";

const FONT_LINK =
    "https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Outfit:wght@300;400;500;600;700;800&display=swap";

function ProbBar({probA, nameA, nameB}) {
    const pctA = (probA * 100).toFixed(1);
    const pctB = ((1 - probA) * 100).toFixed(1);
    return (
        <div style={{marginTop: 16}}>
            <div style={{
                display: "flex",
                justifyContent: "space-between",
                marginBottom: 6,
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 13
            }}>
                <span style={{color: C.PRIMARY}}>{nameA} {pctA}%</span>
                <span style={{color: C.SECONDARY}}>{nameB} {pctB}%</span>
            </div>
            <div style={{height: 10, borderRadius: 5, background: C.BG_BAR_TRACK, overflow: "hidden", display: "flex"}}>
                <div
                    style={{
                        width: `${pctA}%`,
                        background: C.BAR_GRAD_A,
                        borderRadius: "5px 0 0 5px",
                        transition: "width 0.8s cubic-bezier(.4,0,.2,1)",
                    }}
                />
                <div
                    style={{
                        width: `${pctB}%`,
                        background: C.BAR_GRAD_B,
                        borderRadius: "0 5px 5px 0",
                        transition: "width 0.8s cubic-bezier(.4,0,.2,1)",
                    }}
                />
            </div>
        </div>
    );
}

function MatchCard({p1, p2, oddsA, oddsB, onClick, idx}) {
    const [hovered, setHovered] = useState(false);
    return (
        <button
            onClick={onClick}
            onMouseEnter={() => setHovered(true)}
            onMouseLeave={() => setHovered(false)}
            style={{
                width: "100%",
                background: hovered ? C.CARD_BG_HOVER : C.CARD_BG,
                border: `1px solid ${hovered ? C.CARD_BORDER_HOVER : C.CARD_BORDER}`,
                borderRadius: 12,
                padding: "16px 20px",
                cursor: "pointer",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                transition: "all 0.25s ease",
                fontFamily: "'Outfit', sans-serif",
                transform: hovered ? "translateX(4px)" : "none",
                animation: `fadeSlideIn 0.4s ease ${idx * 0.06}s both`,
            }}
        >
            <div style={{display: "flex", alignItems: "center", gap: 12}}>
                <div style={{width: 6, height: 6, borderRadius: "50%", background: C.PRIMARY, flexShrink: 0}}/>
                <span style={{color: C.TEXT, fontSize: 15, fontWeight: 500}}>
          {p1} <span style={{color: C.TEXT_VS, margin: "0 4px"}}>vs</span> {p2}
        </span>
            </div>
            <div style={{fontFamily: "'JetBrains Mono', monospace", fontSize: 13, color: C.TEXT_MUTED}}>
                {oddsA.toFixed(2)} / {oddsB.toFixed(2)}
            </div>
        </button>
    );
}

function ResultPanel({p1, p2, probA, oddsA, oddsB, onClose}) {
    const ev1 = (probA * oddsA - 1) * 100;
    const ev2 = ((1 - probA) * oddsB - 1) * 100;
    return (
        <div
            style={{
                background: C.RESULT_BG,
                border: `1px solid ${C.RESULT_BORDER}`,
                borderRadius: 16,
                padding: 28,
                animation: "fadeSlideIn 0.4s ease both",
                position: "relative",
            }}
        >
            <button
                onClick={onClose}
                style={{
                    position: "absolute",
                    top: 14,
                    right: 16,
                    background: "none",
                    border: "none",
                    color: C.TEXT_SECTION,
                    fontSize: 20,
                    cursor: "pointer",
                    fontFamily: "'Outfit', sans-serif",
                    lineHeight: 1,
                }}
            >
                ×
            </button>

            <div style={{textAlign: "center", marginBottom: 20}}>
                <div style={{
                    fontSize: 12,
                    textTransform: "uppercase",
                    letterSpacing: 3,
                    color: C.RESULT_LABEL,
                    marginBottom: 8,
                    fontWeight: 600
                }}>
                    Прогноз модели
                </div>
                <div style={{fontFamily: "'Outfit', sans-serif", fontSize: 22, fontWeight: 700, color: C.TEXT_WHITE}}>
                    {p1} <span style={{color: C.TEXT_PLACEHOLDER, fontSize: 16}}>vs</span> {p2}
                </div>
            </div>

            <ProbBar probA={probA} nameA={p1} nameB={p2}/>

            <div style={{display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginTop: 20}}>
                {[
                    {label: p1, prob: probA, odds: oddsA, ev: ev1, color: C.PRIMARY},
                    {label: p2, prob: 1 - probA, odds: oddsB, ev: ev2, color: C.SECONDARY},
                ].map((s) => (
                    <div
                        key={s.label}
                        style={{
                            background: C.STAT_CARD_BG,
                            borderRadius: 10,
                            padding: "14px 16px",
                            border: `1px solid ${C.STAT_CARD_BORDER}`,
                        }}
                    >
                        <div style={{
                            fontSize: 11,
                            color: C.TEXT_SUBTITLE,
                            textTransform: "uppercase",
                            letterSpacing: 1,
                            marginBottom: 10,
                            fontWeight: 600
                        }}>
                            {s.label}
                        </div>
                        <div style={{fontFamily: "'JetBrains Mono', monospace"}}>
                            <div style={{display: "flex", justifyContent: "space-between", marginBottom: 6}}>
                                <span style={{fontSize: 12, color: C.TEXT_MUTED}}>P(win)</span>
                                <span style={{
                                    fontSize: 14,
                                    color: s.color,
                                    fontWeight: 600
                                }}>{(s.prob * 100).toFixed(1)}%</span>
                            </div>
                            <div style={{display: "flex", justifyContent: "space-between", marginBottom: 6}}>
                                <span style={{fontSize: 12, color: C.TEXT_MUTED}}>Коэфф</span>
                                <span style={{fontSize: 14, color: C.TEXT}}>{s.odds.toFixed(2)}</span>
                            </div>
                            <div style={{display: "flex", justifyContent: "space-between"}}>
                                <span style={{fontSize: 12, color: C.TEXT_MUTED}}>EV</span>
                                <span style={{fontSize: 14, color: s.ev > 0 ? C.PRIMARY : C.SECONDARY, fontWeight: 600}}>
                  {s.ev > 0 ? "+" : ""}{s.ev.toFixed(1)}%
                </span>
                            </div>
                        </div>
                    </div>
                ))}
            </div>

            <div style={{
                marginTop: 16,
                fontSize: 11,
                color: C.TEXT_PLACEHOLDER,
                textAlign: "center",
                fontFamily: "'JetBrains Mono', monospace"
            }}>
                EV = P(win) × коэфф − 1 · положительный EV = ставка с преимуществом
            </div>
        </div>
    );
}

export default function App() {
    const [matches, setMatches] = useState([]);
    const [input1, setInput1] = useState("");
    const [input2, setInput2] = useState("");
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const resultRef = useRef(null);

    useEffect(() => {
        const link = document.createElement("link");
        link.rel = "stylesheet";
        link.href = FONT_LINK;
        document.head.appendChild(link);

        const style = document.createElement("style");
        style.textContent = `
      @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
      }
      @keyframes pulse { 0%,100%{opacity:.4} 50%{opacity:1} }
      input::placeholder { color: ${C.TEXT_PLACEHOLDER} !important; }
      select option { background: ${C.BG_ELEVATED}; color: ${C.TEXT}; padding: 8px; }
      * { box-sizing: border-box; }
      ::-webkit-scrollbar { width: 4px; }
      ::-webkit-scrollbar-thumb { background: ${C.SCROLLBAR_THUMB}; border-radius: 2px; }
    `;
        document.head.appendChild(style);
    }, []);

    useEffect(() => {
        fetchMatches()
            .then(setMatches)
            .catch(() => setError("Не удалось загрузить матчи"));
    }, []);

    useEffect(() => {
        if (result && resultRef.current) {
            resultRef.current.scrollIntoView({behavior: "smooth", block: "nearest"});
        }
    }, [result]);

    const allPlayers = [...new Set(matches.flatMap((m) => [m.p1, m.p2]))].sort();

    const predict = (p1, p2) => {
        setError("");
        setLoading(true);
        predictMatch(p1, p2)
            .then(({prob_a, odds_a, odds_b}) => {
                setResult({p1, p2, probA: prob_a, oddsA: odds_a, oddsB: odds_b});
            })
            .catch(() => setError("Ошибка при расчёте прогноза"))
            .finally(() => setLoading(false));
    };

    const handleSubmit = () => {
        if (!input1 || !input2) {
            setError("Выберите обоих игроков");
            return;
        }
        predict(input1, input2);
    };

    return (
        <div
            style={{
                minHeight: "100vh",
                background: C.BG_PAGE,
                color: C.TEXT,
                fontFamily: "'Outfit', sans-serif",
                position: "relative",
                overflow: "hidden",
            }}
        >
            {/* BG grain */}
            <div
                style={{
                    position: "fixed",
                    inset: 0,
                    opacity: 0.03,
                    backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E")`,
                    backgroundSize: 200,
                    pointerEvents: "none",
                    zIndex: 0,
                }}
            />
            {/* Glow */}
            <div
                style={{
                    position: "fixed",
                    top: -200,
                    left: "50%",
                    transform: "translateX(-50%)",
                    width: 600,
                    height: 600,
                    borderRadius: "50%",
                    background: C.BG_GLOW,
                    pointerEvents: "none",
                    zIndex: 0,
                }}
            />

            <div style={{position: "relative", zIndex: 1, maxWidth: 520, margin: "0 auto", padding: "40px 20px 60px"}}>
                {/* Header */}
                <header style={{marginBottom: 40, animation: "fadeSlideIn 0.5s ease both"}}>
                    <div style={{display: "flex", alignItems: "center", gap: 10, marginBottom: 6}}>
                        <div style={{fontSize: 22}}>🎾</div>
                        <h1
                            style={{
                                fontSize: 26,
                                fontWeight: 800,
                                margin: 0,
                                letterSpacing: -0.5,
                                background: C.TITLE_GRADIENT,
                                WebkitBackgroundClip: "text",
                                WebkitTextFillColor: "transparent",
                            }}
                        >
                            Tennis Predict
                        </h1>
                    </div>
                    <p style={{color: C.TEXT_SUBTITLE, fontSize: 14, margin: 0, lineHeight: 1.5}}>
                        Прогноз вероятности победы на основе ML-модели
                    </p>
                </header>

                {/* Input Section */}
                <section style={{marginBottom: 36, animation: "fadeSlideIn 0.5s ease 0.1s both"}}>
                    <div style={{
                        fontSize: 11,
                        textTransform: "uppercase",
                        letterSpacing: 2,
                        color: C.TEXT_SECTION,
                        marginBottom: 12,
                        fontWeight: 600
                    }}>
                        Выбор игроков
                    </div>
                    <div style={{display: "flex", gap: 10, marginBottom: 10}}>
                        {[
                            {value: input1, setter: setInput1, placeholder: "Игрок 1"},
                            {value: input2, setter: setInput2, placeholder: "Игрок 2"},
                        ].map(({value, setter, placeholder}) => (
                            <div key={placeholder} style={{flex: 1, position: "relative"}}>
                                <select
                                    value={value}
                                    onChange={(e) => setter(e.target.value)}
                                    style={{
                                        width: "100%",
                                        padding: "12px 36px 12px 16px",
                                        borderRadius: 10,
                                        border: `1px solid ${C.INPUT_BORDER}`,
                                        background: C.INPUT_BG,
                                        color: value ? C.TEXT_WHITE : C.TEXT_PLACEHOLDER,
                                        fontSize: 15,
                                        fontFamily: "'Outfit', sans-serif",
                                        outline: "none",
                                        cursor: "pointer",
                                        appearance: "none",
                                        WebkitAppearance: "none",
                                        transition: "border-color 0.2s",
                                    }}
                                    onFocus={(e) => (e.target.style.borderColor = C.INPUT_BORDER_FOCUS)}
                                    onBlur={(e) => (e.target.style.borderColor = C.INPUT_BORDER)}
                                >
                                    <option value="" disabled
                                            style={{background: C.BG_ELEVATED, color: C.TEXT_SECTION}}>
                                        {placeholder}
                                    </option>
                                    {allPlayers.filter((p) =>
                                        placeholder === "Игрок 2" ? p !== input1 : p !== input2
                                    ).map((p) => (
                                        <option key={p} value={p} style={{background: C.BG_ELEVATED, color: C.TEXT}}>
                                            {p}
                                        </option>
                                    ))}
                                </select>
                                <div
                                    style={{
                                        position: "absolute",
                                        right: 14,
                                        top: "50%",
                                        transform: "translateY(-50%)",
                                        pointerEvents: "none",
                                        color: C.TEXT_VS,
                                        fontSize: 10,
                                    }}
                                >
                                    ▼
                                </div>
                            </div>
                        ))}
                    </div>
                    <button
                        onClick={handleSubmit}
                        disabled={loading}
                        style={{
                            width: "100%",
                            padding: "12px 0",
                            borderRadius: 10,
                            border: "none",
                            background: loading ? C.BTN_BG_LOADING : C.BTN_GRADIENT,
                            color: C.BG_PAGE,
                            fontSize: 14,
                            fontWeight: 700,
                            fontFamily: "'Outfit', sans-serif",
                            cursor: loading ? "wait" : "pointer",
                            transition: "all 0.2s",
                            letterSpacing: 0.5,
                        }}
                    >
                        {loading ? "Расчёт…" : "Рассчитать прогноз"}
                    </button>
                    {error && (
                        <div style={{color: C.SECONDARY, fontSize: 13, marginTop: 8}}>{error}</div>
                    )}
                </section>

                {/* Result */}
                {result && (
                    <div ref={resultRef} style={{marginBottom: 36}}>
                        <ResultPanel
                            p1={result.p1}
                            p2={result.p2}
                            probA={result.probA}
                            oddsA={result.oddsA}
                            oddsB={result.oddsB}
                            onClose={() => setResult(null)}
                        />
                    </div>
                )}

                {/* Matches list */}
                <section style={{animation: "fadeSlideIn 0.5s ease 0.2s both"}}>
                    <div style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        marginBottom: 14
                    }}>
                        <div style={{
                            fontSize: 11,
                            textTransform: "uppercase",
                            letterSpacing: 2,
                            color: C.TEXT_SECTION,
                            fontWeight: 600
                        }}>
                            Ближайшие матчи
                        </div>
                        <div style={{
                            fontSize: 11,
                            color: C.MATCH_COUNT,
                            fontFamily: "'JetBrains Mono', monospace"
                        }}>
                            {matches.length} матч{matches.length > 4 ? "ей" : matches.length > 1 ? "а" : ""}
                        </div>
                    </div>
                    <div style={{display: "flex", flexDirection: "column", gap: 8}}>
                        {matches.map(({p1, p2, odds_a, odds_b}, idx) => (
                            <MatchCard
                                key={`${p1}|${p2}`}
                                p1={p1}
                                p2={p2}
                                oddsA={odds_a}
                                oddsB={odds_b}
                                idx={idx}
                                onClick={() => predict(p1, p2)}
                            />
                        ))}
                    </div>
                </section>

                {/* Footer */}
                <footer style={{
                    marginTop: 48,
                    textAlign: "center",
                    fontSize: 11,
                    color: C.TEXT_FOOTER,
                    fontFamily: "'JetBrains Mono', monospace",
                    lineHeight: 1.6
                }}>
                    <div>Tennis Predict · ML-модель прогнозирования</div>
                    <div style={{marginTop: 4}}>Курсовая работа · 2026</div>
                </footer>
            </div>
        </div>
    );
}
