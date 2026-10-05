# Next short term strategies

Oui. Vu tes résultats, je changerais assez nettement de famille de stratégies.

Ton problème actuel est que tu testes surtout des stratégies **intraday de continuation** :

* ORB → continuation après l'ouverture
* News sentiment → réaction à une information
* VWAP Pullback → continuation après retracement

Et elles ont un point commun : **tu achètes un mouvement déjà en cours**. Cela peut donner exactement le profil que tu observes : quelques gros gagnants, mais beaucoup de petits/moyens faux signaux qui mangent le P&L.

Pour ton architecture — **long only, stock picking, pas de margin/short, horizon minutes → quelques jours, overnight autorisé** — je testerais plutôt les familles suivantes.

### Mon classement

| Priorité | Stratégie                                   | Holding typique | Pourquoi je la testerais                        |
| -------- | ------------------------------------------- | --------------: | ----------------------------------------------- |
| 🥇       | **Post-Earnings Drift / Earnings Momentum** |      1–10 jours | Très bon fit avec long-only                     |
| 🥈       | **Relative Strength + Pullback**            |       1–5 jours | Moins dépendant du timing exact intraday        |
| 🥉       | **52-Week High Breakout / High Tight Flag** |      1–10 jours | Momentum structurel, pas simplement ORB         |
| 4        | **Intraday Exhaustion → Mean Reversion**    |  minutes–1 jour | Complémentaire à tes stratégies de continuation |
| 5        | **Overnight Gap Continuation**              |       1–3 jours | Exploite explicitement le gap                   |
| 6        | **Volatility Contraction → Expansion**      |      1–10 jours | Très intéressant pour du stock picking          |
| 7        | **Relative Strength vs Sector**             |       1–5 jours | Peut réduire la corrélation au SP500            |

---

# 1. 🥇 Post-Earnings Announcement Drift

C'est probablement **la première stratégie que je testerais dans ton cas**.

L'idée n'est surtout pas :

> "L'entreprise publie de bons résultats → j'achète."

Mais plutôt :

```text
Earnings announcement
        ↓
Large positive surprise
        ↓
Gap / strong reaction
        ↓
Stock holds most of the gain
        ↓
Volume remains elevated
        ↓
No immediate reversal
        ↓
BUY
        ↓
Hold 1–10 days
```

Le phénomène de **PEAD** est documenté depuis longtemps : après une surprise bénéficiaire, le prix peut continuer à évoluer dans la direction de la surprise plutôt que de tout intégrer immédiatement. Des travaux récents continuent à étudier cette anomalie, même si la robustesse dépend fortement de la définition du signal et de l'univers. ([ScienceDirect][1])

Et surtout, cela correspond très bien à ton besoin :

**tu n'as pas besoin de prédire la prochaine minute.**

Tu peux laisser le marché faire le travail pendant plusieurs séances.

### Exemple

Supposons :

```text
EPS expected:   $1.20
EPS actual:     $1.52       +26.7%
Revenue:        beat
Guidance:       raised

Previous close: $80
Open:           $89         +11.3%
```

Tu ne rentres **pas nécessairement à $89**.

Ton scanner attend :

```text
09:30   $89
10:00   $90
11:00   $91
12:30   $90.50
14:00   $92
15:30   $93
```

Le signal devient :

```text
positive earnings surprise
+
large abnormal return
+
high relative volume
+
price holds above VWAP
+
no immediate gap fade
+
relative strength vs sector
```

→ **BUY**

Puis tu peux conserver plusieurs jours.

C'est très différent de ton VWAP Pullback.

---

# 2. Relative Strength + Pullback

C'est probablement **la stratégie que je testerais juste après PEAD**.

Pas :

> "Le marché monte → j'achète."

Mais :

> "Quelle action est exceptionnellement forte relativement à son secteur et au marché, puis offre un pullback relativement faible ?"

Exemple :

```text
SPY       +0.4%
XLK       +0.8%
NVDA      +3.2%
```

Puis :

```text
NVDA +3.2%
↓
pullback -0.7%
↓
volume decreases
↓
holds previous breakout
↓
relative strength remains high
↓
BUY
```

Le signal pourrait être quelque chose comme :

```python
relative_strength_1d
relative_strength_5d
relative_strength_20d

volume_ratio
distance_from_20ema
distance_from_vwap

pullback_depth
pullback_volume / impulse_volume
```

Puis un score :

```text
RS vs SPY          25%
RS vs sector       25%
5d momentum        15%
20d momentum       10%
volume expansion   10%
pullback quality   15%
```

Ce que j'aime ici est que **le pullback devient une condition d'entrée et non la stratégie elle-même**.

---

# 3. 52-Week High Momentum

Celui-ci mérite vraiment un backtest.

Il existe une littérature assez importante autour du **52-week high**. George & Hwang montrent notamment que la proximité du plus haut 52 semaines apporte une information importante au-delà des simples rendements passés pour prédire les rendements futurs. ([Wiley Online Library][2])

L'idée :

```text
52-week high
      │
      │
      │       ┌───┐
      │      /    │
      │     /     │
      │────/──────┘
      │
```

Mais je ne ferais **pas** simplement :

```python
close >= high_252 * 0.98
```

Je rechercherais :

### Setup

```text
Price within 2–3% of 52w high
+
20/50 EMA bullish
+
relative strength positive
+
volume expansion
+
consolidation
+
breakout of consolidation
```

Puis :

```text
BUY
```

avec holding :

```text
1 → 10 days
```

Cela est beaucoup plus intéressant pour ton système que l'ORB, car tu exploites un **breakout structurel** plutôt qu'un breakout arbitraire des 15/30 premières minutes.

---

# 4. Intraday Exhaustion → Mean Reversion

Celui-là est presque l'inverse philosophique de ce que tu as testé jusqu'ici.

Tu cherches :

```text
STRONG MOVE
     ↓
EXTREME EXTENSION
     ↓
BUYING EXHAUSTION
     ↓
REVERSION
```

Par exemple :

```text
09:30   100
10:00   104
10:30   107
11:00   110
11:15   111
11:30   109
11:45   108
```

Au lieu d'acheter la continuation à 110–111, tu cherches le moment où :

```text
momentum ↓
volume ↓
distance VWAP extreme
RSI extreme
range expansion
failed continuation
```

et tu achètes le retracement.

Il existe d'ailleurs des travaux montrant que les rendements intraday peuvent présenter des phénomènes de reversal, particulièrement lorsque la liquidité est faible et la volatilité élevée. ([DOI][3])

**Très important dans ton cas :**

Je ne ferais pas un simple :

```python
RSI < 30 → BUY
```

Ce serait probablement un piège.

Je chercherais plutôt :

```text
EXTREME MOVE
+
EXHAUSTION
+
FAILED CONTINUATION
+
REVERSAL CONFIRMATION
```

C'est beaucoup plus robuste.

---

# 5. Overnight Gap Continuation

Une autre stratégie intéressante pour ton architecture.

Tu autorises l'overnight : **profitons-en.**

Exemple :

```text
Previous close     $100
Premarket          $106
Open               $107
```

Mais au lieu de trader immédiatement :

```text
gap +7%
```

tu demandes :

```text
gap > +3%
relative volume high
news/catalyst optional
first 30–60 min holds
VWAP holds
high of day gets challenged
```

Puis :

```text
BUY
```

et éventuellement :

```text
hold overnight
```

Il faut cependant être prudent avec les stratégies basées uniquement sur les gaps. Des travaux récents montrent notamment des phénomènes de **reversal après des jumps overnight**, donc je testerais explicitement *continuation vs reversal* plutôt que de supposer que les gaps se poursuivent. ([ScienceDirect][4])

---

# 6. Volatility Contraction → Expansion

Celui-ci pourrait être **très intéressant pour ton stock picker**.

Tu recherches :

```text
high quality stock
       ↓
uptrend
       ↓
volatility contraction
       ↓
tight range
       ↓
volume dries up
       ↓
breakout
       ↓
expansion
```

Par exemple :

```text
Day 1     +4%
Day 2     +2%
Day 3     +0.8%
Day 4     -0.4%
Day 5     +0.2%
          ↑
       compression
```

Puis :

```text
volume ↑
range ↑
breakout
```

→ BUY.

L'intérêt est que tu n'achètes pas un titre déjà complètement étendu.

---

# 7. Relative Strength vs Sector

Je pense que celui-ci est particulièrement pertinent **compte tenu de ton problème avec le SP500**.

Tu m'avais indiqué que ton cross-sectional momentum reste très corrélé au S&P 500.

Je ne chercherais donc plus seulement :

```text
stock momentum
```

mais :

```text
stock momentum
        -
sector momentum
        -
market momentum
```

Par exemple :

```text
SPY       +0.1%
XLK       +0.2%
NVDA      +2.8%
```

est beaucoup plus intéressant que :

```text
SPY       +1.8%
XLK       +2.1%
NVDA      +2.8%
```

Dans le premier cas :

**NVDA crée réellement de l'alpha relatif.**

Dans le deuxième :

**NVDA monte probablement simplement avec le marché.**

---

# Ce que je ferais avec ton architecture

Je ne multiplierais surtout pas les agents immédiatement.

Je construirais **6 stratégies très différentes**, chacune avec son propre scanner :

```text
                    STOCK UNIVERSE
                          │
              ┌───────────┴───────────┐
              │                       │
        liquidity filter        market filter
              │                       │
              └───────────┬───────────┘
                          │
                    STOCK SCANNERS
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
     MOMENTUM           EVENT           REVERSAL
        │                 │                 │
   RS Pullback          PEAD       Exhaustion MR
   52W High             Gap        Intraday reversal
   Vol Compression
        │                 │                 │
        └─────────────────┼─────────────────┘
                          │
                    RANK / SCORE
                          │
                    RISK FILTER
                          │
                       EXECUTE
```

Et surtout :

### Je séparerais le **signal** du **timing d'entrée**.

Par exemple :

```text
SIGNAL
"Cette action a une forte probabilité de continuer à monter
sur les 1–5 prochains jours."

             ↓

ENTRY AGENT
"Quand exactement dois-je acheter ?"
```

C'est probablement une amélioration importante par rapport à ton approche actuelle.

---

# Mon top 3 à tester maintenant

Si je devais choisir **seulement trois backtests** pour ton projet :

### 🥇 PEAD / Earnings Momentum

```text
Horizon : 1–10 jours
Signal : événement fondamental + price action
Long only : excellent
Overnight : oui
Intraday timing : secondaire
```

### 🥈 Relative Strength Pullback

```text
Horizon : 1–5 jours
Signal : stock > sector > market
Entry : pullback contrôlé
Long only : excellent
```

### 🥉 52W High + Volatility Contraction

```text
Horizon : 1–10 jours
Signal : compression + breakout structurel
Long only : excellent
```

Et je garderais **Intraday Exhaustion Mean Reversion** comme quatrième stratégie, car elle est intéressante précisément parce qu'elle est **anti-correlée conceptuellement** avec tes stratégies de continuation.

---

### Une remarque importante

Je **ne chercherais pas forcément une stratégie "intraday"** maintenant.

Ton objectif réel semble plutôt être :

> **détecter des actions qui ont une asymétrie positive à très court terme, avec une durée de détention de quelques heures à quelques jours.**

C'est différent.

Les travaux récents sur momentum/reversal montrent d'ailleurs que les composantes **intraday et overnight ont des comportements très différents** ; une étude publiée en 2026 trouve notamment du momentum dans les composantes intraday historiques, mais pas dans les composantes overnight. ([OUP Academic][5])

Donc je laisserais ton agent décider :

```text
BUY at 11:30
      ↓
hold until close
      ↓
if signal remains strong
      ↓
hold overnight
      ↓
exit next day / +N days
```

plutôt que d'imposer artificiellement :

```text
BUY → SELL before 16:00
```

**Pour ton prochain test, mon choix serait PEAD + Relative Strength, avec une logique de holding 1–10 jours.** C'est probablement beaucoup plus susceptible de produire un profil P&L différent de ton ORB/VWAP que de tester encore une autre variante de breakout intraday.

[1]: https://www.sciencedirect.com/science/article/pii/S1057521917300212?utm_source=chatgpt.com "Competitive earnings news and post-earnings announcement drift - ScienceDirect"
[2]: https://onlinelibrary.wiley.com/doi/full/10.1111/j.1540-6261.2004.00695.x?utm_source=chatgpt.com "The 52‐Week High and Momentum Investing - GEORGE - 2004 - The Journal of Finance - Wiley Online Library"
[3]: https://doi.org/10.1142/S2010139219500022?utm_source=chatgpt.com "Short-Term Return Reversals and Intraday Transactions | The Quarterly Journal of Finance"
[4]: https://www.sciencedirect.com/science/article/pii/S2214635026000821?utm_source=chatgpt.com "Dark side of the day: Overnight price jumps and short-term return predictability - ScienceDirect"
[5]: https://academic.oup.com/rfs/article-abstract/39/10/3378/8626980?utm_source=chatgpt.com "What Drives Momentum and Reversal? Evidence from Day and Night Signals | The Review of Financial Studies | Oxford Academic"
