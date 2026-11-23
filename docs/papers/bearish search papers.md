1. Time-series momentum / trend following (the core “crisis alpha” strategy)
Paper: Hurst, Ooi & Pedersen, A Century of Evidence on Trend-Following Investing. It tested 67 markets across four major asset classes (29 commodities, 11 equity indices, 15 bond markets and 12 currency pairs) from January 1880 to December 2016, and found that time-series momentum has performed well in eight out of 10 of the largest crisis periods over the century. 
alphaarchitect
Man/Oxford study (1960–2015): The strategy performed consistently well both before 1985 (bond bear market) and after 1985 (bond bull market), and its returns are positively skewed, especially for shorter-term (faster) momentum strategies. 
man
Code:
pysystemtrade is the open source version of Rob Carver’s own backtesting and trading engine. It recently moved to a new github “organisation”, pst-group. 
wasmer
PyTrendFollow is a simpler option for futures.
Caveat: One study found relatively weak returns for trend following strategies for, on average, four years following the start of a financial crisis. 
ucc
2. Faber’s 10-month SMA / GTAA (simplest to test)
Rules: check the price of each asset class on the last day of each month, and if it is greater than its 10-month simple moving average (SMA), you buy (or continue to hold). If the price is less than the 10-month SMA, you sell and move to cash. 
moneysense
Bear-market track record: It would have missed the equity market peak by only one month and avoided the whole mess during the global financial crisis. 
earlyretirementnow
Code: Concretum offers a complete Python implementation of the Global Tactical Asset Allocation Strategy, plus a walkthrough automated with IBKR. Their research also flags rebalance timing luck, so test several rebalance dates. 
concretumgroup
3. Dual Momentum (Antonacci)
Logic: Relative momentum picks the strongest asset, and absolute momentum moves you to bonds when the trend is negative. Antonacci’s paper found that relative momentum can enhance returns, but that absolute momentum does far more to lessen volatility and drawdown. 
scribd
Honest trade-off: Since 2009 this strategy has underperformed the S&P 500 by wide margin. But if you go back to 2008, dual momentum has actually done better than the S&P, because the strategy is built to perform better during bear markets. 
awealthofcommonsense
Recent variant: Quantpedia’s 2007–2026 study applies it to commodity ETFs (DBA, DBB, DBE, DBP). It found the inclusion of an absolute momentum filter structurally mitigates drawdowns. 
quantpedia
4. Keller’s “canary” strategies (DAA, BAA, HAA)
Idea: a separate “canary” universe for signaling the need for crash protection, using the concept of breadth momentum. 
blogspot
DAA rule: When both canary assets VWO and BND register negative 13612W momentum, invest 100% in the single best bond of the cash universe. 
blogspot
BAA: It combines a slow relative momentum with a fast absolute momentum and crash protection (SSRN 4166845). 
thechartist
Code:
HAApy is a Python implementation of Hybrid Asset Allocation.
QuantConnect forum threads have BAA and HAA code. One user reported that results in the back test are not as good as in the paper, partly because it traded at the next day’s open instead of the month-end close. That’s a good lesson in execution assumptions. 
quantconnect
5. Defensive equity factors (quality and low volatility)
Evidence: In COVID and the GFC, defensive factors such as quality and low volatility mitigated the drawdown, while rather cyclical factors, namely value and low size, underperformed. 
invesco
Reality check from Man Group: Low-Volatility strategies can still lose 60% of what markets lose and perform worse during severe stress; simple Quality strategies provide modest protection and take several weeks to work. That makes them dampeners, not hedges. 
man
6. Dynamic momentum (avoiding the bear-market rebound trap)
The problem: Cross-sectional stock momentum is risky in bear markets. Crashes predominantly occur at pivotal points in the economy, when a bear market suddenly ends and stocks dramatically rebound. 
chicagobooth
Fix 1 (Daniel & Moskowitz): An implementable dynamic momentum strategy based on forecasts of momentum’s mean and variance approximately doubles the alpha and Sharpe Ratio of a static momentum strategy (NBER w20439). 
nber
Fix 2 (simpler HSE variant): It switches to the opposite contrarian strategy after a market crash and keeps the contrarian position for three months. 
hse
Toolkits for testing
R: The AssetAllocation package ships with ready-made tactical strategies including Ivy Portfolio, Robust Asset Allocation, Dual Momentum, Adaptive Asset Allocation. 
r-project
Python: bt, backtesting.py and Ilya Kipnis’s PythonBacktesting repo.