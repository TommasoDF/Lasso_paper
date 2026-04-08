"""
10-K × Topic cosine similarity → narrative alignment test.

Pipeline:
  1. Map tensor PERMNOs → tickers via stock_names.csv
  2. Fetch CIK for each ticker from SEC EDGAR company_tickers.json
  3. For each CIK, download the most recent 10-K Item-1 business description
     (cached in _cache/tenk/ to allow resume after interruption)
  4. TF-IDF vectorise firm texts + topic key-term strings
  5. Cosine similarity matrix (stocks × topics) → continuous relevance score
  6. Panel OLS with stock + topic FEs; compare to manual-binary baseline

Usage:
  python tenk_topic_similarity.py [--max-stocks N]  (default: all 993)
"""

import os, time, re, pickle, argparse, json
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
import matplotlib.pyplot as plt
import h5py

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
TENSOR    = os.path.join(ROOT, 'Results/Estimation/Cross_Sectional_volscaled/betas.h5')
NAMES_CSV = os.path.join(ROOT, 'Data/data_raw/stock_names.csv')
TOPICS_CSV= os.path.join(ROOT, 'Narratives/narratives_construction/_data/wsj_topics_ia_c_table_IAI_augmented.csv')
OUT_DIR   = os.path.join(ROOT, 'Results/Estimation/Cross_Sectional_volscaled')
CACHE_DIR = os.path.join(os.path.dirname(__file__), '_cache/tenk')
os.makedirs(CACHE_DIR, exist_ok=True)

HEADERS = {'User-Agent': 'Research project lasso-paper@research.edu'}
SLEEP   = 0.12   # stay under SEC 10 req/s limit

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--max-stocks', type=int, default=None)
args = parser.parse_args()

# ── 1. Load tensor + build PERMNO → ticker map ────────────────────────────────
print("Loading tensor …")
with h5py.File(TENSOR, 'r') as f:
    stocks = f['stocks'][:].astype(str)
    topics = f['topics'][:].astype(str)
    n_stocks, n_topics, n_dates = f['betas'].shape
    print(f"Tensor shape: ({n_stocks}, {n_topics}, {n_dates}) — computing selection rates in chunks …")

    # Compute per_stock_feat_rate without loading full tensor into memory
    # Process in batches of 100 stocks at a time
    per_stock_feat_rate = np.zeros((n_stocks, n_topics), dtype=np.float32)
    valid_count = 0
    BATCH = 100
    for start in range(0, n_stocks, BATCH):
        end = min(start + BATCH, n_stocks)
        chunk = f['betas'][start:end, :, :]   # (batch, n_topics, n_dates)
        # Find valid dates from first batch
        if start == 0:
            valid_mask = np.isfinite(chunk).any(axis=(0, 1))
            valid_count = valid_mask.sum()
            print(f"  Valid dates: {valid_count} / {n_dates}")
        chunk_valid = chunk[:, :, valid_mask]
        per_stock_feat_rate[start:end] = (chunk_valid != 0).mean(axis=2).astype(np.float32)

print(f"Selection rate matrix: {per_stock_feat_rate.shape}, mean={per_stock_feat_rate.mean()*100:.4f}%")

names = pd.read_csv(NAMES_CSV)
names['PERMNO'] = names['PERMNO'].astype(str)
latest = (names.sort_values('date')
               .groupby('PERMNO').last()
               .reset_index()[['PERMNO', 'TICKER', 'COMNAM']])
ticker_df = latest[latest['PERMNO'].isin(stocks)].copy()
print(f"Stocks with ticker: {len(ticker_df)} / {len(stocks)}")

if args.max_stocks:
    ticker_df = ticker_df.head(args.max_stocks)
    print(f"(Limited to {args.max_stocks} stocks for testing)")

# ── 2. CIK lookup ──────────────────────────────────────────────────────────────
cik_cache_path = os.path.join(CACHE_DIR, 'company_tickers.json')
if os.path.exists(cik_cache_path):
    with open(cik_cache_path) as f:
        cik_data = json.load(f)
else:
    print("Fetching company_tickers.json from EDGAR …")
    r = requests.get('https://www.sec.gov/files/company_tickers.json', headers=HEADERS, timeout=30)
    r.raise_for_status()
    cik_data = r.json()
    with open(cik_cache_path, 'w') as f:
        json.dump(cik_data, f)
    time.sleep(SLEEP)

# Build ticker → CIK dict (lowercase keys)
ticker_to_cik = {v['ticker'].upper(): str(v['cik_str']).zfill(10)
                 for v in cik_data.values()}
ticker_df['CIK'] = ticker_df['TICKER'].str.upper().map(ticker_to_cik)
matched = ticker_df['CIK'].notna().sum()
print(f"CIK matched: {matched} / {len(ticker_df)}")
ticker_df = ticker_df[ticker_df['CIK'].notna()].copy()

# ── 3. Fetch 10-K business description (cached) ────────────────────────────────
def get_cache_path(cik):
    return os.path.join(CACHE_DIR, f'{cik}.pkl')

def fetch_10k_text(cik, company_name):
    """Return Item-1 business description text for a given CIK, or '' on failure."""
    cache = get_cache_path(cik)
    if os.path.exists(cache) and os.path.getsize(cache) > 100:
        with open(cache, 'rb') as f:
            return pickle.load(f)

    try:
        # Step A: get submissions to find most recent 10-K accession
        url_sub = f'https://data.sec.gov/submissions/CIK{cik}.json'
        r = requests.get(url_sub, headers=HEADERS, timeout=20)
        time.sleep(SLEEP)
        if r.status_code != 200:
            return _cache_and_return(cache, '')
        sub = r.json()

        filings = sub.get('filings', {}).get('recent', {})
        forms   = filings.get('form', [])
        accnums = filings.get('accessionNumber', [])
        dates   = filings.get('filingDate', [])

        # Find most recent 10-K
        tenk_idx = next((i for i, f in enumerate(forms) if f == '10-K'), None)
        if tenk_idx is None:
            return _cache_and_return(cache, '')

        accession = accnums[tenk_idx].replace('-', '')
        cik_plain = str(int(cik))  # without leading zeros for URL

        # Step B: fetch filing index to locate primary document
        idx_url = (f'https://www.sec.gov/Archives/edgar/data/'
                   f'{cik_plain}/{accession}/{accnums[tenk_idx]}-index.htm')
        r2 = requests.get(idx_url, headers=HEADERS, timeout=20)
        time.sleep(SLEEP)
        if r2.status_code != 200:
            return _cache_and_return(cache, '')

        soup_idx = BeautifulSoup(r2.content, 'html.parser')
        # Find primary 10-K document — look for type '10-K' in column 4
        doc_url = None
        for row in soup_idx.find_all('tr'):
            cells = row.find_all('td')
            if len(cells) >= 4:
                doc_type = cells[3].get_text(strip=True)
                if doc_type == '10-K':
                    link = row.find('a')
                    if link and link.get('href'):
                        href = link['href']
                        # Strip iXBRL viewer prefix /ix?doc=...
                        m = re.match(r'^/ix\?doc=(.+)$', href)
                        if m:
                            href = m.group(1)
                        doc_url = 'https://www.sec.gov' + href
                        break

        if doc_url is None:
            # fallback: first .htm link in the index that isn't an exhibit
            for a in soup_idx.find_all('a', href=True):
                h = a['href']
                if h.endswith('.htm') and accession in h and 'ex' not in h.lower():
                    doc_url = 'https://www.sec.gov' + h
                    break

        if doc_url is None:
            return _cache_and_return(cache, '')

        # Step C: download 10-K document (stream, limit to 800 KB for full Item 1)
        r3 = requests.get(doc_url, headers=HEADERS, timeout=30, stream=True)
        time.sleep(SLEEP)
        if r3.status_code != 200:
            return _cache_and_return(cache, '')

        raw = b''
        for chunk in r3.iter_content(chunk_size=32768):
            raw += chunk
            if len(raw) > 800_000:
                break

        # Parse HTML / iXBRL — text is embedded in tags
        soup = BeautifulSoup(raw, 'html.parser')
        for tag in soup(['script', 'style', 'ix:header', 'link', 'meta']):
            tag.decompose()
        full_text = soup.get_text(separator=' ', strip=True)

        # Extract Item 1 section (heuristic); fall back to full text
        text = _extract_item1(full_text)
        return _cache_and_return(cache, text)

    except Exception as e:
        print(f"  ERROR {cik} ({company_name}): {e}")
        return _cache_and_return(cache, '')


def _cache_and_return(path, text):
    with open(path, 'wb') as f:
        pickle.dump(text, f, protocol=4)
    return text


def _extract_item1(text):
    """Heuristically extract Item 1 (Business) from 10-K plain text."""
    text = re.sub(r'\s+', ' ', text)

    # Look for the actual Item 1 business section (after table of contents)
    # Pattern: "Item 1" followed by "Business" then content until "Item 1A" or "Item 2"
    patterns = [
        r'ITEM\s+1\.\s+BUSINESS\s+(.*?)(?:ITEM\s+1A|ITEM\s+2)',
        r'Item\s+1\.\s+Business\s+(.*?)(?:Item\s+1A|Item\s+2)',
        r'ITEM\s+1\s+BUSINESS\s+(.*?)(?:ITEM\s+1A|ITEM\s+2)',
    ]
    for pat in patterns:
        # Search from second occurrence to skip table of contents
        matches = list(re.finditer(pat, text, re.DOTALL))
        for m in matches:
            content = m.group(1).strip()
            if len(content) > 500:   # must be substantial
                return content[:10_000]

    # Fallback: use a big chunk of the middle of the document
    # (skip first 2000 chars which is usually the cover page)
    return text[2000:12_000]


# Fetch all
print(f"\nFetching 10-K texts for {len(ticker_df)} companies …")
texts = {}
for i, row in ticker_df.iterrows():
    permno, ticker, comnam, cik = row['PERMNO'], row['TICKER'], row['COMNAM'], row['CIK']
    cached = os.path.exists(get_cache_path(cik))
    if not cached:
        print(f"  [{i+1}/{len(ticker_df)}] {ticker} ({comnam}) — fetching …")
    text = fetch_10k_text(cik, comnam)
    texts[permno] = text

n_ok = sum(1 for t in texts.values() if len(t) > 100)
print(f"\nSuccessfully fetched: {n_ok} / {len(texts)}")

# ── 4. TF-IDF vectorisation ────────────────────────────────────────────────────
print("\nVectorising texts …")

# Topic key terms (only news topics, i.e. those in the WSJ table)
topic_df = pd.read_csv(TOPICS_CSV)
topic_df['topic_key'] = topic_df['topic_label'].str.replace(' ', '_').str.replace('/', '/')

# Build dict: tensor_topic_name → key_terms_string
topic_terms = {}
for _, row in topic_df.iterrows():
    key = row['topic_label']
    # match against tensor topic names (underscores vs spaces)
    tensor_key = key.replace(' ', '_')
    terms = str(row['key_terms']).replace('\n', ' ')
    topic_terms[tensor_key] = terms

# Also add macro topics with simple names
macro_topics = {
    'Fed_Funds_Effective_Rate': 'federal funds rate interest rate federal reserve monetary policy',
    '10-Year_Treasury_Yield':   'treasury yield bond interest rate government debt 10 year',
    'VIX_Volatility_Index':     'volatility vix option fear index market volatility',
    'Financial_Stress_Index':   'financial stress credit market risk systemic',
    'High_Yield_Option-Adjusted_Spread': 'high yield junk bond credit spread default risk',
    'Trade_Weighted_USD_Index': 'dollar exchange rate currency trade weighted',
    '3-Month_Treasury_Yield':   'treasury bill short term interest rate money market',
    'WTI_Crude_Oil':            'crude oil wti oil price petroleum energy barrel',
    '10-Year_Breakeven_Inflation_Rate': 'inflation breakeven tips expected inflation',
    'USD_to_JPY':               'dollar yen japan currency exchange rate',
}
topic_terms.update(macro_topics)

# Align to tensor topic order
topic_docs  = []
topic_names_aligned = []
for t in topics:
    doc = topic_terms.get(t, t.replace('_', ' '))
    topic_docs.append(doc)
    topic_names_aligned.append(t)

# Firm documents
valid_permnos = [p for p in ticker_df['PERMNO'] if len(texts.get(p, '')) > 500]
firm_docs     = [texts[p] for p in valid_permnos]
print(f"Firms with usable text: {len(firm_docs)}")

# Fit TF-IDF on combined corpus
all_docs = firm_docs + topic_docs
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    max_df=0.95,
    min_df=2,
    max_features=50_000,
)
tfidf_all = vectorizer.fit_transform(all_docs)

n_firm = len(firm_docs)
tfidf_firms  = tfidf_all[:n_firm]
tfidf_topics = tfidf_all[n_firm:]

# Cosine similarity: (firms, topics)
sim_matrix = cosine_similarity(tfidf_firms, tfidf_topics)   # shape (n_firms, n_topics)
print(f"Similarity matrix: {sim_matrix.shape}")
print(f"Mean similarity: {sim_matrix.mean():.4f}  Max: {sim_matrix.max():.4f}")

# ── 5. Panel regression with continuous similarity score ───────────────────────
print("\nBuilding panel …")
permno_to_stock_idx = {p: i for i, p in enumerate(stocks)}
topic_idx_map = {t: i for i, t in enumerate(topics)}

rows = []
for f_idx, permno in enumerate(valid_permnos):
    s_idx = permno_to_stock_idx.get(permno)
    if s_idx is None:
        continue
    for t_idx, topic in enumerate(topics):
        sel_rate = float(per_stock_feat_rate[s_idx, t_idx])
        sim      = float(sim_matrix[f_idx, t_idx])
        rows.append((permno, topic, sel_rate, sim))

panel = pd.DataFrame(rows, columns=['permno', 'topic', 'sel_rate', 'sim'])
print(f"Panel shape: {panel.shape}")

# Within-transformation (stock FE + topic FE)
def within(df, col):
    ms = df.groupby('permno')[col].transform('mean')
    mt = df.groupby('topic')[col].transform('mean')
    mg = df[col].mean()
    return df[col] - ms - mt + mg

panel['y_w'] = within(panel, 'sel_rate')
panel['x_w'] = within(panel, 'sim')

X = panel['x_w'].values
y = panel['y_w'].values

beta_fe = np.cov(X, y)[0, 1] / np.var(X)
resid   = y - beta_fe * X
n       = len(y)
se      = np.sqrt((resid**2).sum() / (n - 2) / (np.var(X) * n))
t_fe    = beta_fe / se
p_fe    = 2 * stats.t.sf(abs(t_fe), df=n - 2)

print(f"\nPanel OLS (stock + topic FEs):")
print(f"  β(10-K similarity) = {beta_fe:.6f}   t = {t_fe:.3f}   p = {p_fe:.4f}")
print(f"  [A 1-SD increase in cosine similarity → "
      f"{beta_fe * panel['sim'].std():.6f} pp higher selection rate]")

# ── 6. Sector-level summary ───────────────────────────────────────────────────
ind_df = pd.read_csv(os.path.join(ROOT, 'Data/data_raw/industry_codes.csv'))
ind_df = ind_df.groupby('PERMNO')['HSICCD'].last().reset_index()
ind_df['PERMNO'] = ind_df['PERMNO'].astype(str)

def sic_to_sector(sic):
    s = int(sic)
    if 1000 <= s <= 1499 or 2900 <= s <= 2999:    return 'Energy & Mining'
    if 6000 <= s <= 6999:                           return 'Financials'
    if s in range(3570,3580) or s in range(3660,3700) or s in range(7370,7380):
        return 'Technology'
    if (2830 <= s <= 2836) or (8000 <= s <= 8099):  return 'Healthcare'
    if (2000 <= s <= 2399) or (5200 <= s <= 5999):  return 'Consumer'
    if (2400 <= s <= 2829 and not (2830 <= s <= 2836)) or \
       (3000 <= s <= 3569) or (3700 <= s <= 3799):  return 'Industrials'
    if 4800 <= s <= 4999:                            return 'Utilities & Telecom'
    return 'Other'

ind_df['sector'] = ind_df['HSICCD'].apply(sic_to_sector)
permno_to_sector = dict(zip(ind_df['PERMNO'], ind_df['sector']))
panel['sector'] = panel['permno'].map(permno_to_sector)

# For each sector: mean similarity of topics vs selection rate correlation
print(f"\nSpearman ρ(similarity, selection_rate) by sector:")
for sector, grp in panel[panel['sector'].notna()].groupby('sector'):
    rho, p_rho = stats.spearmanr(grp['sim'], grp['sel_rate'])
    print(f"  {sector:<22} n={len(grp):>6}  ρ={rho:+.4f}  p={p_rho:.4f}")

# ── 7. Figures ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: scatter sim vs selection rate (subsample for readability)
ax = axes[0]
sub = panel.sample(min(50000, len(panel)), random_state=42)
ax.scatter(sub['sim'], sub['sel_rate']*100, alpha=0.03, s=1, color='steelblue')
# Overlay mean selection rate binned by similarity percentile
bins = pd.qcut(panel['sim'], 20, duplicates='drop')
means = panel.groupby(bins)['sel_rate'].mean() * 100
mids  = panel.groupby(bins)['sim'].mean()
ax.plot(mids, means, color='red', lw=2, marker='o', ms=4, label='Bin mean')
ax.set_xlabel('Cosine similarity (10-K text vs topic key terms)')
ax.set_ylabel('Mean LASSO selection rate (%)')
ax.set_title(f'10-K similarity vs selection rate\n'
             f'Panel FE β={beta_fe:.5f}, t={t_fe:.2f}, p={p_fe:.4f}')
ax.legend()

# Right: distribution of cosine similarities — own-sector top topics vs rest
# Define "own-sector" as top-quartile similarity for each firm
ax2 = axes[1]
panel['sim_rank'] = panel.groupby('permno')['sim'].rank(pct=True)
high_sim = panel[panel['sim_rank'] >= 0.75]['sel_rate'] * 100
low_sim  = panel[panel['sim_rank'] <  0.25]['sel_rate'] * 100
ax2.hist(high_sim, bins=50, alpha=0.6, label=f'Top-quartile sim  (mean={high_sim.mean():.4f}%)', color='#1f77b4', density=True)
ax2.hist(low_sim,  bins=50, alpha=0.6, label=f'Bot-quartile sim  (mean={low_sim.mean():.4f}%)',  color='#aec7e8', density=True)
t_q, p_q = stats.ttest_ind(high_sim, low_sim)
ax2.set_xlabel('LASSO selection rate (%)')
ax2.set_ylabel('Density')
ax2.set_title(f'Selection rates: high vs low 10-K similarity\nt={t_q:.2f}, p={p_q:.4f}')
ax2.legend(fontsize=9)
ax2.set_xlim(-0.1, 5)

plt.tight_layout()
out_fig = os.path.join(OUT_DIR, 'tenk_topic_alignment.png')
plt.savefig(out_fig, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved {out_fig}")

# Save similarity matrix
sim_df = pd.DataFrame(sim_matrix, index=valid_permnos, columns=topics)
sim_df.to_csv(os.path.join(OUT_DIR, 'tenk_similarity_matrix.csv'))
print(f"Saved tenk_similarity_matrix.csv")

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"Firms with 10-K text : {len(firm_docs)}")
print(f"Panel observations   : {len(panel):,}")
print(f"Panel FE β           : {beta_fe:.6f}  t={t_fe:.3f}  p={p_fe:.4f}")
overall_rho, overall_p = stats.spearmanr(panel['sim'], panel['sel_rate'])
print(f"Overall Spearman ρ   : {overall_rho:+.4f}  p={overall_p:.4f}")
