"""
Beer in the Classroom: Location & Distribution Decisions at Anadolu Efes
Koksalan & Salman (2003), INFORMS Transactions on Education 4(1):65-77

Reproduction in Pyomo
  Part I  : single-period LP  (18 variables, 13 constraints)
  Part II : 3-year multi-period MIP (135 continuous, 18 binary, 72 constraints)
  What-ifs: discount rate, planning horizon, forced early opening, site comparison

Run:   pip install pyomo highspy        (HiGHS solver)   -> python efes_pyomo.py
       or install any of: cbc, glpk, gurobi, cplex      (auto-detected)
Units: malt = 1000 tons, beer = million liters, cost = million $
"""
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, TerminationCondition

# ----------------------------------------------------------------------------
# 1. DATA  (Exhibits 6-10 of the case)
# ----------------------------------------------------------------------------
PLANTS   = ['Afyon', 'Konya', 'Import']                 # malt sources (Import = Izmir harbor)
EXISTING = ['Istanbul', 'Ankara']
POTENTIAL = ['Izmir', 'Sakarya', 'Adana']
BREWERIES = EXISTING + POTENTIAL
DISTS    = ['Istanbul', 'Izmir', 'Antalya', 'Bursa', 'Kayseri', 'Export']
YEARS    = [1, 2, 3]

# Exhibit 6a: malt cost, plant -> brewery  (M$ / 1000 tons)
_mc = {
    'Afyon':  [0.026, 0.017, 0.020, 0.019, 0.032],
    'Konya':  [0.037, 0.017, 0.031, 0.030, 0.022],
    'Import': [0.032, 0.033, 0.004, 0.028, 0.048],
}
MALT_COST = {(i, j): c for i in PLANTS for j, c in zip(BREWERIES, _mc[i])}

# Exhibit 6b: beer cost, brewery -> distribution center  (M$ / million liters)
_bc = {
    'Istanbul': [0.000, 0.040, 0.052, 0.017, 0.055, 0.042],
    'Ankara':   [0.032, 0.041, 0.039, 0.027, 0.023, 0.043],
    'Izmir':    [0.040, 0.000, 0.032, 0.023, 0.062, 0.002],
    'Sakarya':  [0.011, 0.034, 0.041, 0.011, 0.045, 0.036],
    'Adana':    [0.067, 0.064, 0.040, 0.060, 0.024, 0.066],
}
BEER_COST = {(j, k): c for j in BREWERIES for k, c in zip(DISTS, _bc[j])}

# Exhibit 7: demand (million liters / year)
_dem = {1: [103, 74, 50, 60, 102, 13],
        2: [110, 80, 53, 75, 110, 13],
        3: [125, 90, 60, 85, 125, 15]}
DEMAND = {(k, t): _dem[t][n] for t in YEARS for n, k in enumerate(DISTS)}

# Exhibit 8: capacities and yields
MALT_CAP   = {'Afyon': 30, 'Konya': 68, 'Import': 20}          # 1000 tons / yr
BREW_CAP   = {'Istanbul': 220, 'Ankara': 200}                  # million liters / yr
NEW_CAP, EXP_CAP = 70, 50                                      # new brewery / expansion
YIELD      = {'Afyon': 8.333, 'Konya': 8.333, 'Import': 9.091} # M liters beer per 1000 t malt

# Exhibit 10: fixed costs (M$)
OPEN_COST   = {'Izmir': 75, 'Sakarya': 70, 'Adana': 68}
EXPAND_COST = {'Izmir': 30, 'Sakarya': 27, 'Adana': 25}

# ----------------------------------------------------------------------------
# 2. SOLVER HELPER
# ----------------------------------------------------------------------------
def get_solver(prefer=('appsi_highs', 'highs', 'cbc', 'glpk', 'gurobi', 'cplex')):
    for name in prefer:
        try:
            s = SolverFactory(name)
            if s.available(exception_flag=False):
                return s, name
        except Exception:
            continue
    raise RuntimeError("No solver found. Try:  pip install highspy   (or install cbc / glpk)")

# ----------------------------------------------------------------------------
# 3. PART I - SINGLE-PERIOD DISTRIBUTION LP
# ----------------------------------------------------------------------------
def build_lp(keep_relations=False, year=1):
    """x[i,j] malt plant i -> existing brewery j ; y[j,k] beer brewery j -> distributor k"""
    m = pyo.ConcreteModel('Efes_Part1_LP')
    m.P = pyo.Set(initialize=PLANTS)
    m.B = pyo.Set(initialize=EXISTING)
    m.D = pyo.Set(initialize=DISTS)

    m.x = pyo.Var(m.P, m.B, domain=pyo.NonNegativeReals)   # 6 vars
    m.y = pyo.Var(m.B, m.D, domain=pyo.NonNegativeReals)   # 12 vars

    m.obj = pyo.Objective(
        expr=sum(MALT_COST[i, j] * m.x[i, j] for i in m.P for j in m.B)
           + sum(BEER_COST[j, k] * m.y[j, k] for j in m.B for k in m.D),
        sense=pyo.minimize)

    # malt plants: shipments <= capacity                                   (3)
    m.malt_cap = pyo.Constraint(m.P, rule=lambda m, i:
        sum(m.x[i, j] for j in m.B) <= MALT_CAP[i])
    # malt -> beer balance at breweries: beer shipped <= yield * malt in   (2)
    m.balance = pyo.Constraint(m.B, rule=lambda m, j:
        sum(m.y[j, k] for k in m.D) <= sum(YIELD[i] * m.x[i, j] for i in m.P))
    # brewery capacity                                                     (2)
    m.brew_cap = pyo.Constraint(m.B, rule=lambda m, j:
        sum(m.y[j, k] for k in m.D) <= BREW_CAP[j])
    # demand satisfaction                                                  (6)
    m.demand = pyo.Constraint(m.D, rule=lambda m, k:
        sum(m.y[j, k] for j in m.B) == DEMAND[k, year])

    if keep_relations:   # Ankara serves ALL of Bursa, Istanbul serves ALL of Antalya
        m.y['Istanbul', 'Bursa'].fix(0)
        m.y['Ankara', 'Antalya'].fix(0)

    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)       # shadow prices
    m.rc   = pyo.Suffix(direction=pyo.Suffix.IMPORT)       # reduced costs
    return m

def current_plan_cost():
    """Cost of the Exhibit-9 'current' plan, evaluated with the same cost data."""
    malt = {('Afyon', 'Ankara'): 24, ('Konya', 'Istanbul'): 2.42, ('Import', 'Istanbul'): 20}
    beer = {('Istanbul', 'Istanbul'): 103, ('Istanbul', 'Izmir'): 49, ('Istanbul', 'Antalya'): 50,
            ('Ankara', 'Izmir'): 25, ('Ankara', 'Bursa'): 60, ('Ankara', 'Kayseri'): 102,
            ('Ankara', 'Export'): 13}
    return (sum(MALT_COST[a] * v for a, v in malt.items())
          + sum(BEER_COST[a] * v for a, v in beer.items()))

def run_part1():
    solver, name = get_solver()
    print(f"\n=== PART I : single-period LP  (solver: {name}) ===")
    cur = current_plan_cost()
    print(f"Current plan (Exhibit 9) cost            : {cur:8.3f} M$/yr")

    res = {}
    for label, keep in [('Re-optimised, keep relations', True), ('Fully optimised (free)', False)]:
        m = build_lp(keep_relations=keep)
        solver.solve(m)
        res[label] = (m, pyo.value(m.obj))
        print(f"{label:<41}: {pyo.value(m.obj):8.3f} M$/yr")

    m_free = res['Fully optimised (free)'][0]
    opt = res['Fully optimised (free)'][1]
    print(f"Savings vs current plan                  : {cur-opt:8.3f} M$/yr ({100*(cur-opt)/cur:.1f}%)")
    print(f"Cost of keeping the two relations        : {res['Re-optimised, keep relations'][1]-opt:8.3f} M$/yr")

    print("\nOptimal malt shipments (1000 t):")
    for i in PLANTS:
        for j in EXISTING:
            v = pyo.value(m_free.x[i, j])
            if v > 1e-6: print(f"   {i:>7} -> {j:<9}{v:8.2f}")
    print("Optimal beer shipments (million liters):")
    for j in EXISTING:
        for k in DISTS:
            v = pyo.value(m_free.y[j, k])
            if v > 1e-6: print(f"   {j:>9} -> {k:<9}{v:8.1f}")

    # ---- sensitivity (needs a solver that returns duals: cbc / glpk / gurobi / cplex)
    try:
        s2, n2 = get_solver(prefer=('cbc', 'glpk', 'gurobi', 'cplex', 'highs', 'appsi_highs'))
        m = build_lp(); s2.solve(m)
        print(f"\nSensitivity ({n2}):")
        print("  Reduced costs of unused arcs (M$ per unit):")
        for (j, k) in [('Istanbul', 'Antalya'), ('Ankara', 'Bursa')]:
            print(f"   beer {j}->{k}: {m.rc[m.y[j, k]]:.4f}  => cost must fall from "
                  f"{BEER_COST[j,k]:.3f} to {BEER_COST[j,k]-m.rc[m.y[j,k]]:.3f} to be used")
        for j in EXISTING:
            print(f"   malt Import->{j}: {m.rc[m.x['Import', j]]:.4f}")
        print("  Shadow prices of demand (M$ per extra million liters):")
        for k in DISTS:
            print(f"   {k:<9}{m.dual[m.demand[k]]:.4f}")
    except Exception as e:
        print("\n(Sensitivity skipped - this solver interface returned no duals:", e, ")")
    return res

# ----------------------------------------------------------------------------
# 4. PART II - MULTI-PERIOD LOCATION / EXPANSION MIP
# ----------------------------------------------------------------------------
def build_mip(r=0.10, horizon=20, force_open=None):
    """
    open[s,t]   = 1 if new brewery at site s is opened at the start of year t
    expand[s,t] = 1 if it is expanded at the start of year t
    Objective   = PV(transport, 20 yrs) + PV(opening + expansion costs)
    Years 1-3 are modelled explicitly; years 4..horizon repeat the year-3 plan.
    """
    m = pyo.ConcreteModel('Efes_Part2_MIP')
    m.P = pyo.Set(initialize=PLANTS)
    m.B = pyo.Set(initialize=BREWERIES)
    m.S = pyo.Set(initialize=POTENTIAL)
    m.E = pyo.Set(initialize=EXISTING)
    m.D = pyo.Set(initialize=DISTS)
    m.T = pyo.Set(initialize=YEARS, ordered=True)

    m.x = pyo.Var(m.P, m.B, m.T, domain=pyo.NonNegativeReals)   # 45 malt + 90 beer variables = 135
    m.y = pyo.Var(m.B, m.D, m.T, domain=pyo.NonNegativeReals)
    m.open   = pyo.Var(m.S, m.T, domain=pyo.Binary)             # 9
    m.expand = pyo.Var(m.S, m.T, domain=pyo.Binary)             # 9

    # discount factors: transport paid at END of year t, investments at START of year t
    w = {t: 1.0 / (1 + r) ** t for t in YEARS}
    w[3] = sum(1.0 / (1 + r) ** tau for tau in range(3, horizon + 1))   # year 3 repeated to horizon
    fixed_df = {t: 1.0 / (1 + r) ** (t - 1) for t in YEARS}

    m.transport_pv = pyo.Expression(expr=sum(
        w[t] * (sum(MALT_COST[i, j] * m.x[i, j, t] for i in m.P for j in m.B)
              + sum(BEER_COST[j, k] * m.y[j, k, t] for j in m.B for k in m.D))
        for t in m.T))
    m.fixed_pv = pyo.Expression(expr=sum(
        fixed_df[t] * (OPEN_COST[s] * m.open[s, t] + EXPAND_COST[s] * m.expand[s, t])
        for s in m.S for t in m.T))
    m.obj = pyo.Objective(expr=m.transport_pv + m.fixed_pv, sense=pyo.minimize)

    # malt plant capacity (per year)
    m.malt_cap = pyo.Constraint(m.P, m.T, rule=lambda m, i, t:
        sum(m.x[i, j, t] for j in m.B) <= MALT_CAP[i])
    # malt -> beer balance
    m.balance = pyo.Constraint(m.B, m.T, rule=lambda m, j, t:
        sum(m.y[j, k, t] for k in m.D) <= sum(YIELD[i] * m.x[i, j, t] for i in m.P))
    # existing breweries: fixed capacity
    m.cap_exist = pyo.Constraint(m.E, m.T, rule=lambda m, j, t:
        sum(m.y[j, k, t] for k in m.D) <= BREW_CAP[j])
    # potential breweries: capacity only after the fixed cost has been paid ("no free lunch")
    m.cap_new = pyo.Constraint(m.S, m.T, rule=lambda m, s, t:
        sum(m.y[s, k, t] for k in m.D)
        <= NEW_CAP * sum(m.open[s, tau] for tau in m.T if tau <= t)
         + EXP_CAP * sum(m.expand[s, tau] for tau in m.T if tau <= t))
    # demand
    m.demand = pyo.Constraint(m.D, m.T, rule=lambda m, k, t:
        sum(m.y[j, k, t] for j in m.B) == DEMAND[k, t])
    # at most ONE new brewery, opened at most once, expanded at most once
    m.one_brewery = pyo.Constraint(expr=sum(m.open[s, t] for s in m.S for t in m.T) <= 1)
    m.one_expand  = pyo.Constraint(m.S, rule=lambda m, s:
        sum(m.expand[s, t] for t in m.T) <= 1)
    # cannot expand a brewery that is not yet open
    m.expand_needs_open = pyo.Constraint(m.S, m.T, rule=lambda m, s, t:
        sum(m.expand[s, tau] for tau in m.T if tau <= t)
        <= sum(m.open[s, tau] for tau in m.T if tau <= t))

    if force_open:                       # e.g. ('Izmir', 1) for what-if analysis
        m.open[force_open].fix(1)
    return m

def decisions(m):
    out = []
    for s in POTENTIAL:
        for t in YEARS:
            if pyo.value(m.open[s, t]) > 0.5:   out.append(f"OPEN {s} in year {t}")
            if pyo.value(m.expand[s, t]) > 0.5: out.append(f"EXPAND {s} in year {t}")
    return out

def run_part2():
    solver, name = get_solver()
    print(f"\n=== PART II : multi-period MIP  (solver: {name}) ===")
    m = build_mip()
    n_bin = sum(1 for _ in m.open) + sum(1 for _ in m.expand)
    print(f"Variables: {len(m.x)+len(m.y)} continuous, {n_bin} binary")
    solver.solve(m)
    print(f"Optimal PV cost : {pyo.value(m.obj):8.2f} M$   "
          f"(transport {pyo.value(m.transport_pv):.2f} + fixed {pyo.value(m.fixed_pv):.2f})")
    print("Decisions       :", decisions(m))
    for t in YEARS:
        print(f"\n  Year {t} beer shipments (million liters):")
        for j in BREWERIES:
            for k in DISTS:
                v = pyo.value(m.y[j, k, t])
                if v > 1e-6: print(f"     {j:>9} -> {k:<9}{v:7.1f}")
        used = {j: sum(pyo.value(m.y[j, k, t]) for k in DISTS) for j in BREWERIES}
        print("     brewery output:", {j: round(v, 1) for j, v in used.items() if v > 1e-6})
    return m

# ----------------------------------------------------------------------------
# 5. WHAT-IF ANALYSES (case questions 5-7)
# ----------------------------------------------------------------------------
def run_whatifs():
    solver, _ = get_solver()

    print("\n--- What-if 1: force the brewery to open EARLY (Q6) ---")
    base = build_mip(); solver.solve(base)
    print(f"Base case (r=10%): {pyo.value(base.obj):.2f} M$  {decisions(base)}")
    m = build_mip(force_open=('Izmir', 1)); solver.solve(m)
    print(f"Forced Izmir yr 1: {pyo.value(m.obj):.2f} M$  (extra cost {pyo.value(m.obj)-pyo.value(base.obj):.2f} M$)")

    print("\n--- What-if 2: best plan if each site is forced (opening in year 2) ---")
    for s in POTENTIAL:
        m = build_mip(force_open=(s, 2)); solver.solve(m)
        print(f"   {s:<8}: {pyo.value(m.obj):7.2f} M$  {decisions(m)}")

    print("\n--- What-if 3: opportunity cost of capital (Q6) ---")
    first_year1 = None
    for r in [0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.00]:
        m = build_mip(r=r); solver.solve(m)
        d = decisions(m)
        print(f"   r={r:4.0%}: {pyo.value(m.obj):7.2f} M$  {d}")
    print("   -> bisection (see slides): opening in year 1 becomes optimal when r falls below ~4%")

    print("\n--- What-if 4: length of cost horizon (Q5) ---")
    for H in [3, 10, 20, 30]:
        m = build_mip(horizon=H); solver.solve(m)
        print(f"   horizon {H:>2} yrs: {pyo.value(m.obj):7.2f} M$  {decisions(m)}")

if __name__ == '__main__':
    run_part1()
    run_part2()
    run_whatifs()
