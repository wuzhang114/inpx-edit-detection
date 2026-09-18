"""Descriptive budget fits with final ALL endpoints and budget-blocked CV.
Seven budget means, not 21 independent design points. All curves use exact
zero-budget limits. Information criteria are conditional iid-residual summaries,
not evidence for independence of nested budgets or a universal law.
"""
import json
import warnings
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.special import expit

OUT = Path(__file__).resolve().parents[1] / "outputs"
SEEDS = [42, 7, 2024]
KS = np.array([0, 10, 50, 200, 1000, 5000, 14731], dtype=float)


def hill(k, m0, amp, g, lk):
    x = np.log10(np.maximum(k, np.finfo(float).tiny))
    return m0 + amp * np.where(k > 0, expit(g * (x - lk)), 0.)


def mm(k, m0, amp, lk):
    return m0 + amp * k / (10**lk + k)


def weibull(k, m0, amp, lk, beta):
    return m0 + amp * (-np.expm1(-(k / 10**lk)**beta))


def richards(k, m0, amp, g, lk, nu):
    z = g * (np.log10(np.maximum(k, np.finfo(float).tiny)) - lk)
    f = np.exp(-np.logaddexp(0, np.log(nu) - z) / nu)
    return m0 + amp * np.where(k > 0, f, 0.)


def expo(k, m0, amp, lk):
    return m0 + amp * (-np.expm1(-k / 10**lk))


def sat_power(k, m0, amp, lk, beta):
    return m0 + amp * (-np.expm1(-beta * np.log1p(k / 10**lk)))


MODELS = {
    "Log-sigmoid": (hill, [.07, .48, 2.7, 2.6], ([0, 0, .1, -1], [.3, .7, 20, 6])),
    "Michaelis-Menten": (mm, [.07, .48, 2.6], ([0, 0, -1], [.3, .7, 6])),
    "Weibull": (weibull, [.07, .48, 2.6, .8], ([0, 0, -1, .03], [.3, .7, 6, 8])),
    "Richards": (richards, [.07, .48, 2.7, 2.6, 1], ([0, 0, .1, -1, .01], [.3, .7, 20, 6, 30])),
    "Exponential": (expo, [.07, .48, 2.6], ([0, 0, -1], [.3, .7, 6])),
    "Saturating power": (sat_power, [.07, .48, 2.6, 1], ([0, 0, -1, .03], [.3, .7, 6, 30])),
}


def fit(name, x, y):
    fn, initial, bounds = MODELS[name]
    best = None
    for shift in [-.5, 0, .5]:
        p = np.array(initial, float)
        p[3 if name in ("Log-sigmoid", "Richards") else 2] += shift
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            po, _ = curve_fit(fn, x, y, p0=p, bounds=bounds, maxfev=30000)
        sse = float(np.sum((fn(x, *po)-y)**2))
        if best is None or sse < best[0]:
            best = sse, po
    return best


def main():
    series, det, tags = [], [], []
    for s in SEEDS:
        ts = [f"b{int(k)}_s{s}" for k in KS[:-1]] + [f"g2u_s{s}"]
        ds = [json.loads((OUT/f"eval500_eval_{t}.json").read_text()) for t in ts]
        tags.append(ts); series.append([d["fixed_miou"] for d in ds]); det.append([d["det_auc_800"] for d in ds])
    arr, darr = np.array(series), np.array(det)
    y = arr.mean(0)
    result = {"budgets": KS.astype(int).tolist(), "seeds": SEEDS, "tags": tags,
              "loc_runs": arr.tolist(), "loc_mean": y.tolist(), "loc_sd": arr.std(0, ddof=1).tolist(),
              "det_mean": darr.mean(0).tolist(), "det_sd": darr.std(0, ddof=1).tolist(),
              "zero_convention": "exact k=0 boundary for every model",
              "ic_note": "conditional iid residual summaries on seven means; nested-budget dependence not modeled",
              "models": {}}
    for name, (fn, _, bounds) in MODELS.items():
        sse, p = fit(name, KS, y)
        folds = []
        for i, k in enumerate(KS):
            mask = np.arange(len(KS)) != i
            _, pi = fit(name, KS[mask], y[mask])
            pred = float(fn(np.array([k]), *pi)[0])
            folds.append({"k": int(k), "pred": pred, "observed": float(y[i]), "error": pred-float(y[i])})
        n, count = len(KS), len(p)+1
        deviance = n * (np.log(2*np.pi*sse/n)+1)
        ic = deviance+2*count+2*count*(count+1)/(n-count-1) if n > count+1 else None
        result["models"][name] = {"params": p.tolist(), "curve_parameters": len(p), "K_with_variance": count,
                                  "rmse": float(np.sqrt(sse/n)), "aicc": ic, "bic": float(deviance+count*np.log(n)),
                                  "loo_mae": float(np.mean([abs(f["error"]) for f in folds])), "folds": folds}
        print(name, result["models"][name]["loo_mae"], "AICc", ic, flush=True)
    p = np.array(result["models"]["Log-sigmoid"]["params"])
    m0, amp, g, lk = p
    measured = json.loads((OUT/"phase_law_validation.json").read_text())["measured"]
    checks = [{"k": int(k), "measured": v, "refit_prediction": float(hill(np.array([int(k)]), *p)[0])} for k,v in measured.items()]
    for c in checks:
        c["error_measured_minus_prediction"] = c["measured"] - c["refit_prediction"]
    _, p9 = fit("Log-sigmoid", np.r_[KS, [500,2000]], np.r_[y, [measured["500"],measured["2000"]]])
    result["log_sigmoid_summary"] = {"m0": m0, "m_inf": m0+amp, "gamma": g, "kstar": float(10**lk),
        "seed_kstar": {str(s): float(10**fit("Log-sigmoid",KS,arr[i])[1][3]) for i,s in enumerate(SEEDS)},
        "kstar_nine_point": float(10**p9[3]), "heldout_reassessment": checks,
        "heldout_note": "reassessment of previously collected points; not a new prospective validation",
        "efficiency": {str(f): {"training_masks": float(10**(lk+np.log(f/(1-f))/g)),
                                 "training_fraction": float(10**(lk+np.log(f/(1-f))/g)/14731),
                                 "with_validation_fraction": float((10**(lk+np.log(f/(1-f))/g)+1614)/16345)} for f in [.5,.8,.9]}}
    (OUT/"refit_final.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["log_sigmoid_summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
