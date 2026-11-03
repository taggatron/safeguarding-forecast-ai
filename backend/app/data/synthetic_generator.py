"""Realistic synthetic safeguarding dataset generator for South Devon College."""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from app.schemas.models import SyntheticGeneratorConfig


# Typical UK Further Education monthly teaching days (Sept to Aug)
BASE_FE_TEACHING_DAYS: Dict[int, int] = {
    9: 19,   # September: induction + Autumn term start
    10: 18,  # October: Autumn half-term week at end of month
    11: 21,  # November: full teaching month (peak disclosures)
    12: 14,  # December: ends ~18th Dec for winter break
    1: 19,   # January: Spring term start, post-holiday disclosures
    2: 15,   # February: shorter month + February half-term
    3: 20,   # March: full teaching month
    4: 13,   # April: two-week Easter break
    5: 17,   # May: bank holidays + May half-term + exams
    6: 18,   # June: final assessments and transition
    7: 6,    # July: wrap-up, summer school, apprenticeships continue
    8: 2,    # August: GCSE/Level 3 results days & enrolment, apprenticeships active
}

# Monthly seasonal multipliers by provision archetype
# Reflects UK FE safeguarding demand patterns (Autumn induction/Oct-Nov peak, Jan return peak, summer trough)
SEASONAL_MULTIPLIERS_DEFAULT: Dict[int, float] = {
    9: 1.12,   # Induction disclosures & initial needs assessments
    10: 1.36,  # Settling-in / pastoral identification peak
    11: 1.48,  # Highest teaching-day exposure + late autumn pressures
    12: 0.96,  # Pre-holiday anxieties balanced by fewer teaching days
    1: 1.38,   # Post-winter break disclosures & family/welfare pressures
    2: 1.16,   # Mid-winter pastoral demand
    3: 1.22,   # Pre-Easter & assessment build-up
    4: 0.82,   # Easter vacation reduces on-campus days
    5: 1.08,   # Exam stress & mental health referrals
    6: 0.86,   # Course completion & transition anxiety
    7: 0.34,   # Summer break (mainly holiday safeguarding duty & apprentices)
    8: 0.22,   # Summer break / results & enrolment weeks
}

APPRENTICESHIP_SEASONAL_MULTIPLIERS: Dict[int, float] = {
    9: 1.08,
    10: 1.18,
    11: 1.22,
    12: 0.92,
    1: 1.18,
    2: 1.06,
    3: 1.10,
    4: 0.94,
    5: 1.04,
    6: 0.98,
    7: 0.72,   # Apprentices remain in workplace during summer
    8: 0.58,
}


def get_academic_year_label(year: int, month: int, start_month: int = 9) -> str:
    """Return UK academic year label such as '2024/25'."""
    if month >= start_month:
        ay_start = year
    else:
        ay_start = year - 1
    ay_end_short = str(ay_start + 1)[-2:]
    return f"{ay_start}/{ay_end_short}"


def get_academic_term(month: int) -> str:
    """Return UK Further Education academic term name for a calendar month."""
    if month in (9, 10, 11, 12):
        return "Autumn Term"
    if month in (1, 2, 3):
        return "Spring Term"
    if month in (4, 5, 6):
        return "Summer Term"
    return "Summer Period"


def generate_synthetic_safeguarding_dataset(
    config: Optional[SyntheticGeneratorConfig] = None,
) -> pd.DataFrame:
    """
    Generate at least six full academic years of realistic, reproducible synthetic
    monthly safeguarding data across provisions and broad safeguarding categories.

    Contains genuine stochastic overdispersion, enrolment dynamics, teaching-day
    effects, reporting-system step change, and occasional high-volume outlier months
    so forecasting models face realistic operational uncertainty.
    """
    if config is None:
        config = SyntheticGeneratorConfig()

    rng = np.random.default_rng(config.seed)

    provisions = config.provisions or ["EPYP", "Adult", "Apprenticeships"]
    categories = config.categories or [
        "Mental health and wellbeing concerns",
        "Abuse or neglect",
        "Exploitation",
        "Online safety",
        "Other recorded concerns",
    ]

    # Provision weights for student enrolment and baseline safeguarding incidence
    default_prov_enrolment_shares = {
        "EPYP": 0.46,              # 16-18 Study Programmes (~3,450 students)
        "Adult": 0.34,             # 19+ Adult Learners (~2,550 students)
        "Apprenticeships": 0.20,   # Apprentices (~1,500 apprentices)
    }
    default_prov_concern_shares = {
        "EPYP": 0.61,              # Higher safeguarding volume among 16-18 cohort
        "Adult": 0.25,             # Moderate volume among 19+ learners
        "Apprenticeships": 0.14,   # Work-based pastoral & welfare concerns
    }

    # Normalize shares for any custom provision list
    prov_enrol_weights = np.array(
        [default_prov_enrolment_shares.get(p, 1.0 / len(provisions)) for p in provisions],
        dtype=float,
    )
    prov_enrol_weights /= prov_enrol_weights.sum()

    prov_concern_weights = np.array(
        [default_prov_concern_shares.get(p, 1.0 / len(provisions)) for p in provisions],
        dtype=float,
    )
    prov_concern_weights /= prov_concern_weights.sum()

    # Category baseline shares
    default_cat_shares = {
        "Mental health and wellbeing concerns": 0.43,
        "Abuse or neglect": 0.20,
        "Online safety": 0.13,
        "Exploitation": 0.09,
        "Other recorded concerns": 0.15,
    }
    cat_weights = np.array(
        [default_cat_shares.get(c, 1.0 / len(categories)) for c in categories],
        dtype=float,
    )
    cat_weights /= cat_weights.sum()

    start_year = config.start_academic_year
    start_month = config.academic_year_start_month
    total_months = config.num_academic_years * 12

    dates = pd.date_range(
        start=f"{start_year:04d}-{start_month:02d}-01",
        periods=total_months,
        freq="MS",
    )

    # Select 4 occasional unusually high-volume shock months across the historical timeline
    # (avoiding summer months so they reflect term-time community/online/exam shocks)
    eligible_shock_indices = [
        idx for idx, dt in enumerate(dates) if dt.month not in (7, 8) and idx > 4
    ]
    shock_indices = set(
        rng.choice(eligible_shock_indices, size=min(4, len(eligible_shock_indices)), replace=False).tolist()
    )

    rows: List[Dict] = []
    base_college_enrolment = 7200
    base_monthly_college_concerns = config.base_annual_volume / 12.0

    for t_idx, dt in enumerate(dates):
        year = dt.year
        month = dt.month
        period_str = dt.strftime("%Y-%m-%d")
        ay_idx = t_idx // 12
        ay_label = get_academic_year_label(year, month, start_month)
        term_label = get_academic_term(month)

        # Teaching days with small calendar jitter (+/- 1 day in Mar/Apr depending on Easter timing)
        base_td = BASE_FE_TEACHING_DAYS.get(month, 17)
        if month in (3, 4, 10, 2):
            td_jitter = int(rng.choice([-1, 0, 0, 1]))
        else:
            td_jitter = 0
        teaching_days = max(1, base_td + td_jitter)

        # Reporting system / mandatory staff awareness upgrade flag from academic year 3 onwards (e.g. Sep 2023)
        reporting_change_flag = 1 if (config.include_reporting_change and ay_idx >= 3) else 0
        reporting_multiplier = 1.135 if reporting_change_flag == 1 else 1.0

        # Secular underlying trend (~2.6% per academic year)
        secular_trend = (1.0 + 0.026) ** (t_idx / 12.0)

        # Enrolment growth (~1.8% per academic year) + gentle intra-year retention curve
        months_into_ay = t_idx % 12
        retention_factor = 1.0 - (0.0045 * months_into_ay) if month not in (7, 8) else 0.94
        college_enrolment_t = base_college_enrolment * ((1.018) ** ay_idx) * retention_factor

        # Month-level autoregressive / macro disturbance
        is_shock_month = t_idx in shock_indices
        shock_multiplier = float(rng.uniform(1.19, 1.31)) if is_shock_month else 1.0
        macro_month_noise = float(rng.normal(1.0, 0.055))
        macro_month_noise = float(np.clip(macro_month_noise, 0.84, 1.18))

        for p_idx, prov in enumerate(provisions):
            prov_enrolment = int(
                round(college_enrolment_t * prov_enrol_weights[p_idx] * float(rng.normal(1.0, 0.008)))
            )

            # Provision-specific seasonal multiplier
            if "apprentice" in prov.lower():
                seas_mult = APPRENTICESHIP_SEASONAL_MULTIPLIERS.get(month, 1.0)
                base_att = 91.5
            elif "adult" in prov.lower():
                seas_mult = 0.88 * SEASONAL_MULTIPLIERS_DEFAULT.get(month, 1.0) + 0.12
                base_att = 87.2
            else:
                # EPYP / 16-18
                seas_mult = SEASONAL_MULTIPLIERS_DEFAULT.get(month, 1.0)
                base_att = 89.4

            # Teaching day elasticity relative to typical active month (18 days)
            td_ratio = (teaching_days / 18.0) ** 0.35 if month not in (7, 8) else 1.0

            # Realistic attendance rate (%) with winter dip in Dec/Jan/Feb
            winter_att_dip = -2.1 if month in (12, 1, 2) else (1.1 if month in (9, 10) else 0.0)
            attendance_rate = round(
                float(np.clip(base_att + winter_att_dip + rng.normal(0, 0.9), 78.0, 97.5)),
                1,
            )

            # Lower attendance slightly correlates with higher pastoral/welfare concern identification
            att_effect = 1.0 + max(-0.06, min(0.08, (89.0 - attendance_rate) * 0.012))

            prov_expected_total = (
                base_monthly_college_concerns
                * prov_concern_weights[p_idx]
                * seas_mult
                * td_ratio
                * secular_trend
                * reporting_multiplier
                * att_effect
                * shock_multiplier
                * macro_month_noise
            )

            for c_idx, cat in enumerate(categories):
                cat_share = cat_weights[c_idx]
                # Mild category-specific seasonal modulation
                cat_lower = cat.lower()
                cat_mod = 1.0
                if "mental health" in cat_lower:
                    # Slight upward secular share & exam/winter pressures
                    cat_mod = (1.0 + 0.012 * ay_idx) * (1.08 if month in (11, 1, 5) else 0.96)
                elif "online" in cat_lower:
                    cat_mod = (1.0 + 0.018 * ay_idx) * (1.12 if month in (10, 11, 1, 2) else 0.93)
                elif "abuse" in cat_lower or "neglect" in cat_lower:
                    # Post-holiday disclosure bump in Sept and Jan
                    cat_mod = 1.14 if month in (9, 10, 1) else 0.96

                mu = max(0.3, prov_expected_total * cat_share * cat_mod)

                # Negative Binomial overdispersion (var = mu + alpha * mu^2)
                alpha_dispersion = 0.095
                n_param = 1.0 / alpha_dispersion
                p_param = n_param / (n_param + mu)
                sampled_count = int(rng.negative_binomial(n_param, p_param))

                rows.append(
                    {
                        "period_start": period_str,
                        "academic_year": ay_label,
                        "academic_term": term_label,
                        "provision": prov,
                        "safeguarding_category": cat,
                        "concern_count": max(0, sampled_count),
                        "enrolled_students": prov_enrolment,
                        "teaching_days": teaching_days,
                        "attendance_rate": attendance_rate,
                        "reporting_change_flag": reporting_change_flag,
                    }
                )

    df = pd.DataFrame(rows)
    return df

