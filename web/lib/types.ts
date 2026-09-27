export type Segment = "Budget" | "Mid-range" | "Premium" | "Luxury";

export interface DiamondInput {
  carat: number;
  cut: string;
  color: string;
  clarity: string;
  depth: number;
  table: number;
  x: number;
  y: number;
  z: number;
}

export interface PredictResult {
  predicted_price: number;
  price_per_carat: number;
  predicted_segment: Segment;
  segment_probabilities: Record<Segment, number>;
  cluster: number;
  cluster_profile: {
    cluster: number;
    n: number;
    median_carat: number;
    median_price: number;
    median_price_per_carat: number;
    modal_cut: string;
    modal_color: string;
    modal_clarity: string;
  } | null;
  input: DiamondInput;
}

export interface ModelScore {
  model: string;
  mae: number;
  rmse: number;
  r2: number;
  cv_rmse_mean: number;
  cv_rmse_std: number;
}

export interface GradeEffect {
  worst_grade: string;
  best_grade: string;
  ratio_by_band: Record<string, number>;
  ratio_mean: number;
  ratio_largest_band: number;
}

export interface SegmentRow {
  segment: Segment;
  n: number;
  median_price: number;
  price_min: number;
  price_max: number;
  price_std: number;
  iqr: number;
  median_carat: number;
  median_ppc: number;
  best_median_cut: string;
  best_median_color: string;
  best_median_clarity: string;
}

export interface Summary {
  meta: { rows: number; seed: number };
  eda: {
    skew: { price: number; carat: number };
    carat_price_r_log: number;
    carat_price_r_raw: number;
    elasticity: number;
    price_per_carat: {
      median: number;
      p05: number;
      p95: number;
      max: number;
      skew: number;
    };
    correlations: Record<string, number>;
    grade_tables: Record<
      string,
      { order: string[]; median_price: number[]; count: number[] }
    >;
    size_controlled_grade_effect: {
      bands: number;
      cut: GradeEffect;
      color: GradeEffect;
      clarity: GradeEffect;
    };
  };
  regression: {
    best: ModelScore;
    best_model: string;
    linear: ModelScore;
    random_forest: ModelScore;
    gradient_boosting: ModelScore;
    coefficients: Record<string, number>;
    permutation: Record<string, number>;
    grouped: Record<string, number>;
    grouped_note: string;
  };
  classification: {
    best_model: string;
    accuracy: number;
    f1_weighted: number;
    logistic: { accuracy: number; f1_weighted: number };
    logistic_vs_rf: { accuracy: number; f1_weighted: number };
    confusion_matrix: number[][];
    per_segment: {
      segment: string;
      n: number;
      recall: number;
      precision: number;
    }[];
    imbalance_note: string;
  };
  clustering: {
    k: number;
    silhouette: number;
    silhouette_by_k: Record<string, number>;
    profiles: {
      cluster: number;
      n: number;
      median_carat: number;
      median_price: number;
      median_price_per_carat: number;
      modal_cut: string;
      modal_color: string;
      modal_clarity: string;
    }[];
    within_cluster_clarity: {
      cluster: number;
      n: number;
      clarity_min_grade: string;
      clarity_max_grade: string;
      clarity_min_ppc: number;
      clarity_max_ppc: number;
      clarity_spread_x: number;
    }[];
    modal_segment_agreement: number;
    interpretation: string;
  };
  segments: { segments: SegmentRow[] };
  feature_importance: { feature: string; importance: number }[];
}

/* --- the per-stone trace -------------------------------------------------
   Produced by trace.py, which runs one real stone through the fitted models
   and records what each stage actually computed. Every value below is
   measured, not narrated. */

export interface TraceDerived {
  value: number;
  formula: string;
  note: string;
}

export interface TraceNeighbour {
  carat: number;
  cut: string;
  color: string;
  clarity: string;
  price: number;
}

export interface Trace {
  input: DiamondInput;
  derived: {
    volume: TraceDerived;
    carat_per_volume: TraceDerived;
    log_carat: TraceDerived;
  };
  encoding: { feature: string; value: string; why: string }[];
  tree: {
    leaf_size: number;
    leaf_median_price: number;
    top_features: { feature: string; importance: number }[];
    neighbours: TraceNeighbour[];
  };
  result: {
    predicted_price: number;
    price_per_carat: number;
    lowest_quality_price: number;
    quality_premium: number;
    predicted_segment: Segment;
    segment_probabilities: Record<Segment, number>;
    cluster: number;
  };
  reality_check: {
    source_stone_price: number;
    market_median_price: number | null;
    comparable_stones: number;
    comparison_group: string;
    carat_tolerance: number | null;
    error_pct: number | null;
    price_spread: [number, number] | null;
  };
}
