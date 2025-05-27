#![allow(clippy::unused_unit)]
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use std::fmt::Write;
use serde::Deserialize;

#[polars_expr(output_type=String)]
fn pig_latinnify(inputs: &[Series]) -> PolarsResult<Series> {
    let ca: &StringChunked = inputs[0].str()?;
    let out: StringChunked = ca.apply_into_string_amortized(|value: &str, output: &mut String| {
        if let Some(first_char) = value.chars().next() {
            write!(output, "{}{}ay", &value[1..], first_char).unwrap()
        }
    });
    Ok(out.into_series())
}

#[derive(Deserialize)]
struct TripleBarrierKwargs {
    vertical_barrier_threshold: Option<usize>,
    min_return: Option<f64>
}

#[polars_expr(output_type=Int32)]
fn create_triple_barrier_labels(
    inputs: &[Series],
    kwargs: TripleBarrierKwargs
) -> PolarsResult<Series> {
    let price_series = inputs[0].f64()?.to_vec();
    let mut prices: Vec<f64> = Vec::with_capacity(price_series.len());

    for price in price_series {
        match price {
            Some(price) =>{
                prices.push(price);
                Ok(())
            },
            None => Err(PolarsError::NoData(
                "Price column contains nulls".into(),
            ))
        }?
    }

    let lower_barrier_thresholds = inputs[1].f64()?.to_vec();
    let upper_barrier_thresholds = inputs[2].f64()?.to_vec();

    let triple_barrier_labels = calculate_triple_barrier_labels(
        &prices,
        &lower_barrier_thresholds,
        &upper_barrier_thresholds,
        kwargs.vertical_barrier_threshold,
        kwargs.min_return
    );

    Ok(Series::from_vec("triple_barrier_labels".into(), triple_barrier_labels))
}

fn calculate_triple_barrier_labels(
    prices: &Vec<f64>,
    lower_barrier_thresholds: &[Option<f64>],
    upper_barrier_thresholds: &[Option<f64>],
    vertical_barrier_threshold: Option<usize>,
    min_return: Option<f64>
) -> Vec<i32> {
    // with_capacity() allocates space without need for reallocation
    let mut triple_barrier_labels: Vec<i32> = Vec::with_capacity(prices.len());

    // Adjust lower thresholds if minimum return is specified
    let lower_barrier_thresholds: Vec<Option<f64>> = match min_return {
        Some(min_return) => {
            lower_barrier_thresholds.iter().map(
                |lower_thresh| lower_thresh.as_ref().map(
                    |lower_thresh| lower_thresh.min(-min_return)
                )
            ).collect()
        },
        None => lower_barrier_thresholds.to_vec()
    };

    // Adjust upper thresholds if minimum return is specified
    let upper_barrier_thresholds: Vec<Option<f64>> = match min_return {
        Some(min_return) => {
            upper_barrier_thresholds.iter().map(
                |upper_thresh| upper_thresh.as_ref().map(
                    |upper_thresh| upper_thresh.max(min_return)
                )
            ).collect()
        },
        None => upper_barrier_thresholds.to_vec()
    };

    for i in 0..prices.len() {

        let price_path = match vertical_barrier_threshold {
            Some(vertical_barrier_threshold) => {
                compute_price_path_returns(&prices[i..(i+vertical_barrier_threshold).min(prices.len())])
            },
            None => compute_price_path_returns(&prices[i..])
        };
        
        triple_barrier_labels.push(
            compute_path_triple_barrier_label(
                &price_path,
                lower_barrier_thresholds[i],
                upper_barrier_thresholds[i],
                min_return
            )
        )
    }

    return triple_barrier_labels
}

fn compute_price_path_returns(
    price_path: &[f64]
) -> Vec<f64> {
    // Compute Returns over Price Path
    let first_val = price_path.first().unwrap();
    
    price_path.iter().map(|&current_val| (current_val/first_val) - 1.0).collect()
}

fn compute_path_triple_barrier_label(
    price_path_slice: &[f64],
    lower_barrier_threshold: Option<f64>,
    upper_barrier_threshold: Option<f64>,
    min_return: Option<f64>
) -> i32 {

    let mut found_lower_thresh: bool = false;
    let mut found_upper_thresh: bool = false;
    let path_returns: f64 = *price_path_slice.last().unwrap();

    for (_index, value) in price_path_slice.iter().enumerate() {
        // Check if lower barrier threshold is crossed
        if !found_lower_thresh {
            if let Some(lower_barrier_threshold) = lower_barrier_threshold {
                if *value < lower_barrier_threshold {
                    found_lower_thresh = true;
                }
            }
        };
        // Check if upper barrier threshold is crossed
        if !found_upper_thresh {
            if let Some(upper_barrier_threshold) = upper_barrier_threshold {
                if *value > upper_barrier_threshold {
                    found_upper_thresh = true;
                }
            }
        }

        if found_lower_thresh | found_upper_thresh {
            break;
        }
    }

    // Return the Label
    match (found_lower_thresh, found_upper_thresh) {
        (true, false) => -1,
        (false, true) => 1,
        (false, false) => {
            let min_return = match min_return {
                Some(min_return) => min_return,
                None => 0_f64
            };
            if path_returns > min_return {
                2
            } else if path_returns < -min_return {
                -2
            } else {
                0
            }
        },
        (true, true) => 42
    }
}