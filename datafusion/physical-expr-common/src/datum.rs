// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use arrow::array::BooleanArray;
use arrow::array::{ArrayRef, Datum, make_comparator};
use arrow::buffer::{BooleanBuffer, NullBuffer};
use arrow::compute::kernels::boolean::{and, not, or};
use arrow::compute::kernels::cmp::{
    distinct, eq, gt, gt_eq, lt, lt_eq, neq, not_distinct,
};
use arrow::compute::{SortOptions, ilike, like, nilike, nlike};
use arrow::error::ArrowError;
use datafusion_common::utils::{canonicalize_float_scalar, canonicalize_floats};
use datafusion_common::{Result, ScalarValue};
use datafusion_common::{arrow_datafusion_err, assert_or_internal_err, internal_err};
use datafusion_expr_common::columnar_value::ColumnarValue;
use datafusion_expr_common::operator::Operator;
use std::sync::Arc;

/// Applies a binary [`Datum`] kernel `f` to `lhs` and `rhs`
///
/// This maps arrow-rs' [`Datum`] kernels to DataFusion's [`ColumnarValue`] abstraction
pub fn apply(
    lhs: &ColumnarValue,
    rhs: &ColumnarValue,
    f: impl Fn(&dyn Datum, &dyn Datum) -> Result<ArrayRef, ArrowError>,
) -> Result<ColumnarValue> {
    match (&lhs, &rhs) {
        (ColumnarValue::Array(left), ColumnarValue::Array(right)) => {
            Ok(ColumnarValue::Array(f(&left.as_ref(), &right.as_ref())?))
        }
        (ColumnarValue::Scalar(left), ColumnarValue::Array(right)) => Ok(
            ColumnarValue::Array(f(&left.to_scalar()?, &right.as_ref())?),
        ),
        (ColumnarValue::Array(left), ColumnarValue::Scalar(right)) => Ok(
            ColumnarValue::Array(f(&left.as_ref(), &right.to_scalar()?)?),
        ),
        (ColumnarValue::Scalar(left), ColumnarValue::Scalar(right)) => {
            let array = f(&left.to_scalar()?, &right.to_scalar()?)?;
            let scalar = ScalarValue::try_from_array(array.as_ref(), 0)?;
            Ok(ColumnarValue::Scalar(scalar))
        }
    }
}

/// Applies a binary [`Datum`] comparison operator `op` to `lhs` and `rhs`
pub fn apply_cmp(
    op: Operator,
    lhs: &ColumnarValue,
    rhs: &ColumnarValue,
) -> Result<ColumnarValue> {
    if lhs.data_type().is_nested() {
        apply_cmp_for_nested(op, lhs, rhs)
    } else {
        let f = match op {
            Operator::Eq => eq,
            Operator::NotEq => neq,
            Operator::Lt => lt,
            Operator::LtEq => lt_eq,
            Operator::Gt => gt,
            Operator::GtEq => gt_eq,
            Operator::IsDistinctFrom => distinct,
            Operator::IsNotDistinctFrom => not_distinct,

            Operator::LikeMatch => like,
            Operator::ILikeMatch => ilike,
            Operator::NotLikeMatch => nlike,
            Operator::NotILikeMatch => nilike,

            _ => {
                return internal_err!("Invalid compare operator: {}", op);
            }
        };

        // Arrow's comparison kernels use IEEE 754 totalOrder semantics for
        // floats, which treats `-0.0` and `+0.0` as distinct and orders NaNs
        // relative to other values. Canonicalize float operands so
        // `+0.0 == -0.0` holds and `IS [NOT] DISTINCT FROM` sees every NaN
        // as one value (SQL grouping equality). No-op for non-float types.
        let lhs = canonicalize_cmp_input(lhs);
        let rhs = canonicalize_cmp_input(rhs);
        if is_ieee_cmp(op) && lhs.data_type().is_floating() {
            // The ordering comparison operators follow IEEE 754 rather than
            // totalOrder semantics: any comparison involving NaN is false,
            // except `!=` which is true.
            apply(&lhs, &rhs, |l, r| {
                let result = f(l, r)?;
                ieee_nan_cmp_fixup(op, l, r, result)
            })
        } else {
            apply(&lhs, &rhs, |l, r| Ok(Arc::new(f(l, r)?)))
        }
    }
}

fn is_ieee_cmp(op: Operator) -> bool {
    matches!(
        op,
        Operator::Eq
            | Operator::NotEq
            | Operator::Lt
            | Operator::LtEq
            | Operator::Gt
            | Operator::GtEq
    )
}

fn canonicalize_cmp_input(cv: &ColumnarValue) -> ColumnarValue {
    match cv {
        ColumnarValue::Array(a) => ColumnarValue::Array(canonicalize_floats(a)),
        ColumnarValue::Scalar(s) => {
            ColumnarValue::Scalar(canonicalize_float_scalar(s.clone()))
        }
    }
}

/// Overrides `result` rows where either float operand is NaN with the IEEE 754
/// comparison outcome: false for every operator except `!=`, which is true.
/// Null rows stay null. The common case - no NaN present - returns `result`
/// unchanged.
fn ieee_nan_cmp_fixup(
    op: Operator,
    lhs: &dyn Datum,
    rhs: &dyn Datum,
    result: BooleanArray,
) -> Result<ArrayRef, ArrowError> {
    let num_rows = result.len();
    let mask = match (nan_mask(lhs, num_rows), nan_mask(rhs, num_rows)) {
        (None, None) => return Ok(Arc::new(result)),
        (Some(mask), None) | (None, Some(mask)) => mask,
        (Some(l), Some(r)) => or(&l, &r)?,
    };
    let fixed = if op == Operator::NotEq {
        or(&result, &mask)?
    } else {
        and(&result, &not(&mask)?)?
    };
    Ok(Arc::new(fixed))
}

/// Returns a per-row mask of NaN values in a float [`Datum`], broadcast to
/// `num_rows` for scalars, or `None` if the datum contains no NaN (including
/// all non-float types).
fn nan_mask(datum: &dyn Datum, num_rows: usize) -> Option<BooleanArray> {
    use arrow::array::{Array, AsArray};
    use arrow::datatypes::{DataType, Float16Type, Float32Type, Float64Type};

    fn mask<T: arrow::datatypes::ArrowPrimitiveType>(
        array: &arrow::array::PrimitiveArray<T>,
        is_scalar: bool,
        num_rows: usize,
        is_nan: impl Fn(T::Native) -> bool,
    ) -> Option<BooleanArray> {
        if is_scalar {
            // A null scalar makes every result row null, so no fixup is
            // needed; a NaN scalar makes every row a NaN comparison.
            (array.null_count() == 0 && is_nan(array.value(0)))
                .then(|| BooleanArray::new(BooleanBuffer::new_set(num_rows), None))
        } else {
            // Values at null positions may hold arbitrary bits, but those
            // result rows are already null and the boolean kernels keep them
            // null.
            array
                .values()
                .iter()
                .any(|v| is_nan(*v))
                .then(|| BooleanArray::from_unary(array, &is_nan))
        }
    }

    let (array, is_scalar) = datum.get();
    match array.data_type() {
        DataType::Float16 => mask(
            array.as_primitive::<Float16Type>(),
            is_scalar,
            num_rows,
            |v| v.is_nan(),
        ),
        DataType::Float32 => mask(
            array.as_primitive::<Float32Type>(),
            is_scalar,
            num_rows,
            |v| v.is_nan(),
        ),
        DataType::Float64 => mask(
            array.as_primitive::<Float64Type>(),
            is_scalar,
            num_rows,
            |v| v.is_nan(),
        ),
        _ => None,
    }
}

/// Applies a binary [`Datum`] comparison operator `op` to `lhs` and `rhs` for nested type like
/// List, FixedSizeList, LargeList, Struct, Union, Map, or a dictionary of a nested type
pub fn apply_cmp_for_nested(
    op: Operator,
    lhs: &ColumnarValue,
    rhs: &ColumnarValue,
) -> Result<ColumnarValue> {
    let left_data_type = lhs.data_type();
    let right_data_type = rhs.data_type();

    assert_or_internal_err!(
        matches!(
            op,
            Operator::Eq
                | Operator::NotEq
                | Operator::Lt
                | Operator::Gt
                | Operator::LtEq
                | Operator::GtEq
                | Operator::IsDistinctFrom
                | Operator::IsNotDistinctFrom
        ) && left_data_type.equals_datatype(&right_data_type),
        "invalid operator or data type mismatch for nested data, op {op} left {left_data_type}, right {right_data_type}",
    );

    apply(lhs, rhs, |l, r| {
        Ok(Arc::new(compare_op_for_nested(op, l, r)?))
    })
}

/// Compare with eq with either nested or non-nested
pub fn compare_with_eq(
    lhs: &dyn Datum,
    rhs: &dyn Datum,
    is_nested: bool,
) -> Result<BooleanArray> {
    if is_nested {
        compare_op_for_nested(Operator::Eq, lhs, rhs)
    } else {
        eq(lhs, rhs).map_err(|e| arrow_datafusion_err!(e))
    }
}

/// Compare on nested type List, Struct, and so on
pub fn compare_op_for_nested(
    op: Operator,
    lhs: &dyn Datum,
    rhs: &dyn Datum,
) -> Result<BooleanArray> {
    let (l, is_l_scalar) = lhs.get();
    let (r, is_r_scalar) = rhs.get();
    let l_len = l.len();
    let r_len = r.len();

    assert_or_internal_err!(l_len == r_len || is_l_scalar || is_r_scalar, "len mismatch");

    let len = match is_l_scalar {
        true => r_len,
        false => l_len,
    };

    // fast path, if compare with one null and operator is not 'distinct', then we can return null array directly
    if !matches!(op, Operator::IsDistinctFrom | Operator::IsNotDistinctFrom)
        && (is_l_scalar && l.null_count() == 1 || is_r_scalar && r.null_count() == 1)
    {
        return Ok(BooleanArray::new_null(len));
    }

    // TODO: make SortOptions configurable
    // we choose the default behaviour from arrow-rs which has null-first that follow spark's behaviour
    let cmp = make_comparator(l, r, SortOptions::default())?;

    let cmp_with_op = |i, j| match op {
        Operator::Eq | Operator::IsNotDistinctFrom => cmp(i, j).is_eq(),
        Operator::Lt => cmp(i, j).is_lt(),
        Operator::Gt => cmp(i, j).is_gt(),
        Operator::LtEq => !cmp(i, j).is_gt(),
        Operator::GtEq => !cmp(i, j).is_lt(),
        Operator::NotEq | Operator::IsDistinctFrom => !cmp(i, j).is_eq(),
        _ => unreachable!("unexpected operator found"),
    };

    let values = match (is_l_scalar, is_r_scalar) {
        (false, false) => BooleanBuffer::collect_bool(len, |i| cmp_with_op(i, i)),
        (true, false) => BooleanBuffer::collect_bool(len, |i| cmp_with_op(0, i)),
        (false, true) => BooleanBuffer::collect_bool(len, |i| cmp_with_op(i, 0)),
        (true, true) => std::iter::once(cmp_with_op(0, 0)).collect(),
    };

    // Distinct understand how to compare with NULL
    // i.e NULL is distinct from NULL -> false
    if matches!(op, Operator::IsDistinctFrom | Operator::IsNotDistinctFrom) {
        Ok(BooleanArray::new(values, None))
    } else {
        // If one of the side is NULL, we return NULL
        // i.e. NULL eq NULL -> NULL
        // For nested comparisons, we need to ensure the null buffer matches the result length
        let nulls = match (is_l_scalar, is_r_scalar) {
            (false, false) | (true, true) => NullBuffer::union(l.nulls(), r.nulls()),
            (true, false) => {
                // When left is null-scalar and right is array, expand left nulls to match result length
                match l.nulls().filter(|nulls| nulls.is_null(0)) {
                    Some(_) => Some(NullBuffer::new_null(len)), // Left scalar is null
                    None => r.nulls().cloned(),                 // Left scalar is non-null
                }
            }
            (false, true) => {
                // When right is null-scalar and left is array, expand right nulls to match result length
                match r.nulls().filter(|nulls| nulls.is_null(0)) {
                    Some(_) => Some(NullBuffer::new_null(len)), // Right scalar is null
                    None => l.nulls().cloned(), // Right scalar is non-null
                }
            }
        };
        Ok(BooleanArray::new(values, nulls))
    }
}
