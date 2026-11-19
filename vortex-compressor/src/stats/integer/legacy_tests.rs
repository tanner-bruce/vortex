// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Differential checks using the pre-change counter and map construction.

use std::hash::Hash;
use std::sync::LazyLock;

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rustc_hash::FxBuildHasher;
use vortex_array::ArrayContext;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::NativeValue;
use vortex_array::dtype::IntegerPType;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_array::serde::SerializeOptions;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;

use super::DistinctCounter;
use super::DistinctInfo;
use super::DistinctValues;
use super::ErasedStats;
use super::TypedStats;
use super::typed_int_stats;
use crate::builtins::integer_dictionary_encode;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

struct LegacyDistinctInfo<T> {
    distinct_values: HashMap<NativeValue<T>, u32, FxBuildHasher>,
    distinct_count: u32,
    most_frequent_value: T,
    top_frequency: u32,
}

mod legacy {
    use std::hash::Hash;

    use rustc_hash::FxBuildHasher;
    use vortex_array::arrays::primitive::NativeValue;
    use vortex_array::dtype::IntegerPType;
    use vortex_error::VortexExpect;
    use vortex_utils::aliases::hash_map::HashMap;

    use super::LegacyDistinctInfo;

    pub(super) fn count<T: IntegerPType>(
        values: &[T],
        valid: &[bool],
        min: T,
        max: T,
    ) -> LegacyDistinctInfo<T>
    where
        NativeValue<T>: Eq + Hash,
    {
        let mut counter = DistinctCounter::new(min, max, values.len());
        for (&value, &valid) in values.iter().zip(valid) {
            if valid {
                counter.add(value, 1);
            }
        }
        counter.finish().unwrap()
    }

    /// Value ranges up to this many values may be counted in a dense array instead of a hash map.
    const DENSE_DISTINCT_MAX_RANGE: usize = 1 << 16;

    /// Value ranges up to this many values are always counted in a dense array, regardless of the array
    /// length. This covers every `u8` and `i8` array.
    const DENSE_DISTINCT_ALWAYS_RANGE: usize = 1 << 8;

    /// Accumulates the occurrences of each distinct valid value.
    enum DistinctCounter<T> {
        /// Distinct values are not being counted.
        #[expect(
            dead_code,
            reason = "preserve the legacy counter; this test uses counting"
        )]
        Off,
        /// Counts indexed by `value - min`, used when the value range is narrow. This avoids hashing
        /// entirely.
        Dense {
            /// The minimum valid value.
            min: T,
            /// `min` cast to `usize`. Casting preserves values modulo `2^usize::BITS`, so the wrapping
            /// difference of two cast values is the exact difference of any two values in the range.
            min_index: usize,
            /// The number of occurrences of `min + index`.
            counts: Vec<u32>,
        },
        /// Counts keyed by value.
        Hashed(HashMap<NativeValue<T>, u32, FxBuildHasher>),
    }

    impl<T: IntegerPType> DistinctCounter<T>
    where
        NativeValue<T>: Eq + Hash,
    {
        /// Chooses a dense counter when the valid values span a narrow range, otherwise a hash map.
        fn new(min: T, max: T, len: usize) -> Self {
            let range_len = match (min.to_i128(), max.to_i128()) {
                (Some(min), Some(max)) => usize::try_from(max - min + 1).unwrap_or(usize::MAX),
                _ => usize::MAX,
            };

            // A dense counter is only worthwhile when it is not much larger than the array itself.
            if range_len <= DENSE_DISTINCT_ALWAYS_RANGE
                || (range_len <= DENSE_DISTINCT_MAX_RANGE && range_len <= len)
            {
                Self::Dense {
                    min,
                    min_index: min.as_(),
                    counts: vec![0; range_len],
                }
            } else {
                Self::Hashed(HashMap::with_capacity_and_hasher(len / 2, FxBuildHasher))
            }
        }

        /// Records `count` occurrences of a valid value.
        #[allow(clippy::inline_always)]
        #[inline(always)]
        fn add(&mut self, value: T, count: u32) {
            match self {
                Self::Off => {}
                Self::Dense {
                    min_index, counts, ..
                } => counts[value.as_().wrapping_sub(*min_index)] += count,
                Self::Hashed(distinct_values) => {
                    *distinct_values.entry(NativeValue(value)).or_insert(0) += count
                }
            }
        }

        /// Returns the distinct value information, or `None` if counting was off.
        fn finish(self) -> Option<LegacyDistinctInfo<T>> {
            let distinct_values: HashMap<NativeValue<T>, u32, FxBuildHasher> = match self {
                Self::Off => return None,
                Self::Dense { min, counts, .. } => {
                    let min = min.to_i128().vortex_expect("integers fit in i128");
                    counts
                        .iter()
                        .enumerate()
                        .filter(|&(_, &count)| count > 0)
                        .map(|(index, &count)| {
                            let value = <T as num_traits::NumCast>::from(min + index as i128)
                                .vortex_expect("values between min and max fit in the type");
                            (NativeValue(value), count)
                        })
                        .collect()
                }
                Self::Hashed(distinct_values) => distinct_values,
            };

            let (&top_value, &top_count) = distinct_values
                .iter()
                .max_by_key(|&(_, &count)| count)
                .vortex_expect("we know this is non-empty");

            Some(LegacyDistinctInfo {
                distinct_count: u32::try_from(distinct_values.len())
                    .vortex_expect("there are more than `u32::MAX` distinct values"),
                most_frequent_value: top_value.0,
                top_frequency: top_count,
                distinct_values,
            })
        }
    }
}

fn compare<T>(
    values: Vec<T>,
    valid: &[bool],
    validity: Validity,
    project: impl for<'a> Fn(&'a mut ErasedStats) -> &'a mut TypedStats<T>,
) -> VortexResult<()>
where
    T: IntegerPType + for<'a> TryFrom<&'a Scalar, Error = VortexError>,
    PValue: From<T>,
    TypedStats<T>: Into<ErasedStats>,
    NativeValue<T>: Eq + Hash,
{
    let mut ctx = SESSION.create_execution_ctx();
    let array = PrimitiveArray::new(Buffer::copy_from(&values), validity);
    let mut actual = typed_int_stats::<T>(&array, true, &mut ctx)?;

    let valid_values = values
        .iter()
        .zip(valid)
        .filter_map(|(&value, &valid)| valid.then_some(value))
        .collect::<Vec<_>>();
    assert_eq!(actual.value_count() as usize, valid_values.len());
    assert_eq!(
        actual.null_count() as usize,
        values.len() - valid_values.len()
    );
    let (Some(&min), Some(&max)) = (valid_values.iter().min(), valid_values.iter().max()) else {
        assert_eq!(actual.average_run_length(), 0);
        if values.is_empty() {
            assert_eq!(actual.distinct_count(), None);
            assert_eq!(actual.erased().top_frequency(), None);
            assert!(project(&mut actual.erased).distinct().is_none());
        } else {
            assert_eq!(actual.distinct_count(), Some(0));
            assert_eq!(actual.erased().top_frequency(), Some(0));
            let distinct = project(&mut actual.erased).distinct().unwrap();
            assert_eq!(distinct.most_frequent_value(), T::zero());
            assert!(distinct.distinct_values().is_empty());
        }
        return Ok(());
    };
    let runs = 1 + valid_values
        .windows(2)
        .filter(|pair| pair[0] != pair[1])
        .count();
    assert_eq!(
        actual.average_run_length() as usize,
        valid_values.len() / runs
    );

    let expected = legacy::count(&values, valid, min, max);
    assert_eq!(actual.distinct_count(), Some(expected.distinct_count));
    assert_eq!(
        actual.erased().top_frequency(),
        Some(expected.top_frequency)
    );

    let typed = project(&mut actual.erased);
    assert_eq!(typed.min, min);
    assert_eq!(typed.max, max);
    let distinct = typed.distinct().unwrap();
    let actual_top = distinct.most_frequent_value();
    assert_eq!(actual_top, expected.most_frequent_value);
    assert!(
        distinct
            .distinct_values()
            .iter()
            .eq(expected.distinct_values.iter())
    );

    let retained_bytes = match &distinct.distinct_values {
        DistinctValues::Dense(pairs) => pairs.capacity() * size_of::<(NativeValue<T>, u32)>(),
        DistinctValues::Hashed(map) => map.allocation_size(),
    };
    assert!(retained_bytes <= expected.distinct_values.allocation_size());

    let actual_dict = integer_dictionary_encode(array.as_view(), &actual)?.into_array();
    project(&mut actual.erased).distinct = Some(DistinctInfo {
        distinct_values: DistinctValues::Hashed(expected.distinct_values),
        top_is_tied: false,
        most_frequent_value: expected.most_frequent_value,
        top_frequency: expected.top_frequency,
    });
    let expected_dict = integer_dictionary_encode(array.as_view(), &actual)?.into_array();
    let options = SerializeOptions::default();
    let array_ctx = ArrayContext::empty();
    assert_eq!(
        actual_dict.serialize(&array_ctx, &SESSION, &options)?,
        expected_dict.serialize(&array_ctx, &SESSION, &options)?,
    );

    Ok(())
}

fn compare_integer_type<T>(
    project: impl for<'a> Fn(&'a mut ErasedStats) -> &'a mut TypedStats<T> + Copy,
) -> VortexResult<()>
where
    T: IntegerPType + for<'a> TryFrom<&'a Scalar, Error = VortexError>,
    PValue: From<T>,
    TypedStats<T>: Into<ErasedStats>,
    NativeValue<T>: Eq + Hash,
{
    assert!(size_of::<DistinctInfo<T>>() <= size_of::<LegacyDistinctInfo<T>>());

    let mut rng = StdRng::seed_from_u64(812_823);
    let min = <T as num_traits::Bounded>::min_value().to_i128().unwrap();
    let max = <T as num_traits::Bounded>::max_value().to_i128().unwrap();
    let value = |v| <T as num_traits::NumCast>::from(v).unwrap();

    compare(Vec::new(), &[], Validity::NonNullable, project)?;
    for len in [1, 9, 63, 64, 65, 255, 511, 1_281] {
        let cases: [Vec<T>; 4] = [
            (0..len).map(|i| value(min + i % (max - min + 1))).collect(),
            (0..len)
                .map(|i| value(if i % 2 == 0 { min } else { min + 1 }))
                .collect(),
            (0..len)
                .map(|_| value(min + i128::from(rng.next_u32() % 37)))
                .collect(),
            (0..len)
                .map(|_| value(min + i128::from(rng.next_u64()) % (max - min + 1)))
                .collect(),
        ];
        for values in cases {
            for validity_kind in 0..5 {
                let valid: Vec<_> = (0..len)
                    .map(|i| match validity_kind {
                        0 | 1 => true,
                        2 => false,
                        3 => rng.next_u32() % 4 != 0,
                        // Retain both values in the two-way tie despite more than 90% nulls.
                        _ => i % 17 == 0,
                    })
                    .collect();
                let validity = match validity_kind {
                    0 => Validity::NonNullable,
                    1 => Validity::AllValid,
                    2 => Validity::AllInvalid,
                    _ => Validity::from_iter(valid.iter().copied()),
                };
                compare(values.clone(), &valid, validity, project)?;
            }
        }
    }
    Ok(())
}

macro_rules! legacy_case {
    ($name:ident, $variant:ident) => {
        #[test]
        fn $name() -> VortexResult<()> {
            compare_integer_type(|stats| {
                let ErasedStats::$variant(typed) = stats else {
                    unreachable!()
                };
                typed
            })
        }
    };
}

legacy_case!(matches_legacy_u8, U8);
legacy_case!(matches_legacy_u16, U16);
legacy_case!(matches_legacy_u32, U32);
legacy_case!(matches_legacy_u64, U64);
legacy_case!(matches_legacy_i8, I8);
legacy_case!(matches_legacy_i16, I16);
legacy_case!(matches_legacy_i32, I32);
legacy_case!(matches_legacy_i64, I64);

#[test]
fn matches_legacy_for_the_full_u16_tie() -> VortexResult<()> {
    let values: Vec<_> = (0..=u16::MAX).collect();
    let valid = vec![true; values.len()];
    compare(values, &valid, Validity::NonNullable, |stats| {
        let ErasedStats::U16(typed) = stats else {
            unreachable!()
        };
        typed
    })
}

#[test]
fn matches_legacy_across_the_dense_range_boundary() -> VortexResult<()> {
    const LIMIT: u32 = 65_536;
    let valid = vec![true; LIMIT as usize];
    for (max, dense) in [(LIMIT - 1, true), (LIMIT, false)] {
        let values: Vec<_> = (0..LIMIT - 1).chain([max]).collect();
        assert_eq!(
            matches!(
                DistinctCounter::new(0u32, max, values.len()),
                DistinctCounter::Dense { .. }
            ),
            dense,
        );
        compare(values, &valid, Validity::NonNullable, |stats| {
            let ErasedStats::U32(typed) = stats else {
                unreachable!()
            };
            typed
        })?;
    }
    Ok(())
}
