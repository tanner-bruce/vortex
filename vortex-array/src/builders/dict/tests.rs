// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::alloc::Layout;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::LazyLock;

use allocator_api2::alloc::AllocError;
use allocator_api2::alloc::Allocator;
use allocator_api2::alloc::Global;
use parking_lot::Mutex;
use rstest::rstest;
use vortex_buffer::Alignment;
use vortex_buffer::BufferAllocatorRef;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use super::DictConstraints;
use super::UNCONSTRAINED;
use super::dict_encode;
use super::dict_encode_with_constraints;
use super::dict_encoder_in;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBinArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::dict::DictArraySlotsExt;
use crate::assert_arrays_eq;
use crate::dtype::PType;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

#[rstest]
#[case(0)]
#[case(1)]
#[case(255)]
#[case(256)]
#[case(257)]
#[case(65_535)]
#[case(65_536)]
#[case(65_537)]
fn one_shot_matches_unconstrained(#[case] len: usize) -> VortexResult<()> {
    let strings = (0..len).map(|i| format!("value-{i}")).collect::<Vec<_>>();
    let inputs = [
        VarBinViewArray::from_iter_str(&strings).into_array(),
        VarBinArray::from(strings).into_array(),
        PrimitiveArray::from_iter((0..len).map(|i| i as i64)).into_array(),
        PrimitiveArray::from_option_iter((0..len).map(|i| (i % 3 != 0).then_some(i as i64)))
            .into_array(),
    ];
    let mut ctx = SESSION.create_execution_ctx();
    for input in inputs {
        let baseline = dict_encode_with_constraints(&input, &UNCONSTRAINED, &mut ctx)?;
        let candidate = dict_encode(&input, &mut ctx)?;
        assert_eq!(candidate.len(), len);
        assert_eq!(candidate.dtype(), input.dtype());
        assert_arrays_eq!(candidate.codes(), baseline.codes(), &mut ctx);
        assert_arrays_eq!(candidate.values(), baseline.values(), &mut ctx);
        assert_arrays_eq!(candidate.into_array(), input, &mut ctx);
    }
    Ok(())
}

#[test]
fn nullable_slices_and_repeated_values() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    for values in [
        vec![None; 7],
        vec![Some(""), None, Some("red"), Some("red"), None, Some("")],
    ] {
        let input = values.into_iter().collect::<VarBinViewArray>().into_array();
        for range in [0..0, 0..input.len(), 1..input.len()] {
            let input = input.slice(range)?;
            let dict = dict_encode(&input, &mut ctx)?;
            assert_eq!(dict.len(), input.len());
            assert_arrays_eq!(dict.into_array(), input, &mut ctx);
        }
    }
    Ok(())
}

#[test]
fn constraints_and_stateful_encoder_are_unchanged() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let first = VarBinViewArray::from_iter_str(["first"]).into_array();
    let second = VarBinViewArray::from_iter_str(["second", "third", "first"]).into_array();
    let mut encoder = dict_encoder_in(&first, &UNCONSTRAINED, ctx.allocator().clone());
    assert_eq!(encoder.codes_ptype(), PType::U64);
    encoder.encode(&first, &mut ctx)?;
    assert_eq!(encoder.encode(&second, &mut ctx)?.len(), 3);
    assert_eq!(encoder.reset().len(), 3);
    for max_len in [0, 1, 2] {
        let constraints = DictConstraints {
            max_len,
            ..UNCONSTRAINED
        };
        let prefix = dict_encode_with_constraints(&second, &constraints, &mut ctx)?;
        assert_eq!(prefix.len(), max_len);
    }
    let constraints = DictConstraints {
        max_bytes: 0,
        ..UNCONSTRAINED
    };
    assert_eq!(
        dict_encode_with_constraints(&second, &constraints, &mut ctx)?.len(),
        0
    );
    Ok(())
}

#[test]
fn constrained_first_entry_honors_byte_and_length_limits() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let value = "first string larger than the positive byte limit";
    for input in [
        VarBinViewArray::from_iter_str([value]).into_array(),
        VarBinArray::from(vec![value]).into_array(),
        vec![None::<&str>]
            .into_iter()
            .collect::<VarBinViewArray>()
            .into_array(),
        PrimitiveArray::from_iter([1_u64]).into_array(),
        PrimitiveArray::from_option_iter([None::<u64>]).into_array(),
    ] {
        for constraints in [
            DictConstraints {
                max_bytes: 0,
                ..UNCONSTRAINED
            },
            DictConstraints {
                max_len: 0,
                ..UNCONSTRAINED
            },
            DictConstraints {
                max_bytes: 1,
                ..UNCONSTRAINED
            },
        ] {
            let encoded = dict_encode_with_constraints(&input, &constraints, &mut ctx)?;
            assert_eq!(encoded.len(), 0);
            assert_eq!(encoded.values().len(), 0);
            // The upstream stateful layout path retains its progress guarantee.
            let mut encoder = dict_encoder_in(&input, &constraints, ctx.allocator().clone());
            assert_eq!(encoder.encode(&input, &mut ctx)?.len(), 1);
        }
    }
    let input = VarBinViewArray::from_iter_str([value]).into_array();
    let entry_bytes = 16 + value.len();
    for (max_bytes, expected) in [(entry_bytes - 1, 0), (entry_bytes, 1)] {
        let constraints = DictConstraints {
            max_bytes,
            ..UNCONSTRAINED
        };
        assert_eq!(
            dict_encode_with_constraints(&input, &constraints, &mut ctx)?.len(),
            expected
        );
    }
    Ok(())
}

#[derive(Debug)]
struct RecordingAllocator(Arc<Mutex<Vec<usize>>>);

// SAFETY: all memory operations delegate to Global, with allocation sizes recorded separately.
unsafe impl Allocator for RecordingAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.0.lock().push(layout.size());
        Global.allocate(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: ptr and layout came from Global.
        unsafe { Global.deallocate(ptr, layout) }
    }
}

#[test]
fn one_shot_allocates_u32_working_codes() -> VortexResult<()> {
    let len = 65_537;
    let input = VarBinViewArray::from_iter_str(std::iter::repeat_n("same", len)).into_array();
    let sizes = Arc::new(Mutex::new(Vec::new()));
    let mut ctx = SESSION
        .create_execution_ctx()
        .with_allocator(BufferAllocatorRef::new(RecordingAllocator(Arc::clone(
            &sizes,
        ))));
    let dict = dict_encode(&input, &mut ctx)?;
    assert_eq!(dict.len(), len);
    let sizes = sizes.lock();
    let alignment_slack = Alignment::DEFAULT_ALIGNMENT.as_usize();
    assert!(sizes.contains(&(len * 4 + alignment_slack)));
    assert!(!sizes.contains(&(len * 8 + alignment_slack)));
    Ok(())
}
