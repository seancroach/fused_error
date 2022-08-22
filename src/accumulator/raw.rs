use crate::CompoundError;

use std::fmt::{self, Debug};

/// A raw error accumulator, with a minimal API for handling memory/layout
/// state.
#[must_use = "accumulators will panic on drop if not handled"]
pub(crate) struct Accumulator<E> {
    // This is `pub` for low-level access by result::raw::CompoundResult.
    pub errors: Option<Vec<E>>,
}

impl<E> Accumulator<E> {
    /// Creates an empty accumulator.
    #[inline]
    pub fn new() -> Self {
        Accumulator::from_vec(Vec::new())
    }

    /// Creates a new accumulator from a vector.
    #[inline]
    pub fn from_vec(vec: Vec<E>) -> Self {
        Accumulator { errors: Some(vec) }
    }

    /// Asserts that this accumulator isn't being accessed after getting
    /// handled.
    ///
    /// # Panics
    ///
    /// Panics if the accumulator's errors are `None`.
    #[inline]
    #[track_caller]
    fn check(&self) {
        assert!(
            !(cfg!(debug_assertions) && self.is_handled()),
            "cannot access accumulator after it's been handled"
        );
    }

    /// Returns a reference to the internal vector.
    ///
    /// # Panics
    ///
    /// Panics if the accumulator's errors are `None`.
    #[must_use]
    #[inline]
    #[track_caller]
    pub fn as_ref(&self) -> &Vec<E> {
        self.check();
        // SAFETY: previously asserted via `check()`
        unsafe { self.errors.as_ref().unwrap_unchecked() }
    }

    /// Returns a mutable reference to the internal vector.
    ///
    /// # Panics
    ///
    /// Panics if the accumulator's errors are `None`.
    #[must_use]
    #[inline]
    #[track_caller]
    pub fn as_mut(&mut self) -> &mut Vec<E> {
        self.check();
        // SAFETY: previously asserted via `check()`
        unsafe { self.errors.as_mut().unwrap_unchecked() }
    }

    /// Appends an error to this accumulator.
    ///
    /// # Panics
    ///
    /// Panics if the accumulator's errors are `None`.
    #[inline]
    #[track_caller]
    pub fn push(&mut self, err: E) {
        self.as_mut().push(err);
    }

    /// Returns the accumulated errors, leaving `None` in its place.
    ///
    /// # Safety
    ///
    /// This method semantically moves out the contained errors without
    /// preventing further usage. It is your responsibility to ensure that this
    /// accumulator is not used again.
    ///
    /// # Panics
    ///
    /// Panics if the accumulator's errors are `None`.
    #[must_use]
    #[inline]
    #[track_caller]
    pub unsafe fn take(&mut self) -> Vec<E> {
        self.check();
        // SAFETY: previously asserted via `check()`
        self.errors.take().unwrap_unchecked()
    }

    /// Returns the reduced compound error, leaving `None` in its place.
    ///
    /// # Safety
    ///
    /// This method semantically moves out the contained errors without
    /// preventing further usage. It is your responsibility to ensure that this
    /// accumulator is not used again.
    ///
    /// # Panics
    ///
    /// Panics if the accumulator's errors are `None`.
    #[must_use]
    #[inline]
    #[track_caller]
    pub unsafe fn reduce(&mut self) -> Option<E>
    where
        E: CompoundError,
    {
        // SAFETY: safety is ensured by the caller
        self.take().into_iter().reduce(CompoundError::merge)
    }

    /// Discards the collected errors, leaving `None` in its place.
    ///
    /// # Safety
    ///
    /// This is often a semantic error, as error suppression is often not the
    /// intention when using an accumulator.
    ///
    /// This method semantically discards the contained errors without
    /// preventing further usage. It is your responsibility to ensure that this
    /// accumulator is not used again.
    ///
    /// # Panics
    ///
    /// Panics if the accumulator's errors are `None`.
    #[inline]
    #[track_caller]
    pub unsafe fn ignore(&mut self) {
        self.check();
        self.errors = None;
    }

    /// Returns `true` if this accumulator has been handled.
    #[must_use]
    #[inline]
    pub fn is_handled(&self) -> bool {
        self.errors.is_none()
    }
}

impl<E> Debug for Accumulator<E>
where
    E: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.errors {
            Some(ref errors) => Debug::fmt(errors, f),
            None => f.write_str("Accumulator(None)"),
        }
    }
}

impl<E> Drop for Accumulator<E> {
    fn drop(&mut self) {
        if cfg!(debug_assertions) {
            if let Some(ref errors) = self.errors {
                let len = errors.len();

                match len {
                    0 => panic!("compound_error::Accumulator dropped without getting handled"),
                    n => panic!(
                        "compound_error::Accumulator dropped with {n} unhandled error{}",
                        if n == 1 { "" } else { "s" },
                    ),
                }
            }
        }
    }
}
