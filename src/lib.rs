//! A simple library for working with composable errors.
//!
//! # Examples
//!
//! Basic usage:
//!
//! ```
//! use std::{
//!     num::{IntErrorKind, ParseIntError},
//!     iter::Sum,
//!     str::FromStr
//! };
//!
//! use fused_error::{Accumulator, FusedResult, IteratorExt};
//!
//! /// Take an iterator of textual data, adding up all of the parsed numbers.
//! ///
//! /// Unlike the standard way of returning a `Result<N, N::Err>`, this doesn't
//! /// short-circuit, it keeps track of the current sum, and reports any
//! /// further diagnostics past the first failure.
//! fn calculate_sum<N, E, I>(iter: I) -> FusedResult<N, N::Err>
//! where
//!     N: FromStr + Sum,
//!     E: AsRef<str>,
//!     I: IntoIterator<Item = E>,
//! {
//!     // Error accumulators collect errors to defer handling them, providing
//!     // more holistic diagnostics:
//!     let mut acc = Accumulator::new();
//!     let sum = iter
//!         .into_iter()
//!         .map(|item| item.as_ref().parse::<N>())
//!         // `fused_error` adds certain methods to iterators; no more
//!         // disrupting iterator chains and `collect` hells for results!
//!         .accumulate(&mut acc)
//!         .sum();
//!     // fused results let you easily pass around error accumulators and
//!     // are perfect for cases where a yielded "ok" value and an error case
//!     // aren't mutually exclusive.
//!     FusedResult::new(sum, acc)
//! }
//!
//! let result: FusedResult<i32, _> = calculate_sum(["1", "2", "3", "4"]);
//! assert_eq!(result.value(), &10);
//! assert_eq!(result.errors(), []);
//!
//! let result: FusedResult<i8, _> = calculate_sum(["", "-129", "foo", "128"]);
//! assert_eq!(result.value(), &0);
//! assert_eq!(
//!     result
//!         .errors()
//!         .into_iter()
//!         .map(|err| err.kind().clone())
//!         .collect::<Vec<_>>(),
//!     [
//!         IntErrorKind::Empty,
//!         IntErrorKind::NegOverflow,
//!         IntErrorKind::InvalidDigit,
//!         IntErrorKind::PosOverflow,
//!     ],
//! );
//!
//! let result: FusedResult<u8, _> = calculate_sum(["-1", "", "0", "1"]);
//! assert_eq!(result.value(), &1);
//! assert_eq!(
//!     result
//!         .errors()
//!         .into_iter()
//!         .map(|err| err.kind().clone())
//!         .collect::<Vec<_>>(),
//!     [IntErrorKind::InvalidDigit, IntErrorKind::Empty],
//! );
//! ```
//!
//! # Features
//!
//! So far, there is only one opt-in feature: `syn`. Enabling this feature
//! implements [`FusedError`] on [`syn::Error`], as that was one of the main
//! motivations for creating this library.
//!
//! # Motivation
//!
//! [`syn`] already has great composable errors that you combine with
//! [`syn::Error::combine`]. Also, [`darling`] has a great system that was the
//! primary inspiration for error accumulators and their drop mechanic. However,
//! of course, [`darling`]'s accumulators are only to be used with [`darling`]
//! errors and their accumulator API is far more limited to reflect this.
//!
//! The original use case for this crate, deferring and collecting multiple
//! errors, is primarily helpful in parsing: the more diagnostics you can
//! provide in one pass limits the need of frequently changing something,
//! building, fixing the one error, and trying again.
//!
//! [`darling`]: https://docs.rs/darling/latest/darling/

// fused_error types in rustdoc of other crates get linked to here
#![doc(html_root_url = "https://docs.rs/fused_error/0.1.2")]
#![cfg_attr(doc_cfg, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(
    clippy::mismatching_type_param_order,
    clippy::missing_errors_doc,
    clippy::module_name_repetitions
)]

pub mod accumulated;
pub mod accumulator;
pub mod iter;
pub mod result;

#[doc(inline)]
pub use accumulated::Accumulated;
#[doc(inline)]
pub use accumulator::Accumulator;
#[doc(inline)]
pub use iter::IteratorExt;
#[doc(inline)]
pub use result::{FusedResult, PackedResult};

/// Interface for error types that can store multiple error messages within one
/// instance.
///
/// Instead of making your own newtype to implement `FusedError` on a remote
/// error type, consider using [`Accumulated`] instead.
///
/// # Examples
///
/// ```
/// use fused_error::{Accumulated, FusedError};
///
/// struct Error<'a> {
///     messages: Vec<&'a str>,
/// }
///
/// impl FusedError for Error<'_> {
///     fn combine(&mut self, mut other: Self) {
///         self.messages.append(&mut other.messages);
///     }
/// }
///
/// let mut err1 = Error { messages: vec!["foo"] };
/// let err2 = Error { messages: vec!["bar", "baz"] };
///
/// err1.combine(err2);
/// assert_eq!(err1.messages, ["foo", "bar", "baz"]);
/// ```
pub trait FusedError: Sized {
    /// Drains `other`'s error messages into `self`'s error messages.
    ///
    /// If you find yourself frequently calling `err.combine(other)` only to
    /// return `err`, consider using [`merge`] instead.
    ///
    /// [`merge`]: FusedError::merge
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::{Accumulated, FusedError};
    ///
    /// let mut err1 = Accumulated::from("foo");
    /// let err2 = Accumulated::from(vec!["bar", "baz"]);
    /// let err3 = Accumulated::from("qux");
    ///
    /// err1.combine(err2);
    /// err1.combine(err3);
    ///
    /// assert_eq!(err1, ["foo", "bar", "baz", "qux"]);
    /// ```
    fn combine(&mut self, other: Self);

    /// Calls [`combine`] and returns `self` as a convenience for closures that
    /// need to return `Self` like [`Iterator::fold`] or [`Iterator::reduce`].
    ///
    /// [`combine`]: FusedError::combine
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::{Accumulated, FusedError};
    ///
    /// let errors = [
    ///     Accumulated::from("foo"),
    ///     Accumulated::from(vec!["bar", "baz"]),
    ///     Accumulated::from("qux"),
    /// ];
    ///
    /// let error = errors.into_iter().reduce(|a, b| a.merge(b)).unwrap();
    /// assert_eq!(error, ["foo", "bar", "baz", "qux"]);
    /// ```
    #[must_use]
    #[inline]
    fn merge(mut self, other: Self) -> Self {
        self.combine(other);
        self
    }
}

#[cfg(feature = "syn")]
#[cfg_attr(doc_cfg, doc(cfg(feature = "syn")))]
impl FusedError for syn::Error {
    #[inline]
    fn combine(&mut self, other: Self) {
        self.combine(other);
    }
}

/// A trait for splitting a result value into its "ok" and "error" parts.
///
/// # Safety
///
/// When implementing [`into_result_parts`], it is guaranteed that it **can
/// not** return [`None`] for both the "ok" and "error" parts. In other words,
/// it's only valid to return one of the following combinations:
///
/// * `(Some(T), Some(E))`
/// * `(Some(T), None)`
/// * `(None, Some(E))`
///
/// [`into_result_parts`]: IntoResultParts::into_result_parts
pub unsafe trait IntoResultParts {
    /// The "ok" component of the result value.
    type Ok;

    /// The "error" component of the result value.
    type Err;

    /// Splits a result-like value into its "ok" and "error" parts.
    ///
    /// It is **guaranteed** that `into_result_parts` **never** returns
    /// `(None, None)`. However, "ok" and "error" parts **are not** mutually
    /// exclusive. It's entirely possible `into_result_parts` returns
    /// `(Some(T), Some(E))`.
    ///
    /// # Examples
    ///
    /// [`Result<T, E>`](Result) implements `IntoResultParts<T, E>`:
    ///
    /// ```
    /// use fused_error::IntoResultParts;
    ///
    /// let result = "1".parse::<i32>();
    /// let (ok, error) = result.into_result_parts();
    /// assert_eq!(ok, Some(1));
    /// assert!(error.is_none());
    /// ```
    ///
    /// [`FusedResult<T, E>`](FusedResult) also implements
    /// `IntoResultParts<T, E>` when `E` implements [`FusedError`].n
    ///
    /// It is also **guaranteed** that the "ok" part for a fused result is
    /// **always** [`Some(T)`](Some).
    ///
    /// ```
    /// use fused_error::{Accumulated, Accumulator, FusedResult, IntoResultParts};
    ///
    /// let mut acc = Accumulator::<Accumulated<&str>>::new();
    /// acc.push("foo");
    /// acc.push("bar");
    ///
    /// let result = FusedResult::new(1i32, acc);
    /// let (ok, error) = result.into_result_parts();
    /// assert_eq!(ok.unwrap(), 1);
    /// assert_eq!(error.unwrap(), ["foo", "bar"]);
    /// ```
    #[must_use]
    fn into_result_parts(self) -> (Option<Self::Ok>, Option<Self::Err>);
}

unsafe impl<T, E> IntoResultParts for Result<T, E> {
    type Ok = T;
    type Err = E;

    #[inline]
    fn into_result_parts(self) -> (Option<T>, Option<E>) {
        match self {
            Ok(ok) => (Some(ok), None),
            Err(err) => (None, Some(err)),
        }
    }
}

/// A prelude to import the main items exported by this library:
///
/// * [`Accumulator<E>`]
/// * [`FusedError`]
/// * [`FusedResult<T, E>`](FusedResult)
/// * [`IteratorExt`]
/// * [`PackedResult<T, E>`](PackedResult)
///
/// *Note:* [`Accumulated<E>`] isn't in the prelude, as its use isn't advised
/// unless necessary. It's expected a minority of projects will actually use
/// [`Accumulated<E>`].
pub mod prelude {
    #[doc(inline)]
    pub use crate::{
        accumulator::Accumulator,
        iter::IteratorExt,
        result::{FusedResult, PackedResult},
        FusedError,
    };
}
