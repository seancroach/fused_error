//! Assigning [error accumulators] to working values with [`FusedResult<T, E>`].
//!
//! [error accumulators]: Accumulator
//! [`FusedResult<T, E>`]: FusedResult
//!
//! # fused results
//!
//! A [fused result](FusedResult) is meant for operations that always
//! yield at least *some* usable value. For instance, in this crate's
//! documentation found [here](crate), the example shows how a fused result
//! can be used to yield the sum of all valid inputs, yielding `0` even if every
//! input failed.
//!
//! # Packed Results
//!
//! ```ignore
//! type PackedResult<T, E> = Result<FusedResult<T, E>, E>;
//! ```
//!
//! A [packed result](PackedResult) is a [fused result](FusedResult)
//! which, itself, can have an [`Err`] case and short-circuit.
//!
//! While [fused results](FusedResult) are good for reporting multiple
//! errors, a [packed result](PackedResult) represents an operation which might
//! yield an internal error where panicking isn't desired.

use crate::{accumulator, Accumulator, FusedError, IntoResultParts};
use std::fmt::Debug;

mod iter;
mod raw;

pub use iter::{IntoIter, Iter, IterMut};

/// A [`FusedResult<T, E>`](FusedResult) that can short-circuit with an
/// error, dropping the "ok" value.
///
/// This type alias is particularly useful for representing operations that can
/// enter a state where any value yielded by the fused result should be seen
/// as invalid, such as reporting internal errors.
pub type PackedResult<T, E> = Result<FusedResult<T, E>, E>;

/// A result type that links an [`Accumulator<E>`] with a persistently "ok"
/// value of `T`.
///
/// # Panics
///
/// fused results will panic on drop if not handled correctly **even if no
/// errors are present**:
///
/// ```should_panic
/// use fused_error::FusedResult;
///
/// let res = FusedResult::<_, &str>::from_value(0);
///
/// // We dropped a fused result without handling it, so panic!
/// ```
///
/// To prevent it from panicking on drop, you must terminate a fused error
/// with one of the following methods:
///
/// # Examples
///
/// To see fused results in action, refer to the example in this crate's root
/// documentation [here](crate).
///
/// # Method overview
///
/// Here is the basic outline of the fused result methods at your disposal:
///
/// ## Creation
///
/// Methods to create a new fused result:
///
/// * [`new`] for creating a fused result with an "ok" value and an error
///   accumulator.
/// * [`from_value`] if no error accumulator already exists
///
/// ## Inspection
///
/// To get the internal values:
///
/// * [`value`] for getting a reference to the "ok" value
/// * [`value_mut`] for getting a mutable reference to the "ok" value
/// * [`acc`] for getting a reference to the internal error accumulator
/// * [`acc_mut`] for getting a mutable reference to the internal error
///   accumulator
///
/// For inspecting the general state of the fused result:
///
/// * [`err_count`] for getting the amount of errors accumulated
/// * [`is_ok`] returns `true` if the error count is zero.
/// * [`has_errors`] is the inverse of [`is_ok`]
///
/// For more specific inspection methods, consider [`is_ok_and`],
/// [`has_errors_and`], [`inspect`], or [`inspect_errors`].
///
/// ## Accumulation
///
/// The following accumulation methods should be familiar to an accumulator's:
///
/// * [`push_err`] appends an error to the internal error accumulator
/// * [`extend_err`] appends an error iterator to the accumulator
/// * [`trace`] pushes the error only if other errors are present
/// * [`trace_iter`] pushes the error iterator only if other errors are present
/// * [`trace_with`] is the lazy version of [`trace`]
///
/// ## Transformation
///
/// Unfortunately, unlike accumulators, map methods are needed to transform the
/// internal values:
///
/// * [`map`] maps the "ok" value
/// * [`map_err`] maps any possible error values
///
/// If you have a `FusedResult<&T, E>`:
///
/// * [`copied`] maps the "ok" value of `&T` to `T` using [`Copy`]
/// * [`cloned`] maps the "ok" value of `&T` to `T` using [`Clone`]
///
/// If you have a `FusedResult<Result<T, A>, B>`, you can call [`transpose`]
/// which will convert it into `Result<FusedResult<T, B>, A>`.
///
/// Or, if you have `FusedResult<Result<T, IE>, E>` where `E` implements
/// [`FusedError`] and `IE` implements [`Into<E>`], you can call [`flatten`]
/// which will convert it into `FusedResult<Option<T>, E>`.
///
/// ## Iteration
///
/// The following methods yields an iterator of exactly one element over the
/// "ok" value:
///
/// * [`iter`] yields an iterator over the "ok" value
/// * [`iter_mut`] yields a mutable iterator of the "ok" value
/// * [`into_iter`] is considered unsafe, dropping any errors and moving the
///   "ok" value into an iterator
///
/// Likewise, for the errors:
///
/// * [`err_iter`] yields an iterator over the error values
/// * [`err_iter_mut`] yields a mutable iterator over the error values
/// * [`into_err_iter`] drops the "ok" value, moving the errors into an iterator
///
/// ## Destructuring
///
/// These methods are useful for destructuring a fused result without any
/// form of panicking:
///
/// * [`split`] destructures the fused error into its "ok" value and internal
///   error accumulator
/// * [`errors`] returns the vector of collected errors, dropping the "ok" value
/// * [`err`] returns the reduced error if the error type implements
///   [`FusedError`] as an option
///
/// ## Extracting the "ok" value
///
/// These methods extract the contained "ok" value in a [`FusedResult<T, E>`]
/// when there are no errors present:
///
/// * [`expect`] panics with a provided custom message
/// * [`unwrap`] panics with a generic message
/// * [`unwrap_unchecked`] discards any accumulated errors
///
/// The panicking methods [`expect`] and [`unwrap`] require `T` and `E` to
/// implement the [`Debug`] trait.
///
/// ## Extracting a [`FusedError`]
///
/// These methods extract the contained error value if any are present and `E`
/// implements [`FusedError`]:
///
/// * [`expect_err`] panics with a provided custom message
/// * [`unwrap_err`] panics with a generic message
/// * [`unwrap_err_unchecked`] does not check if there are no errors
///
/// ## Conversion
///
/// These methods convert the fused result into a potentially more usable
/// type
///
/// * [`into_iter`] discards any errors and yields an iterator over the "ok"
///   value; it's considered semantically unsafe
/// * [`into_err_iter`] drops the "ok" value and yields an iterator over the
///   errors
/// * [`into_result`] drops the "ok" value if any errors are present, yielding a
///   [`Result<T, E>`](Result).
///
/// If you want to destructure the fused result explicitly while discarding
/// its value, for whatever reason, use [`ignore`] which drops the "ok" value
/// and discards any errors that are present.
///
/// [`FusedResult<T, E>`]: FusedResult
///
/// [`new`]: FusedResult::new
/// [`from_value`]: FusedResult::from_value
/// [`value`]: FusedResult::value
/// [`value_mut`]: FusedResult::value_mut
/// [`acc`]: FusedResult::acc
/// [`acc_mut`]: FusedResult::acc_mut
/// [`err_count`]: FusedResult::err_count
/// [`is_ok`]: FusedResult::is_ok
/// [`is_ok_and`]: FusedResult::is_ok_and
/// [`has_errors`]: FusedResult::has_errors
/// [`has_errors_and`]: FusedResult::has_errors_and
/// [`inspect`]: FusedResult::inspect
/// [`inspect_errors`]: FusedResult::inspect_errors
/// [`split`]: FusedResult::split
/// [`errors`]: FusedResult::errors
/// [`err`]: FusedResult::err
/// [`push_err`]: FusedResult::push_err
/// [`extend_err`]: FusedResult::extend_err
/// [`trace`]: FusedResult::trace
/// [`trace_iter`]: FusedResult::trace_iter
/// [`trace_with`]: FusedResult::trace_with
/// [`map`]: FusedResult::map
/// [`map_err`]: FusedResult::map_err
/// [`iter`]: FusedResult::iter
/// [`iter_mut`]: FusedResult::iter_mut
/// [`into_iter`]: FusedResult::into_iter
/// [`err_iter`]: FusedResult::err_iter
/// [`err_iter_mut`]: FusedResult::err_iter_mut
/// [`into_err_iter`]: FusedResult::into_err_iter
/// [`into_result`]: FusedResult::into_result
/// [`ignore`]: FusedResult::ignore
/// [`expect`]: FusedResult::expect
/// [`unwrap`]: FusedResult::unwrap
/// [`unwrap_unchecked`]: FusedResult::unwrap_unchecked
/// [`expect_err`]: FusedResult::expect_err
/// [`unwrap_err`]: FusedResult::unwrap_err
/// [`unwrap_err_unchecked`]: FusedResult::unwrap_err_unchecked
/// [`transpose`]: FusedResult::transpose
/// [`flatten`]: FusedResult::flatten
/// [`copied`]: FusedResult::copied
/// [`cloned`]: FusedResult::cloned
#[derive(Debug)]
#[must_use = "compund results will panic on drop if not handled"]
pub struct FusedResult<T, E> {
    inner: raw::FusedResult<T, E>,
}

impl<T, E> FusedResult<T, E> {
    /// Constructs a new fused result from a value and an
    /// [error accumulator](Accumulator).
    ///
    /// If you don't already have an error accumulator instantiated, consider
    /// using [`from_value`](FusedResult::from_value) instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::{Accumulator, FusedResult};
    ///
    /// let value = 1;
    /// let mut acc = Accumulator::<&str>::new();
    /// let res = FusedResult::new(value, acc);
    ///
    /// // ...
    /// # unsafe { res.ignore(); }
    /// ```
    ///
    /// *Note*: You may have noticed the trailing ellipsis comment. This is
    /// because, as outlined in the documentation for
    /// [`FusedResult<T, E>`](Self), any unhandled result will panic on drop,
    /// similar to [`Accumulator<E>`](Accumulator). That comment in any example
    /// is meant to signify the fused result getting handled at a later point
    /// in the program.
    #[inline]
    pub fn new(value: T, acc: Accumulator<E>) -> Self {
        let inner = raw::FusedResult::new(value, acc);
        FusedResult { inner }
    }

    /// Constructs a new fused result from a value.
    ///
    /// If you already have an [error accumulator](Accumulator) you wish to link
    /// to the created fused result, consider using
    /// [`new`](FusedResult::new) instead.
    ///
    /// Because fused results are so generalized, it can cause problems with
    /// type inference. As such, `from_value` should probably get called with
    /// the 'turbofish' syntax: `::<>`. This helps the inference algorithm
    /// understand specifically which error type you're accumulating with this
    /// fused result.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::{FusedResult};
    ///
    /// let mut res = FusedResult::<_, &str>::from_value(0);
    ///
    /// // ...
    /// # unsafe { res.ignore(); }
    /// ```
    #[inline]
    pub fn from_value(value: T) -> Self {
        FusedResult::new(value, Accumulator::new())
    }

    /// Returns a reference to the contained value.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::FusedResult;
    ///
    /// let mut res = FusedResult::<_, &str>::from_value(0);
    /// assert_eq!(res.value(), &0);
    ///
    /// // ...
    /// # unsafe { res.ignore(); }
    /// ```
    pub fn value(&self) -> &T {
        &self.inner.ok
    }

    /// Returns a mutable reference to the contained value.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::FusedResult;
    ///
    /// let mut res = FusedResult::<_, &str>::from_value(0);
    /// *res.value_mut() = 42;
    ///
    /// assert_eq!(res.value(), &42);
    ///
    /// // ...
    /// # unsafe { res.ignore(); }
    /// ```
    pub fn value_mut(&mut self) -> &mut T {
        &mut self.inner.ok
    }

    /// Returns a reference to the contained [error accumulator](Accumulator).
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::{Accumulator, FusedResult};
    ///
    /// let mut res = FusedResult::<_, &str>::from_value(0);
    /// let acc: &Accumulator<&str> = res.acc();
    ///
    /// // ...
    /// # unsafe { res.ignore(); }
    /// ```
    pub fn acc(&self) -> &Accumulator<E> {
        &self.inner.errors
    }

    /// Returns a mutable reference to the contained
    /// [error accumulator](Accumulator).
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::{Accumulator, FusedResult};
    ///
    /// let mut res = FusedResult::<_, &str>::from_value(0);
    /// let acc: &mut Accumulator<&str> = res.acc_mut();
    ///
    /// // ...
    /// # unsafe { res.ignore(); }
    /// ```
    pub fn acc_mut(&mut self) -> &mut Accumulator<E> {
        &mut self.inner.errors
    }

    /// Returns the amount of accumulated errors within the contained
    /// [error accumulator](Accumulator).
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::{Accumulator, FusedResult};
    ///
    /// let mut acc = Accumulator::<&str>::new();
    /// acc.push("foo");
    ///
    /// let mut res = FusedResult::new(0, acc);
    /// res.push_err("bar");
    ///
    /// assert_eq!(res.err_count(), 2);
    ///
    /// // ...
    /// # unsafe { res.ignore(); }
    /// ```
    #[must_use]
    #[inline]
    pub fn err_count(&self) -> usize {
        self.inner.errors.len()
    }

    /// Returns `true` if the amount of collected errors within the contained
    /// [error accumulator](Accumulator) is equal to zero.
    ///
    /// The inverse of this method is
    /// [`has_errors`](FusedResult::has_errors).
    ///
    /// ```
    /// use fused_error::FusedResult;
    ///
    /// let mut res = FusedResult::<_, &str>::from_value(0);
    /// assert!(res.is_ok());
    ///
    /// res.push_err("foo");
    /// assert!(!res.is_ok());
    ///
    /// // ...
    /// # unsafe { res.ignore(); }
    /// ```
    #[must_use]
    #[inline]
    pub fn is_ok(&self) -> bool {
        self.err_count() == 0
    }

    /// Returns `true` if the amount of collected errors within the contained
    /// [error accumulator](Accumulator) is equal to zero and the value inside
    /// of this fused result matches a predicate.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::FusedResult;
    ///
    /// let mut res1 = FusedResult::<_, &str>::from_value(0);
    /// assert_eq!(res1.is_ok_and(|&x| x < 1), true);
    ///
    /// let mut res2 = FusedResult::<_, &str>::from_value(2);
    /// assert_eq!(res2.is_ok_and(|&x| x < 1), false);
    ///
    /// let mut res3 = FusedResult::<_, &str>::from_value(0);
    /// res3.push_err("foo");
    /// assert_eq!(res3.is_ok_and(|&x| x < 1), false);
    ///
    /// // ...
    /// # unsafe {
    /// #     res1.ignore();
    /// #     res2.ignore();
    /// #     res3.ignore();
    /// # }
    /// ```
    #[must_use]
    #[inline]
    pub fn is_ok_and<F>(&self, f: F) -> bool
    where
        F: FnOnce(&T) -> bool,
    {
        self.is_ok() && f(&self.inner.ok)
    }

    /// Returns `true` if there are no errors present within the contained
    /// [error accumulator](Accumulator).
    ///
    /// The inverse of this method is [`is_ok`](FusedResult::is_ok).
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::FusedResult;
    ///
    /// let mut res = FusedResult::<_, &str>::from_value(0);
    /// assert!(!res.has_errors());
    ///
    /// res.push_err("foo");
    /// assert!(res.has_errors());
    ///
    /// // ...
    /// # unsafe { res.ignore(); }
    /// ```
    #[must_use]
    #[inline]
    pub fn has_errors(&self) -> bool {
        self.err_count() != 0
    }

    /// Returns `true` if there are no errors present within the contained
    /// [error accumulator](Accumulator) and the components of this fused
    /// result matches a predicate.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::{Accumulator, FusedResult};
    ///
    /// fn errors_are_lowercase(acc: &Accumulator<&str>) -> bool {
    ///     acc.iter().all(|s| s.chars().all(char::is_lowercase))
    /// }
    ///
    /// let mut res1 = FusedResult::<_, &str>::from_value(0);
    /// res1.push_err("foo");
    /// assert_eq!(
    ///     res1.has_errors_and(|&x, acc| x < 1 && errors_are_lowercase(acc)),
    ///     true,
    /// );
    ///
    /// let mut res2 = FusedResult::<_, &str>::from_value(2);
    /// res2.push_err("foo");
    /// res2.push_err("BAR");
    /// assert_eq!(
    ///     res2.has_errors_and(|&x, acc| x < 1 && errors_are_lowercase(acc)),
    ///     false,
    /// );
    ///
    /// let mut res3 = FusedResult::<_, &str>::from_value(0);
    /// assert_eq!(
    ///     res3.has_errors_and(|&x, acc| x < 1 && errors_are_lowercase(acc)),
    ///     false,
    /// );
    ///
    /// // ...
    /// # unsafe {
    /// #     res1.ignore();
    /// #     res2.ignore();
    /// #     res3.ignore();
    /// # }
    /// ```
    #[must_use]
    #[inline]
    pub fn has_errors_and<F>(&self, f: F) -> bool
    where
        F: FnOnce(&T, &Accumulator<E>) -> bool,
    {
        self.has_errors() && f(&self.inner.ok, &self.inner.errors)
    }

    /// Calls the provided closure with a reference to the contained value.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::FusedResult;
    ///
    /// let res = FusedResult::<i32, &str>::from_value(4);
    ///
    /// let x = res
    ///     .inspect(|x| println!("original: {x}"))
    ///     .map(|x| x.pow(3))
    ///     .unwrap();
    /// ```
    pub fn inspect<F>(self, f: F) -> Self
    where
        F: FnOnce(&T),
    {
        f(&self.inner.ok);
        self
    }

    /// Calls the provided closure with a reference to the contained
    /// [error accumulator](Accumulator).
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::FusedResult;
    ///
    /// let mut res = FusedResult::<_, &str>::from_value(0);
    /// res.push_err("foo");
    /// res.push_err("bar");
    ///
    /// let mut res = res
    ///     .inspect_errors(|acc| println!("formatting {} errors", acc.len()))
    ///     .map_err(|msg| format!("failed operation: {msg}"));
    ///
    /// // ...
    /// # unsafe { res.ignore(); }
    /// ```
    pub fn inspect_errors<F>(self, f: F) -> Self
    where
        F: FnOnce(&Accumulator<E>),
    {
        f(&self.inner.errors);
        self
    }

    /// Deconstructs the [`FusedResult<T, E>`](Self) into its base
    /// components: the contained value, `T`, and the [`Accumulator<E>`].
    ///
    /// Calling this method ensures the fused result **will not** panic on
    /// drop.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::{Accumulator, FusedResult};
    ///
    /// let mut acc = Accumulator::<&str>::new();
    /// acc.push("foo");
    ///
    /// let mut res = FusedResult::new(0, acc);
    /// res.push_err("bar");
    ///
    /// let (x, acc) = res.split();
    /// assert_eq!(x, 0);
    /// assert_eq!(acc.into_vec(), ["foo", "bar"]);
    /// ```
    pub fn split(mut self) -> (T, Accumulator<E>) {
        unsafe { self.inner.split() }
    }

    /// Discards the contained value, returning a vector of all the accumulated
    /// errors.
    ///
    /// Calling this method ensures the fused result **will not** panic on
    /// drop.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::FusedResult;
    ///
    /// let mut res = FusedResult::<_, &str>::from_value(0);
    /// res.push_err("foo");
    /// res.push_err("bar");
    ///
    /// assert_eq!(res.errors(), ["foo", "bar"]);
    /// ```
    #[must_use]
    #[inline]
    pub fn errors(mut self) -> Vec<E> {
        unsafe { self.inner.take_errors() }
    }

    /// Drops the contained value and calls [`err`] on the underlying
    /// [error accumulator](Accumulator).
    ///
    /// Calling this method ensures the fused result **will not** panic on
    /// drop.
    ///
    /// # Examples
    ///
    /// For examples of [`err`], refer to the [`Accumulator<E>`] item and
    /// [`accumulator`](crate::accumulator) module documentation.
    ///
    /// [`err`]: Accumulator::err
    #[must_use]
    #[inline]
    pub fn err(mut self) -> Option<E>
    where
        E: FusedError,
    {
        unsafe { self.inner.take_err() }
    }

    /// Calls [`push`] on the underlying [error accumulator].
    ///
    /// # Panics
    ///
    /// Panics if the new capacity of the [error accumulator] exceeds
    /// `isize::MAX` bytes.
    ///
    /// # Examples
    ///
    /// For examples of [`push`], refer to its documentation
    /// [here](Accumulator::push).
    ///
    /// [error accumulator]: Accumulator
    /// [`push`]: Accumulator::push
    #[inline]
    pub fn push_err<IE>(&mut self, err: IE)
    where
        IE: Into<E>,
    {
        self.inner.errors.push(err.into());
    }

    /// Calls [`Extend::extend`] on the underlying [error accumulator].
    ///
    /// # Panics
    ///
    /// Panics if the new capacity of the [error accumulator] exceeds
    /// `isize::MAX` bytes.
    ///
    /// # Examples
    ///
    /// For examples of [`extend`](Extend::extend), refer to its documentation
    /// [here](Extend::extend).
    ///
    /// [error accumulator]: Accumulator
    pub fn extend_err<IE, I>(&mut self, iter: I)
    where
        IE: Into<E>,
        I: IntoIterator<Item = IE>,
    {
        self.inner.errors.extend(iter);
    }

    /// Calls [`trace`] on the underlying [error accumulator].
    ///
    /// # Panics
    ///
    /// Panics if the new capacity of the [error accumulator] exceeds
    /// `isize::MAX` bytes.
    ///
    /// # Examples
    ///
    /// For examples of [`trace`], refer to its documentation [here].
    ///
    /// [error accumulator]: Accumulator
    /// [`trace`]: Accumulator::trace
    /// [here]: Accumulator::trace
    pub fn trace<IE>(&mut self, err: IE)
    where
        IE: Into<E>,
    {
        self.inner.errors.trace(err);
    }

    /// Calls [`trace_iter`] on the underlying [error accumulator].
    ///
    /// # Panics
    ///
    /// Panics if the new capacity of the [error accumulator] exceeds
    /// `isize::MAX` bytes.
    ///
    /// # Examples
    ///
    /// For examples of [`trace_iter`], refer to its documentation [here].
    ///
    /// [error accumulator]: Accumulator
    /// [`trace_iter`]: Accumulator::trace_iter
    /// [here]: Accumulator::trace_iter
    pub fn trace_iter<IE, I>(&mut self, iter: I)
    where
        IE: Into<E>,
        I: IntoIterator<Item = IE>,
    {
        self.inner.errors.trace_iter(iter);
    }

    /// Calls [`trace_with`] on the underlying [error accumulator].
    ///
    /// # Panics
    ///
    /// Panics if the new capacity of the [error accumulator] exceeds
    /// `isize::MAX` bytes.
    ///
    /// # Examples
    ///
    /// For examples of [`trace_with`], refer to its documentation [here].
    ///
    /// [error accumulator]: Accumulator
    /// [`trace_with`]: Accumulator::trace_with
    /// [here]: Accumulator::trace
    pub fn trace_with<IE, F>(&mut self, f: F)
    where
        IE: Into<E>,
        F: FnOnce() -> IE,
    {
        self.inner.errors.trace_with(f);
    }

    /// Maps a `FusedResult<T, E>` to `FusedResult<U, E>` by applying a
    /// function to the contained value.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::FusedResult;
    ///
    /// let res = FusedResult::<_, &str>::from_value(21);
    /// let mut mapped = res.map(|x| x * 2);
    /// assert_eq!(mapped.value(), &42);
    ///
    /// // ...
    /// # unsafe { mapped.ignore(); }
    /// ```
    #[inline]
    pub fn map<U, O>(mut self, op: O) -> FusedResult<U, E>
    where
        O: FnOnce(T) -> U,
    {
        let (value, acc) = unsafe { self.inner.split() };
        FusedResult::new(op(value), acc)
    }

    /// Maps a `FusedResult<T, E>` to `FusedResult<T, F>` by applying a
    /// function to every accumulated error.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::FusedResult;
    ///
    /// let mut res = FusedResult::<_, &str>::from_value(0);
    /// res.push_err("foo");
    /// res.push_err("bar");
    ///
    /// let mut res = res.map_err(|err| err.to_uppercase());
    /// assert_eq!(res.errors(), ["FOO", "BAR"]);
    /// ```
    ///
    #[inline]
    pub fn map_err<F, O>(mut self, op: O) -> FusedResult<T, F>
    where
        O: FnMut(E) -> F,
    {
        let (value, acc) = unsafe { self.inner.split() };
        let acc = acc.into_iter().map(op).collect();
        FusedResult::new(value, acc)
    }

    /// Returns an iterator over the contained value.
    ///
    /// This iterator yields exactly one value.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::FusedResult;
    ///
    /// let mut res = FusedResult::<_, &str>::from_value(0);
    /// let mut iter = res.iter();
    /// assert_eq!(iter.next(), Some(&0));
    /// assert_eq!(iter.next(), None);
    ///
    /// // ...
    /// # unsafe { res.ignore() };
    /// ```
    pub fn iter(&self) -> Iter<T> {
        Iter {
            inner: Some(&self.inner.ok),
        }
    }

    /// Returns a mutable iterator over the contained value.
    ///
    /// This iterator yields exactly one value.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::FusedResult;
    ///
    /// let mut res = FusedResult::<_, &str>::from_value(0);
    ///
    /// for v in res.iter_mut() {
    ///     *v = 42;
    /// }
    ///
    /// assert_eq!(res.unwrap(), 42);
    /// ```
    pub fn iter_mut(&mut self) -> IterMut<T> {
        IterMut {
            inner: Some(&mut self.inner.ok),
        }
    }

    /// Returns a consuming iterator over the contained value.
    ///
    /// Calling this method ensures the fused result **will not** panic on
    /// drop.
    ///
    /// # Safety
    ///
    /// This discards all possibly contained errors, making it about as
    /// dangerous as [`unwrap_unchecked`](FusedResult::unwrap_unchecked).
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::FusedResult;
    ///
    /// let mut res = FusedResult::<_, &str>::from_value(0);
    /// res.push_err("foo");
    /// res.push_err("bar");
    ///
    /// // SAFETY: This discards the two previous errors without a trace!
    /// // Be careful.
    /// let mut iter = unsafe { res.into_iter() };
    /// assert_eq!(iter.next(), Some(0));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub unsafe fn into_iter(mut self) -> IntoIter<T> {
        Result::<T, ()>::Ok(self.inner.take_value()).into_iter()
    }

    /// Returns an iterator over the contained errors.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::FusedResult;
    ///
    /// let mut res = FusedResult::<_, &str>::from_value(0);
    /// res.push_err("foo");
    /// res.push_err("bar");
    ///
    /// let mut err_iter = res.err_iter();
    /// assert_eq!(err_iter.next(), Some(&"foo"));
    /// assert_eq!(err_iter.next(), Some(&"bar"));
    /// assert_eq!(err_iter.next(), None);
    ///
    /// // ...
    /// # unsafe { res.ignore() }
    /// ```
    pub fn err_iter(&self) -> accumulator::Iter<E> {
        self.inner.errors.iter()
    }

    /// Returns a mutable iterator over the contained errors.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::FusedResult;
    ///
    /// let mut res = FusedResult::<_, String>::from_value(0);
    /// res.push_err("foo");
    /// res.push_err("bar");
    ///
    /// for err in res.err_iter_mut() {
    ///     err.make_ascii_uppercase();
    /// }
    ///
    /// assert_eq!(res.errors(), ["FOO", "BAR"]);
    /// ```
    pub fn err_iter_mut(&mut self) -> accumulator::IterMut<E> {
        self.inner.errors.iter_mut()
    }

    /// Drops the contained value and returns a consuming iterator over the
    /// contained errors.
    ///
    /// Calling this method ensures the fused result **will not** panic on
    /// drop.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::FusedResult;
    ///
    /// let mut res = FusedResult::<_, &str>::from_value(0);
    /// res.push_err("foo");
    /// res.push_err("bar");
    ///
    /// let mut iter = res.into_err_iter();
    /// assert_eq!(iter.next(), Some("foo"));
    /// assert_eq!(iter.next(), Some("bar"));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn into_err_iter(mut self) -> accumulator::IntoIter<E> {
        let errors = unsafe { self.inner.take_errors() };
        errors.into_iter()
    }

    /// If no errors are present, `into_result` returns `Ok(T)`. However,
    /// if errors are present, `into_result` returns `Err(E)`, dropping the
    /// contained value.
    ///
    /// Calling this method ensures the fused result **will not** panic on
    /// drop.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::{Accumulated, FusedResult};
    ///
    /// let mut res = FusedResult::<_, Accumulated<&str>>::from_value(0);
    /// assert_eq!(res.into_result(), Ok(0));
    /// ```
    #[inline]
    pub fn into_result(mut self) -> Result<T, E>
    where
        E: FusedError,
    {
        unsafe { self.inner.result() }
    }

    /// Handles this accumulator, discarding all errors. Unlike
    /// [`unwrap`](FusedResult::unwrap) or similar methods, this method does
    /// not panic.
    ///
    /// Calling this method ensures the fused result **will not** panic on
    /// drop.
    ///
    /// # Safety
    ///
    /// It is considered semantically unsafe to discard errors, especially the
    /// internal error accumulator due to the volume of errors they can store.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::FusedResult;
    ///
    /// let mut res = FusedResult::<_, &str>::from_value(0);
    ///
    /// // TODO: properly handle result
    /// unsafe { res.ignore() };
    /// ```
    #[inline]
    pub unsafe fn ignore(mut self) {
        self.inner.ignore();
    }
}

impl<T, E> FusedResult<T, E>
where
    T: Debug,
    E: Debug,
{
    /// Returns the contained value, consuming the `self` value.
    ///
    /// # Panics
    ///
    /// Panics if any errors have been accumulated, with a panic message
    /// including the passed message, and the content of the internal
    /// accumulator.
    ///
    /// # Examples
    ///
    /// ```should_panic
    /// use fused_error::FusedResult;
    ///
    /// let mut res = FusedResult::<_, &str>::from_value(0);
    /// res.push_err("foo");
    /// res.push_err("bar");
    ///
    /// res.expect("testing expect"); // panics with "testing expect: ["foo", "bar"]"
    /// ```
    #[inline]
    #[track_caller]
    pub fn expect(mut self, msg: &str) -> T {
        if self.has_errors() {
            unsafe { self.inner.fail_with(msg) };
        }
        unsafe { self.inner.take_value() }
    }

    /// Returns the contained value, consuming the `self` value.
    ///
    /// # Panics
    ///
    /// Panics if any errors have been accumulated with a panic message
    /// including the content of the internal accumulator.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::FusedResult;
    ///
    /// let mut res = FusedResult::<_, &str>::from_value(0);
    /// assert_eq!(res.unwrap(), 0);
    /// ```
    ///
    /// ```should_panic
    /// use fused_error::FusedResult;
    ///
    /// let mut res = FusedResult::<_, &str>::from_value(0);
    /// res.push_err("foo");
    /// res.unwrap(); // panics due to previous appended error
    /// ```
    #[inline]
    #[track_caller]
    pub fn unwrap(mut self) -> T {
        if self.has_errors() {
            unsafe { self.inner.fail("unwrap") };
        }
        unsafe { self.inner.take_value() }
    }

    /// Returns the contained value, consuming the `self` value, without
    /// checking if any errors are accumulated.
    ///
    /// # Safety
    ///
    /// Calling this method when errors are present is not undefined behavior,
    /// but it is a semantic error.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::FusedResult;
    ///
    /// let mut res = FusedResult::<_, &str>::from_value(0);
    /// assert_eq!(unsafe { res.unwrap_unchecked() }, 0);
    /// ```
    #[must_use]
    #[inline]
    #[track_caller]
    pub unsafe fn unwrap_unchecked(mut self) -> T {
        if cfg!(debug_assertions) && self.has_errors() {
            self.inner.fail("unwrap_unchecked");
        }
        self.inner.take_value()
    }

    /// Returns the contained error reduced via [`FusedError`], consuming the
    /// `self` value.
    ///
    /// # Panics
    ///
    /// Panics if no errors are present, with a panic message including the
    /// passed message, and the content of the internal accumulator.
    ///
    /// # Examples
    ///
    /// ```should_panic
    /// use fused_error::{Accumulated, FusedResult};
    ///
    /// let mut res = FusedResult::<_, Accumulated<&str>>::from_value(0);
    /// res.expect_err("testing expect_err"); // panics with "testing expect_err: []"
    /// ```
    #[inline]
    #[track_caller]
    pub fn expect_err(mut self, msg: &str) -> E
    where
        E: FusedError,
    {
        if cfg!(debug_assertions) && !self.has_errors() {
            unsafe { self.inner.fail_with(msg) };
        }
        unsafe { self.inner.take_err().unwrap_unchecked() }
    }

    /// Returns the contained error reduced via [`FusedError`], consuming the
    /// `self` value.
    ///
    /// # Panics
    ///
    /// Panics if no errors are present, with a panic message including the
    /// content of the internal accumulator.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::{Accumulated, FusedResult};
    ///
    /// let mut res = FusedResult::<_, Accumulated<&str>>::from_value(0);
    /// res.push_err("foo");
    /// res.push_err("bar");
    /// assert_eq!(res.unwrap_err(), ["foo", "bar"]);
    /// ```
    #[inline]
    #[track_caller]
    pub fn unwrap_err(mut self) -> E
    where
        E: FusedError,
    {
        if cfg!(debug_assertions) && !self.has_errors() {
            unsafe { self.inner.fail("unwrap_err") };
        }
        unsafe { self.inner.take_err().unwrap_unchecked() }
    }

    /// Returns the contained error reduced via [`FusedError`], consuming the
    /// `self` value, without checking if any errors are accumulated.
    ///
    /// # Safety
    ///
    /// Calling this method when errors aren't present is not undefined
    /// behavior, but it is a semantic error.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::{Accumulated, FusedResult};
    ///
    /// let mut res = FusedResult::<_, Accumulated<&str>>::from_value(0);
    /// res.push_err("foo");
    /// res.push_err("bar");
    /// assert_eq!(unsafe { res.unwrap_err_unchecked() }, ["foo", "bar"]);
    /// ```
    #[inline]
    #[track_caller]
    pub unsafe fn unwrap_err_unchecked(mut self) -> E
    where
        E: FusedError,
    {
        if cfg!(debug_assertions) && !self.has_errors() {
            self.inner.fail("unwrap_err");
        }
        self.inner.take_err().unwrap_unchecked()
    }
}

impl<T, A, B> FusedResult<Result<T, A>, B> {
    /// Converts a `FusedResult<Result<T, A>, B>` into a
    /// `Result<FusedResult<T, B>, A>`.
    ///
    /// # Safety
    ///
    /// This drops all of the accumulated errors within the fused result if
    /// the internal result is an [`Err`] value.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::FusedResult;
    ///
    /// let inner: Result<u32, i32> = Err(0);
    ///
    /// let mut fused_result = FusedResult::<_, &str>::from_value(inner);
    /// fused_result.push_err("foo");
    /// fused_result.push_err("bar");
    ///
    /// // SAFETY: This *will* drop the previously appended errors because the
    /// // inner result is an `Err` value:
    /// let result: Result<FusedResult<u32, &str>, i32> = unsafe { fused_result.transpose() };
    /// assert_eq!(result.unwrap_err(), 0)
    /// ```

    pub unsafe fn transpose(mut self) -> Result<FusedResult<T, B>, A> {
        let (res, acc) = self.inner.split();
        match res {
            Ok(t) => Ok(FusedResult::new(t, acc)),
            Err(e) => {
                acc.ignore();
                Err(e)
            }
        }
    }
}

impl<T, E, IE> FusedResult<Result<T, IE>, E> {
    /// Flattens a `FusedResult<Result<T, IE>, E>` into a
    /// `FusedResult<Option<T>, E>` where `IE` implements [`Into<E>`](Into).
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::FusedResult;
    ///
    /// let inner: Result<i32, &str> = Err("foo");
    ///
    /// let mut fused_result = FusedResult::<_, &str>::from_value(inner);
    /// fused_result.push_err("bar");
    /// fused_result.push_err("baz");
    ///
    /// let flattened = fused_result.flatten();
    /// assert!(flattened.value().is_none());
    /// assert_eq!(flattened.errors(), ["bar", "baz", "foo"]);
    /// ```
    pub fn flatten(mut self) -> FusedResult<Option<T>, E>
    where
        IE: Into<E>,
    {
        let (res, mut acc) = unsafe { self.inner.split() };
        let opt = match res {
            Ok(t) => Some(t),
            Err(e) => {
                acc.push(e);
                None
            }
        };
        FusedResult::new(opt, acc)
    }
}

impl<T, E> FusedResult<&T, E> {
    /// Maps a `FusedResult<&T, E>` to a `FusedResult<T, E>` by copying
    /// the "ok" contents.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::FusedResult;
    ///
    /// let value = 1;
    ///
    /// let res = FusedResult::<&i32, &str>::from_value(&value);
    /// assert_eq!(res.value(), &&1);
    /// let copied: FusedResult<i32, &str> = res.copied();
    /// assert_eq!(copied.value(), &1);
    ///
    /// // ...
    /// # unsafe { copied.ignore() };
    /// ```
    #[inline]
    pub fn copied(self) -> FusedResult<T, E>
    where
        T: Copy,
    {
        self.map(|&t| t)
    }

    /// Maps a `FusedResult<&T, E>` to a `FusedResult<T, E>` by cloning
    /// the "ok" contents.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::FusedResult;
    ///
    /// let value = 1;
    ///
    /// let res = FusedResult::<&i32, &str>::from_value(&value);
    /// assert_eq!(res.value(), &&1);
    /// let cloned: FusedResult<i32, &str> = res.cloned();
    /// assert_eq!(cloned.value(), &1);
    ///
    /// // ...
    /// # unsafe { cloned.ignore() };
    /// ```
    #[inline]
    pub fn cloned(self) -> FusedResult<T, E>
    where
        T: Clone,
    {
        self.map(Clone::clone)
    }
}

impl<T, E> FusedResult<&mut T, E> {
    /// Maps a `FusedResult<&mut T, E>` to a `FusedResult<T, E>` by
    /// copying the "ok" contents.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::FusedResult;
    ///
    /// let mut value = 1;
    ///
    /// let res = FusedResult::<&mut i32, &str>::from_value(&mut value);
    /// assert_eq!(res.value(), &&mut 1);
    /// let copied: FusedResult<i32, &str> = res.copied();
    /// assert_eq!(copied.value(), &1);
    ///
    /// // ...
    /// # unsafe { copied.ignore() };
    /// ```
    #[inline]
    pub fn copied(self) -> FusedResult<T, E>
    where
        T: Copy,
    {
        self.map(|&mut t| t)
    }

    /// Maps a `FusedResult<&mut T, E>` to a `FusedResult<T, E>` by
    /// cloning the "ok" contents.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::FusedResult;
    ///
    /// let mut value = 1;
    ///
    /// let res = FusedResult::<&mut i32, &str>::from_value(&mut value);
    /// assert_eq!(res.value(), &&mut 1);
    /// let cloned: FusedResult<i32, &str> = res.cloned();
    /// assert_eq!(cloned.value(), &1);
    ///
    /// // ...
    /// # unsafe { cloned.ignore() };
    /// ```
    #[inline]
    pub fn cloned(self) -> FusedResult<T, E>
    where
        T: Clone,
    {
        self.map(|t| t.clone())
    }
}

unsafe impl<T, E> IntoResultParts for FusedResult<T, E>
where
    E: FusedError,
{
    type Ok = T;
    type Err = E;

    fn into_result_parts(self) -> (Option<T>, Option<E>) {
        let (ok, acc) = self.split();
        (Some(ok), acc.err())
    }
}

impl<T: Default, E> Default for FusedResult<T, E> {
    #[inline]
    fn default() -> Self {
        FusedResult::from_value(T::default())
    }
}

impl<T, R> FromIterator<R> for FusedResult<T, R::Err>
where
    T: FromIterator<R::Ok>,
    R: IntoResultParts,
{
    fn from_iter<I: IntoIterator<Item = R>>(iter: I) -> Self {
        let mut acc = Accumulator::<R::Err>::new();
        let collected: T = iter.into_iter().filter_map(|res| acc.handle(res)).collect();
        FusedResult::new(collected, acc)
    }
}
