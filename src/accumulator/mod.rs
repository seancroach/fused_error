//! Error accumulation via [`Accumulator<E>`].
//!
//! Error accumulators are useful for collecting multiple errors throughout an
//! operation to provide more holistic diagnostics or to defer stages of error
//! handling.
//!
//! # Panics
//!
//! Accumulators will panic on drop if not handled correctly **even if empty**:
//!
//! ```should_panic
//! use fused_error::Accumulator;
//!
//! let mut acc = Accumulator::<&str>::new();
//!
//! // We dropped an accumulator without handling it, so panic!
//! ```
//!
//! To prevent it from panicking on drop, you must terminate an accumulator with
//! any of the methods that take `self` by value. For a comprehensive overview,
//! look at the "Termination" section.
//!
//! The accumulator panicking, despite the annoying error stack trace, is a
//! massive safety net for the developer. It's possible that an accumulator
//! could store tens, hundreds, even thousands of errors. To drop an accumulator
//! and lose them silently and implicitly could be catastrophic.
//!
//! # Examples
//!
//! Basic usage:
//!
//! ```
//! use fused_error::Accumulator;
//!
//! // Note that the turbofish operator is typically required for `new`:
//! let mut acc = Accumulator::<String>::new();
//!
//! // Any method that collects errors performs conversions identical to the
//! // question mark operator, `?`:
//! acc.push(String::from("foo"));
//! acc.push("bar");
//!
//! assert_eq!(acc.len(), 2);
//!
//! // Conditionally push errors as further diagnostics only if necessary. If
//! // the accumulator was empty, this would do nothing:
//! acc.trace("baz");
//!
//! assert_eq!(acc.len(), 3);
//!
//! let results = vec![
//!     Ok(1),
//!     Ok(2),
//!     Err("qux"),
//!     Ok(4),
//!     Err("quux"),
//! ];
//!
//! let mut sum = 0;
//!
//! for res in results {
//!     // Easily unwrap and handle results. Look at `IteratorExt::accumulate`
//!     // as that might fit your needs better with iterators:
//!     if let Some(n) = acc.handle(res) {
//!         sum += n;
//!     }
//! }
//!
//! assert_eq!(sum, 7);
//! assert_eq!(acc.len(), 5);
//!
//! // `into_vec` is one of the methods that terminate an accumulator properly.
//! // Therefore, no panicking occurs:
//! assert_eq!(acc.into_vec(), ["foo", "bar", "baz", "qux", "quux"]);
//! ```
//!
//! Using an error that implements `FusedError`:
//!
//! ```
//! use fused_error::{Accumulated, Accumulator};
//!
//! let mut acc = Accumulator::<Accumulated<&str>>::new();
//! assert_eq!(acc.finish(), Ok(()));
//!
//! let mut acc = Accumulator::<Accumulated<&str>>::new();
//! acc.push("foo");
//! acc.push("bar");
//! assert_eq!(acc.finish().unwrap_err(), ["foo", "bar"]);
//!
//! let mut acc = Accumulator::<Accumulated<&str>>::new();
//! assert_eq!(acc.err_or(0), Ok(0));
//!
//! fn using_checkpoint() -> Result<(), Accumulated<&'static str>> {
//!     let mut acc = Accumulator::<Accumulated<&str>>::new();
//!     acc.push("baz");
//!     acc.push("qux");
//!
//!     // This is shorthand for calling `finish` and making a new accumulator:
//!     let mut acc = acc.checkpoint()?;
//!
//!     unreachable!()
//! }
//!
//! assert_eq!(using_checkpoint().unwrap_err(), ["baz", "qux"])
//! ```
//!
//! The following uses the `syn` feature to implement [`FusedError`] on
//! [`syn::Error`]:
//!
//! ```
//! # extern crate proc_macro;
//! use fused_error::{Accumulator, FusedError};
//! use proc_macro::TokenStream;
//! use proc_macro2::TokenStream as TokenStream2;
//! use syn::{AttributeArgs, DeriveInput, ItemFn};
//!
//! # const IGNORE: &str = stringify! {
//! #[proc_macro_attribute]
//! # };
//! pub fn my_attribute(args: TokenStream, input: TokenStream) -> TokenStream {
//!     fn inner(args: TokenStream, input: TokenStream) -> syn::Result<TokenStream2> {
//!         let mut acc = Accumulator::<syn::Error>::new();
//!         // fn(TokenStream) -> syn::Result<AttributeArgs>
//!         let args = acc.handle(parse_args(args));
//!         let item = acc.handle(syn::parse::<ItemFn>(input));
//!
//!         // Throw all of the collected parse errors, or continue:
//!         acc = acc.checkpoint()?;
//!
//!         // SAFETY: any `None` value would short-circuit at the last
//!         // checkpoint.
//!         let mut args = unsafe { args.unwrap_unchecked() };
//!         let mut item = unsafe { item.unwrap_unchecked() };
//!
//!         // fn(&AttributeArgs) -> syn::Result<()>
//!         acc.handle(validate_args(&args));
//!         // fn(&ItemFn) -> syn::Result<()>
//!         acc.handle(validate_item(&item));
//!
//!         // Throw all of the validation parse errors, or continue:
//!         acc = acc.checkpoint()?;
//!
//!         // Do multiple steps that can short-circuit:
//!         let tokens = acc.handle_in(|| {
//!             // fn(&mut AttributeArgs, &mut ItemFn) -> syn::Result<()>
//!             prepare(&mut args, &mut item)?;
//!             // fn(AttributeArgs, ItemFn) -> syn::Result<TokenStream2>
//!             expand(args, item)
//!         });
//!
//!         // If this closure is called, we know `tokens` is `Some`:
//!         acc.err_or_else(|| unsafe { tokens.unwrap_unchecked() })
//!     }
//!
//!     inner(args, input)
//!         .unwrap_or_else(syn::Error::into_compile_error)
//!         .into()
//! }
//! # fn parse_args(args: TokenStream) -> syn::Result<AttributeArgs> {
//! #     todo!()
//! # }
//! # fn validate_args(args: &AttributeArgs) -> syn::Result<()> {
//! #     todo!()
//! # }
//! #
//! # fn validate_item(item: &ItemFn) -> syn::Result<()> {
//! #     todo!()
//! # }
//! #
//! # fn prepare(args: &mut AttributeArgs, item: &mut ItemFn) -> syn::Result<()> {
//! #     todo!()
//! # }
//! #
//! # fn expand(args: AttributeArgs, item: ItemFn) -> syn::Result<TokenStream2> {
//! #     todo!()
//! # }
//! ```
//!
//! # Method Overview
//!
//! Here is a basic outline of the accumulator methods at your disposal
//! presented in the consecutive stages they'll most likely get called in.
//!
//! ## Creation
//!
//! To create an accumulator, use one of the following functions:
//!
//! * [`new`] to create an empty accumulator
//! * [`from_vec`] if a vector of errors already exists
//! * [`from_iter`](FromIterator::from_iter) (through the [`FromIterator`]
//!   trait)
//!
//! ## Accumulation
//!
//! All functions concerning the actual collection of errors are generic such
//! that they accept any input of `IE`, which is any type that implements
//! [`Into<E>`]. This is to reflect how the [`?`](std::ops::Try) operator
//! already works.
//!
//! To collect an error into the accumulator, use one of the following:
//!
//! * [`push`] appends a single error
//! * [`extend`](Extend::extend) (through the [`Extend`] trait)
//!
//! To append errors only if errors have already been collected, such as
//! sub-diagnostics, use one of the following:
//!
//! * [`trace`] which conditionally calls [`push`]
//! * [`trace_with`] to lazily [`trace`]
//!
//! To unwrap and handle results, use one of the following:
//!
//! * [`handle`] to handle any type that implements [`IntoResultParts`]
//! * [`handle_in`] to collect from closures that use [`?`]
//!
//! ## Transformation
//!
//! Accumulators are, in "simplified" terms, protected vectors that validate
//! their emptiness on drop with some utility methods. This, however, means that
//! most operations should be thought of as iterators.
//!
//! For instance, a convoluted "map" method does not exist, because operating
//! with iterator adapters like the following is encouraged instead:
//!
//! ```
//! use fused_error::Accumulator;
//!
//! let mut acc = Accumulator::<&str>::new();
//! acc.push("foo");
//! acc.push("bar");
//!
//! let mut acc: Accumulator<String> = acc
//!     .into_iter()
//!     .enumerate()
//!     .map(|(i, s)| format!("{i}: {s}"))
//!     .collect();
//!
//! // Note that we can still use `&str` as an input, because &str
//! // implements Into<String>.
//! acc.push("2: baz");
//!
//! assert_eq!(acc.into_vec(), ["0: foo", "1: bar", "2: baz"])
//! ```
//!
//! ## Termination
//!
//! **One of these methods *must* get called at the end of the accumulator's
//! lifetime. It will panic otherwise.** For more information, look above at the
//! "Panics" section.
//!
//! If the error type **does not** implement [`FusedError`], there are only
//! two methods to properly handle an accumulator:
//!
//! * [`into_vec`] which returns a vector of the collected errors
//! * [`ignore`] which is considered unsafe because it silently discards all
//!   errors
//!
//! However, if the error type **does** implement [`FusedError`], you can use
//! one of the following:
//!
//! * [`finish`] which returns `Result<(), E>`
//! * [`err`] which returns `Option<E>`
//! * [`err_or`] which returns `Result<T, E>`
//! * [`err_or_else`] which is the lazy version of [`err_or`]
//!
//! Then, if the error type **does** implement [`FusedError`], you can use
//! [`checkpoint`] as shorthand for calling [`finish`] and then [`new`]. This
//! isn't listed above, though, since it still returns another accumulator you
//! have to handle.
//!
//! # Types
//!
//! In most instances, signatures concerning accumulators are purposefully
//! generic. This design choice stems from the fact that Rust developers are
//! already used to the question mark operator ([`?`]) and most error
//! conversions happening implicitly. It's already enough technical debt to
//! rethink errors and results as non-binary; there's no need to introduce
//! conversion friction at call sites when the cost is just the odd 'turbofish'
//! (`::<>`), often only when [`new`] is called.
//!
//! [`?`]: std::ops::Try
//!
//! [`new`]: Accumulator::new
//! [`from_vec`]: Accumulator::from_vec
//! [`len`]: Accumulator::len
//! [`is_empty`]: Accumulator::is_empty
//! [`iter`]: Accumulator::iter
//! [`iter_mut`]: Accumulator::iter_mut
//! [`push`]: Accumulator::push
//! [`trace`]: Accumulator::trace
//! [`trace_with`]: Accumulator::trace_with
//! [`handle`]: Accumulator::handle
//! [`handle_in`]: Accumulator::handle_in
//! [`into_vec`]: Accumulator::into_vec
//! [`ignore`]: Accumulator::ignore
//! [`finish`]: Accumulator::finish
//! [`err`]: Accumulator::err
//! [`err_or`]: Accumulator::err_or
//! [`err_or_else`]: Accumulator::err_or_else
//! [`checkpoint`]: Accumulator::checkpoint

use crate::{FusedError, IntoResultParts};

use std::fmt::{self, Debug};

mod raw;

/// An error accumulator.
///
/// See the [module documentation] for details.
///
/// # Panics
///
/// **Accumulators panic on drop if not handled.** Be sure to read the "Panics"
/// section in the [module documentation] for details.
///
/// [module documentation]: self
#[must_use = "accumulators will panic on drop if not handled"]
pub struct Accumulator<E> {
    // This is `pub(crate)` for low-level access by result::raw::FusedResult.
    pub(crate) inner: raw::Accumulator<E>,
}

impl<E> Accumulator<E> {
    /// Constructs a new, empty accumulator.
    ///
    /// Because accumulators are so generalized, it can cause problems with
    /// type inference. As such, `new` should probably get called with the
    /// 'turbofish' syntax: `::<>`. This helps the inference algorithm
    /// understand specifically which error type you're accumulating.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::Accumulator;
    ///
    /// let mut acc = Accumulator::<&str>::new();
    ///
    /// // ...
    /// # unsafe { acc.ignore() };
    /// ```
    ///
    /// *Note:* You may have noticed the trailing ellipsis comment. This is
    /// because, as outlined in the documentation for [`Accumulator<E>`], any
    /// unhandled accumulator will panic on drop. That comment in any example
    /// is meant to signify the accumulator getting handled at a later point in
    /// the program.
    #[inline]
    pub fn new() -> Self {
        let inner = raw::Accumulator::new();
        Accumulator { inner }
    }

    /// Constructs a new accumulator from a vector of errors.
    ///
    /// Unlike [`new`](Accumulator::new), it's very unlikely that the type
    /// inference algorithm can't guess the error type you're meaning to
    /// accumulate.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::Accumulator;
    ///
    /// let vec = vec!["foo", "bar"];
    /// let mut acc = Accumulator::from_vec(vec);
    ///
    /// // ...
    /// # unsafe { acc.ignore() };
    /// ```
    #[inline]
    pub fn from_vec(vec: Vec<E>) -> Self {
        let inner = raw::Accumulator::from_vec(vec);
        Accumulator { inner }
    }

    /// Returns the number of errors in the accumulator.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::Accumulator;
    ///
    /// let vec = vec!["foo", "bar"];
    /// let mut acc = Accumulator::from_vec(vec);
    /// assert_eq!(acc.len(), 2);
    ///
    /// // ...
    /// # unsafe { acc.ignore() };
    /// ```
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.as_ref().len()
    }

    /// Returns `true` if the accumulator contains no errors.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::Accumulator;
    ///
    /// let mut acc = Accumulator::<&str>::new();
    /// assert!(acc.is_empty());
    ///
    /// acc.push("foo");
    /// assert!(!acc.is_empty());
    ///
    /// // ...
    /// # unsafe { acc.ignore() };
    /// ```
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns an iterator over the accumulated errors.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::Accumulator;
    ///
    /// let vec = vec!["foo", "bar"];
    /// let mut acc = Accumulator::from_vec(vec);
    /// let mut iter = acc.iter();
    ///
    /// assert_eq!(iter.next(), Some(&"foo"));
    /// assert_eq!(iter.next(), Some(&"bar"));
    /// assert_eq!(iter.next(), None);
    ///
    /// // ...
    /// # unsafe { acc.ignore() };
    /// ```
    #[inline]
    pub fn iter(&self) -> Iter<'_, E> {
        self.inner.as_ref().iter()
    }

    /// Returns an iterator of the accumulated errors that allows modifying each
    /// value.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::Accumulator;
    ///
    /// let vec = vec![" foo\n", " bar\n"];
    /// let mut acc = Accumulator::from_vec(vec);
    ///
    /// for err in acc.iter_mut() {
    ///     *err = err.trim()
    /// }
    ///
    /// assert_eq!(acc.into_vec(), ["foo", "bar"]);
    /// ```
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, E> {
        self.inner.as_mut().iter_mut()
    }

    /// Pushes an error into the accumulator.
    ///
    /// To emulate the [question mark operator's](std::ops::Try) behavior of
    /// performing the necessary conversions behind the scenes, `push` accepts
    /// any type that can get converted into the error type the accumulator is
    /// collecting.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity of the accumulator exceeds `isize::MAX`
    /// bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::Accumulator;
    ///
    /// // We're collecting `String` errors, so `push` can except anything that
    /// // implements `Into<String>`.
    /// let mut acc = Accumulator::<String>::new();
    ///
    /// acc.push(String::from("foo"));
    /// acc.push("bar");
    ///
    /// assert_eq!(acc.into_vec(), ["foo", "bar"]);
    /// ```
    ///
    /// If you have to push an `Option<E>`, consider taking advantage of the
    /// [`Extend`] implementation and the fact that `Option` implements
    /// [`IntoIterator`] instead:
    ///
    /// ```
    /// use fused_error::Accumulator;
    ///
    /// let mut acc = Accumulator::<String>::new();
    ///
    /// let err1 = Some("foo");
    /// let err2: Option<String> = None;
    /// let err3 = Some("baz");
    /// let err4 = Some(String::from("qux"));
    ///
    /// // This is what you might be inclined to do:
    /// if let Some(e) = err1 {
    ///     acc.push(e);
    /// }
    ///
    /// // Instead, you can just do this:
    /// acc.extend(err2);
    /// acc.extend(err3);
    /// acc.extend(err4);
    ///
    /// assert_eq!(acc.into_vec(), ["foo", "baz", "qux"]);
    /// ```
    #[inline]
    #[track_caller]
    pub fn push<IE>(&mut self, err: IE)
    where
        IE: Into<E>,
    {
        self.inner.push(err.into());
    }

    /// Pushes an error only if the accumulator **is not** empty.
    ///
    /// Arguments passed to `trace` are eagerly evaluated; if you are passing
    /// the result of a function call, it is recommended to use
    /// [`trace_with`](Accumulator::trace_with), which is lazily evaluated.
    ///
    /// To emulate the [question mark operator's](std::ops::Try) behavior of
    /// performing the necessary conversions behind the scenes, `trace` accepts
    /// any type that can get converted into the error type the accumulator is
    /// collecting.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity of the accumulator exceeds `isize::MAX`
    /// bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::Accumulator;
    ///
    /// let mut acc = Accumulator::<String>::new();
    /// acc.trace(String::from("foo"));
    /// assert!(acc.is_empty());
    ///
    /// acc.push("bar");
    /// acc.trace("baz");
    /// assert_eq!(acc.into_vec(), ["bar", "baz"]);
    /// ```
    #[inline]
    #[track_caller]
    pub fn trace<IE>(&mut self, err: IE)
    where
        IE: Into<E>,
    {
        if !self.is_empty() {
            self.push(err);
        }
    }

    /// Emulates [`extend`] but only if the accumulator **is not** empty.
    ///
    /// To emulate the [question mark operator's](std::ops::Try) behavior of
    /// performing the necessary conversions behind the scenes, `trace_iter`
    /// accepts any iterator whose item's type can get converted into the error
    /// type the accumulator is collecting.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity of the accumulator exceeds `isize::MAX`
    /// bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::Accumulator;
    ///
    /// let mut acc = Accumulator::<&str>::new();
    ///
    /// acc.trace_iter(["foo", "bar"]);
    /// assert!(acc.is_empty());
    ///
    /// acc.push("baz");
    /// acc.trace_iter(["qux", "quux"]);
    ///
    /// assert_eq!(acc.into_vec(), ["baz", "qux", "quux"]);
    /// ```
    ///
    /// Similar to using [`extend`], consider using `trace_iter`:
    ///
    /// ```
    /// use fused_error::Accumulator;
    ///
    /// let mut acc = Accumulator::<&str>::new();
    /// acc.push("foo");
    ///
    /// let trace1 = Some("bar");
    /// let trace2: Option<&str> = None;
    /// let trace3 = Some("qux");
    ///
    /// // This is what you might be inclined to do:
    /// if let Some(e) = trace1 {
    ///     acc.trace(e);
    /// }
    ///
    /// // Instead, you can just do this:
    /// acc.trace_iter(trace2);
    /// acc.trace_iter(trace3);
    ///
    /// assert_eq!(acc.into_vec(), ["foo", "bar", "qux"]);
    /// ```
    ///
    /// [`extend`]: Extend::extend
    #[inline]
    #[track_caller]
    pub fn trace_iter<IE, I>(&mut self, iter: I)
    where
        IE: Into<E>,
        I: IntoIterator<Item = IE>,
    {
        if !self.is_empty() {
            iter.into_iter().for_each(|err| self.push(err));
        }
    }

    /// Calls `f` if there are any errors in the accumulator and collects the
    /// returned error.
    ///
    /// To emulate the [question mark operator's](std::ops::Try) behavior of
    /// performing the necessary conversions behind the scenes, `f` can return
    /// any type that can get converted into the error type the accumulator is
    /// collecting.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity of the accumulator exceeds `isize::MAX`
    /// bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::Accumulator;
    ///
    /// let mut acc = Accumulator::<String>::new();
    /// acc.trace_with(||"foo");
    /// assert!(acc.is_empty());
    ///
    /// acc.push("bar");
    /// acc.trace_with(|| "baz");
    /// assert_eq!(acc.into_vec(), ["bar", "baz"]);
    /// ```
    #[inline]
    pub fn trace_with<IE, F>(&mut self, f: F)
    where
        IE: Into<E>,
        F: FnOnce() -> IE,
    {
        if !self.is_empty() {
            self.push(f());
        }
    }

    /// Handles a result, collecting the error and returning any "ok" value, if
    /// present.
    ///
    /// If you are working with iterators, whenever possible, prefer
    /// [`IteratorExt::accumulate`](crate::IteratorExt::accumulate).
    ///
    /// This method is incredibly versatile. A result **does not** have to
    /// necessarily be a [`Result<T, IE>`](Result) (where `IE` is any type that
    /// implements [`Into<E>`]). Instead, it just has to implement
    /// [`IntoResultParts`].
    ///
    /// # Panics
    ///
    /// Panics if the new capacity of the accumulator exceeds `isize::MAX`
    /// bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::ParseIntError;
    /// use fused_error::Accumulator;
    ///
    /// let mut acc = Accumulator::<ParseIntError>::new();
    /// assert_eq!(acc.handle("1".parse::<i32>()), Some(1));
    ///
    /// assert_eq!(acc.handle("invalid".parse::<i32>()), None);
    /// assert_eq!(acc.len(), 1);
    ///
    /// // ...
    /// # unsafe { acc.ignore() };
    /// ```
    #[inline]
    pub fn handle<R>(&mut self, res: R) -> Option<<R as IntoResultParts>::Ok>
    where
        R: IntoResultParts,
        <R as IntoResultParts>::Err: Into<E>,
    {
        let (ok, err) = res.into_result_parts();
        if let Some(err) = err {
            self.push(err.into());
        }
        ok
    }

    /// Calls `f`, returning the successful value as `Some`, or collecting the
    /// error and returning `None`.
    ///
    /// Because the closure's return type is a result, you can use the
    /// question mark operator inside of it: [`?`](std::ops::Try).
    ///
    /// Unlike `handle`, this method only accepts results instead of any type
    /// that implements [`IntoResultParts`]. This is because
    /// [`Try`](std::ops::Try) is not on stable and developer ergonomics
    /// would suffer immensely otherwise.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity of the accumulator exceeds `isize::MAX`
    /// bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::ParseIntError;
    /// use fused_error::Accumulator;
    ///
    /// let mut acc = Accumulator::<ParseIntError>::new();
    ///
    /// let sum = acc.handle_in(|| {
    ///     // All of these inputs are valid. No errors expected.
    ///     let a: i32 = "1".parse()?;
    ///     let b: i32 = "2".parse()?;
    ///     let c: i32 = "3".parse()?;
    ///     Ok(a + b + c)
    /// });
    /// assert_eq!(sum, Some(6));
    /// assert!(acc.is_empty());
    ///
    /// let product = acc.handle_in(|| {
    ///     // All inputs are invalid. Because of how `?` works, `a` will
    ///     // short-circuit the closure.
    ///     let a: i32 = "foo".parse()?;
    ///     let b: i32 = "bar".parse()?;
    ///     let c: i32 = "baz".parse()?;
    ///     Ok(a * b * c)
    /// });
    /// assert!(product.is_none());
    /// assert_eq!(acc.len(), 1);
    ///
    /// // ...
    /// # unsafe { acc.ignore() };
    /// ```
    #[inline]
    pub fn handle_in<T, F>(&mut self, f: F) -> Option<T>
    where
        F: FnOnce() -> Result<T, E>,
    {
        self.handle(f())
    }

    /// Extracts the vector of collected errors.
    ///
    /// Calling this method ensures the accumulator **will not** panic on drop.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::Accumulator;
    ///
    /// let mut acc = Accumulator::<&str>::new();
    /// acc.push("foo");
    /// acc.push("bar");
    ///
    /// assert_eq!(acc.into_vec(), ["foo", "bar"]);
    /// ```
    #[must_use = "if you want to ignore the accumulator, consider `.ignore()` instead"]
    #[inline]
    pub fn into_vec(mut self) -> Vec<E> {
        // SAFETY: accumulator is dropped immediately after
        unsafe { self.inner.take() }
    }

    /// Handles this accumulator, discarding all errors.
    ///
    /// Calling this method ensures the accumulator **will not** panic on drop.
    ///
    /// # Safety
    ///
    /// It is considered semantically unsafe to discard errors, especially an
    /// accumulator due to the volume of errors they can store.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::Accumulator;
    ///
    /// let mut acc = Accumulator::<&str>::new();
    /// acc.push("foo");
    /// acc.push("bar");
    ///
    /// // TODO: properly handle errors
    /// unsafe { acc.ignore() };
    /// ```
    #[inline]
    pub unsafe fn ignore(mut self) {
        // SAFETY: accumulator is dropped immediately after and discarding is
        // ensured to be semantically safe by the caller.
        self.inner.ignore();
    }

    /// Returns `true` if further operations on the accumulator are safe.
    #[must_use]
    #[inline]
    pub(crate) fn is_handled(&self) -> bool {
        self.inner.is_handled()
    }
}

impl<E> Accumulator<E>
where
    E: FusedError,
{
    /// Returns `Ok(())` if the accumulator is empty, otherwise reduces the
    /// [`FusedError`] type into `Err(E)`.
    ///
    /// Calling this method ensures the accumulator **will not** panic on drop.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::{Accumulated, Accumulator};
    ///
    /// type Error = Accumulated<String>;
    ///
    /// let mut acc = Accumulator::<Error>::new();
    /// assert_eq!(acc.finish(), Ok(()));
    ///
    /// let mut acc = Accumulator::<Error>::new();
    /// acc.push("foo".to_string());
    /// acc.push("bar".to_string());
    /// assert_eq!(acc.finish().unwrap_err(), ["foo", "bar"]);
    /// ```
    pub fn finish(self) -> Result<(), E> {
        self.err_or(())
    }

    /// Returns `None` if the accumulator is empty, otherwise reduces the
    /// [`FusedError`] type into `Some(E)`.
    ///
    /// Calling this method ensures the accumulator **will not** panic on drop.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::{Accumulated, Accumulator};
    ///
    /// let mut acc = Accumulator::<Accumulated<&str>>::new();
    /// assert_eq!(acc.err(), None);
    ///
    /// let mut acc = Accumulator::<Accumulated<&str>>::new();
    /// acc.push("foo");
    /// acc.push("bar");
    /// assert_eq!(acc.err().unwrap(), ["foo", "bar"]);
    /// ```
    #[must_use]
    #[inline]
    pub fn err(mut self) -> Option<E> {
        // SAFETY: accumulator is dropped immediately after
        unsafe { self.inner.reduce() }
    }

    /// Returns `Ok(ok)` if the accumulator is empty, otherwise reduces the
    /// [`FusedError`] type into `Err(E)`.
    ///
    /// Calling this method ensures the accumulator **will not** panic on drop.
    ///
    /// Arguments passed to `err_or` are eagerly evaluated; if you are passing
    /// the result of a function call, it is recommended to use
    /// [`err_or_else`](Accumulator::err_or_else), which is lazily evaluated.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::{Accumulated, Accumulator};
    ///
    /// let mut acc = Accumulator::<Accumulated<&str>>::new();
    /// assert_eq!(acc.err_or(0), Ok(0));
    ///
    /// let mut acc = Accumulator::<Accumulated<&str>>::new();
    /// acc.push("foo");
    /// acc.push("bar");
    /// assert_eq!(acc.err_or(0).unwrap_err(), ["foo", "bar"]);
    /// ```
    #[inline]
    pub fn err_or<T>(self, ok: T) -> Result<T, E> {
        match self.err() {
            None => Ok(ok),
            Some(err) => Err(err),
        }
    }

    /// Returns `Ok(f())` if the accumulator is empty, otherwise reduces the
    /// [`FusedError`] type into `Err(E)`.
    ///
    /// Calling this method ensures the accumulator **will not** panic on drop.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::{Accumulated, Accumulator};
    ///
    /// let mut acc = Accumulator::<Accumulated<&str>>::new();
    /// assert_eq!(acc.err_or_else(|| 0), Ok(0));
    ///
    /// let mut acc = Accumulator::<Accumulated<&str>>::new();
    /// acc.push("foo");
    /// acc.push("bar");
    /// assert_eq!(acc.err_or_else(|| 0).unwrap_err(), ["foo", "bar"]);
    /// ```
    #[inline]
    pub fn err_or_else<T, F>(self, f: F) -> Result<T, E>
    where
        F: FnOnce() -> T,
    {
        match self.err() {
            None => Ok(f()),
            Some(err) => Err(err),
        }
    }

    /// Returns `Ok(Accumulator<E>)` if the accumulator is empty, otherwise
    /// reduces the [`FusedError`] type into `Err(E)`.
    ///
    /// This method is particularly useful for short-circuiting with the
    /// [`?`](std::ops::Try) operator.
    ///
    /// # Examples
    ///
    /// ```should_panic
    /// use fused_error::{Accumulated, Accumulator};
    ///
    /// # fn main() -> Result<(), Accumulated<String>> {
    /// let mut acc = Accumulator::<Accumulated<String>>::new();
    /// let a = acc.handle("200".parse::<u8>().map_err(|e| e.to_string()));
    /// let b = acc.handle("100".parse::<u8>().map_err(|e| e.to_string()));
    ///
    /// acc = acc.checkpoint()?;
    ///
    /// // SAFETY: We know all past "handle" calls have returned as successes.
    /// let a: u8 = unsafe { a.unwrap_unchecked() };
    /// let b: u8 = unsafe { b.unwrap_unchecked() };
    ///
    /// // These will both error:
    /// let sum = acc.handle(a.checked_add(b).ok_or("addition overflow".to_string()));
    /// let product = acc.handle(a.checked_mul(b).ok_or("multiplication overflow".to_string()));
    ///
    /// // This will short-circuit with the following error:
    /// //
    /// // Error {
    /// //     messages: [
    /// //         "addition overflow",
    /// //         "multiplication overflow",
    /// //     ],
    /// // }
    /// acc = acc.checkpoint()?;
    ///
    /// // ...
    /// # unsafe { acc.ignore(); }
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn checkpoint(self) -> Result<Self, E> {
        self.err_or_else(Accumulator::new)
    }
}

////////////////////////////////////////////////////////////////////////////////
// Trait implementations
////////////////////////////////////////////////////////////////////////////////

impl<E> Debug for Accumulator<E>
where
    E: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Accumulator").field(&self.inner).finish()
    }
}

impl<E> Default for Accumulator<E> {
    #[inline]
    fn default() -> Self {
        Accumulator::new()
    }
}

////////////////////////////////////////////////////////////////////////////////
// Iterator traits
////////////////////////////////////////////////////////////////////////////////

impl<E> FromIterator<E> for Accumulator<E> {
    #[inline]
    fn from_iter<T: IntoIterator<Item = E>>(iter: T) -> Self {
        let vec: Vec<E> = iter.into_iter().collect();
        Accumulator::from_vec(vec)
    }
}

impl<E, IE> Extend<IE> for Accumulator<E>
where
    IE: Into<E>,
{
    #[inline]
    fn extend<T: IntoIterator<Item = IE>>(&mut self, iter: T) {
        self.inner.as_mut().extend(iter.into_iter().map(Into::into));
    }
}

impl<E> IntoIterator for Accumulator<E> {
    type Item = E;
    type IntoIter = IntoIter<E>;

    /// Calling this method ensures the accumulator **will not** panic on drop.
    #[inline]
    fn into_iter(mut self) -> Self::IntoIter {
        // SAFETY: accumulator is dropped immediately after
        unsafe { self.inner.take() }.into_iter()
    }
}

impl<'a, E> IntoIterator for &'a Accumulator<E> {
    type Item = &'a E;
    type IntoIter = Iter<'a, E>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, E> IntoIterator for &'a mut Accumulator<E> {
    type Item = &'a mut E;
    type IntoIter = IterMut<'a, E>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

////////////////////////////////////////////////////////////////////////////////
// Iterator type aliases
////////////////////////////////////////////////////////////////////////////////

/// Immutable error iterator.
///
/// This is created by the [`iter`](Accumulator::iter) method on
/// [`accumulators`](Accumulator).
pub type Iter<'a, T> = std::slice::Iter<'a, T>;

/// Mutable error iterator.
///
/// This is created by the [`iter_mut`](Accumulator::iter_mut) method on
/// [`accumulators`](Accumulator).
pub type IterMut<'a, T> = std::slice::IterMut<'a, T>;

/// An iterator that moves out of an accumulator.
///
/// This is created by the `into_iter` method on [`accumulators`](Accumulator)
/// (provided by the [`IntoIterator`] trait).
pub type IntoIter<T> = std::vec::IntoIter<T>;
